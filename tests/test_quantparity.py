import json
import math
import pathlib
import random

import pytest

import quantparity as qp

DATA = pathlib.Path(__file__).resolve().parent.parent / "examples" / "data"


@pytest.fixture(scope="module")
def market():
    return json.loads((DATA / "eur_market.json").read_text())


@pytest.fixture(scope="module")
def portfolio():
    return json.loads((DATA / "portfolio.json").read_text())


@pytest.fixture(scope="module")
def history():
    lines = (DATA / "factor_history.csv").read_text().strip().splitlines()
    cols = lines[0].split(",")[1:]
    return {c: [float(l.split(",")[i + 1]) for l in lines[1:]] for i, c in enumerate(cols)}


def test_bsm_hull():
    # Hull, Options, Futures and Other Derivatives: S=42, K=40, r=10%, sigma=20%, T=0.5
    assert qp.bsm("call", 42, 40, 0.10, 0.20, 0.5)["price"] == pytest.approx(4.7594, abs=1e-4)
    assert qp.bsm("put", 42, 40, 0.10, 0.20, 0.5)["price"] == pytest.approx(0.8086, abs=1e-4)


def test_put_call_parity():
    s, k, r, q, v, t = 100.0, 95.0, 0.03, 0.01, 0.25, 1.5
    c = qp.bsm("call", s, k, r, v, t, dividend=q)["price"]
    p = qp.bsm("put", s, k, r, v, t, dividend=q)["price"]
    assert c - p == pytest.approx(s * math.exp(-q * t) - k * math.exp(-r * t), abs=1e-12)


def test_black76_haug():
    assert qp.black76("call", 19, 19, 0.10, 0.28, 0.75)["price"] == pytest.approx(1.7011, abs=1e-4)


def test_crr_hull_tree():
    assert qp.crr("put", 50, 50, 0.10, 0.40, 5 / 12, steps=5, american=True) == pytest.approx(4.49, abs=5e-3)


def test_implied_vol_roundtrip():
    p = qp.bsm("put", 100, 110, 0.02, 0.37, 2.0)["price"]
    assert qp.implied_vol("put", p, 100, 110, 0.02, 2.0) == pytest.approx(0.37, abs=1e-12)


def test_bond_par():
    assert qp.bond(0.04, 10, 0.04, frequency=2)["price"] == pytest.approx(100.0, abs=1e-10)


def test_invalid_input_raises():
    with pytest.raises(ValueError, match="strictly positive"):
        qp.bsm("call", -1, 100, 0.0, 0.2, 1.0)
    with pytest.raises(ValueError, match="option_type"):
        qp.bsm("straddle", 100, 100, 0.0, 0.2, 1.0)


def test_multicurve_reprices_inputs(market):
    m = qp.Market(market)
    assert set(m.curves()) == {"EUR-ESTR", "EUR-6M"}
    report = m.report("EUR-6M")
    assert report["max_repricing_error"] < 1e-10
    for tenor, quote in [("5Y", 0.02489), ("10Y", 0.02742)]:
        v = qp.price({"id": "s", "product": "vanilla_swap", "notional": 1e6, "fixed_rate": quote, "tenor": tenor,
                      "forward_curve": "EUR-6M"}, market)[0]
        assert abs(v["price"]) < 1e-6
        assert v["measures"]["par_rate"] == pytest.approx(quote, abs=1e-12)


def test_portfolio_risk(portfolio, history, market):
    out = qp.revalue(portfolio, history, market)
    assert len(out["scenarios"]) == 300
    pnl = [s["pnl"] for s in out["scenarios"]]
    r = qp.var_es(pnl, 0.99, 0.975)
    assert r["var"] > 0 and r["es"] > 0

    sens = qp.sensitivities(portfolio, market)
    factors = {s["factor"] for s in sens}
    assert {"SPOT:SX5E", "VOL:SX5E", "CURVE:EUR-6M:10Y", "YIELD:OAT"} <= factors


def test_pla_green_for_small_moves(portfolio, history, market):
    out = qp.pla_test(portfolio, history, market)
    assert out["result"]["zone"] == "green"


def test_stress(portfolio, market):
    sc = json.loads((DATA / "stress.json").read_text())["scenarios"]
    res = {r["name"]: r for r in qp.stress(portfolio, sc, market)}
    assert res["Sovereign crisis"]["status"] == "fail"


def test_benchmark_flags_wrong_price():
    trade = {"id": "A", "product": "european_option", "option_type": "call", "spot": 100, "strike": 100,
             "rate": 0.02, "vol": 0.2, "maturity": 1.0}
    ref = qp.price(trade)[0]["price"]
    ok = qp.compare([trade], {"A": ref + 1e-12})
    bad = qp.compare([trade], {"A": ref + 1e-3})
    assert ok["status"] == "pass" and bad["status"] == "fail"


def test_var_backtests():
    k = qp.kupiec(250, 12, 0.99)
    assert k["status"] == "fail"
    assert qp.traffic_light(250, 4, 0.99)["zone"] == "green"
    assert qp.traffic_light(250, 7, 0.99)["zone"] == "yellow"
    assert qp.traffic_light(250, 10, 0.99)["zone"] == "red"


def test_psi_and_credit():
    ref = [i / 1000 for i in range(1000)]
    assert qp.psi(ref, ref, 10)["psi"] == pytest.approx(0.0, abs=1e-12)
    score = [float(i) for i in range(100)]
    default = [i >= 50 for i in range(100)]
    d = qp.discrimination(score, default)
    assert d["auc"] == pytest.approx(1.0) and d["gini"] == pytest.approx(1.0)


def test_var_es_models_and_es_backtest():
    assert qp.norm_inv(0.99) == pytest.approx(2.3263478740408408, abs=1e-15)
    assert qp.t_inv(0.99, 5) == pytest.approx(3.364929998907218, abs=1e-13)
    assert qp.t_cdf(qp.t_inv(0.025, 4), 4) == pytest.approx(0.025, abs=1e-15)
    returns = [0.012 * math.sin(i * 1.7) * (3 if i % 53 == 0 else 1) for i in range(700)]
    normal = qp.var_forecasts(returns)
    t5 = qp.var_forecasts(returns, innovations="t", nu=5)
    fhs = qp.var_forecasts(returns, innovations="fhs", min_residuals=250)
    assert (normal["model"], t5["model"], fhs["model"]) == ("ewma-normal", "ewma-t", "fhs")
    assert normal["sigma"] == t5["sigma"] and fhs["start"] == 280
    # Same volatility, fatter tail: higher VaR 99% and ES
    assert all(b > a for a, b in zip(normal["var"], t5["var"]))
    assert all(b > a for a, b in zip(normal["es"], t5["es"]))
    r = returns[normal["start"]:]
    b = qp.es_backtest(r, normal["es_var"], normal["es"], 0.975)
    assert b["observations"] == len(r) and b["zone"] in ("green", "yellow", "red")
    with pytest.raises(ValueError):
        qp.var_forecasts(returns, innovations="t")
    with pytest.raises(ValueError):
        qp.var_forecasts(returns, innovations="t", nu=2)


def test_z2_simulation():
    rng = random.Random(5)
    returns = [0.01 * rng.gauss(0, 1) for _ in range(600)]
    f = qp.var_forecasts(returns, innovations="t", nu=4)
    r = returns[f["start"]:]
    z2 = qp.es_backtest(r, f["es_var"], f["es"], 0.975)["z2"]
    s = qp.z2_simulation(returns, f, z2, simulations=2000, seed=3)
    assert 0 < s["p_value"] <= 1 and s["observations"] == len(r) and s["zone"] in ("green", "yellow", "red")
    assert s == qp.z2_simulation(returns, f, z2, simulations=2000, seed=3)
    w = qp.z2_simulation(returns, f, -0.7, from_=300, to=550, simulations=2000)
    assert w["observations"] == 250 and w["critical_5"] < 0
    f["es"][0] *= 2
    with pytest.raises(ValueError):
        qp.z2_simulation(returns, f, z2)


def test_garch_filter():
    rng = random.Random(7)
    s2, returns = 1e-4, []
    for _ in range(1500):
        r = math.sqrt(s2) * rng.gauss(0, 1)
        s2 = 2e-6 + (0.03 + 0.12 * (r < 0)) * r * r + 0.88 * s2
        returns.append(r)
    fit = qp.garch_fit(returns, asymmetric=True)
    p = fit["params"]
    assert fit["model"] == "gjr" and p["gamma"] > p["alpha"] and fit["persistence"] < 1
    ll, var = qp.garch_loglik(returns, p["omega"], p["alpha"], p["gamma"], p["beta"], fit["backcast"])
    assert abs(ll - fit["loglik"]) < 1e-9 and len(var) == len(returns) + 1
    f = qp.var_forecasts(returns, warmup=500, volatility="gjr", refit=500)
    assert f["model"] == "gjr-normal" and [x["from"] for x in f["fits"]] == [500, 1000]
    assert f["fits"][0]["fit"] == qp.garch_fit(returns[:500], asymmetric=True)
    assert "lambda" not in f and f["volatility"]["kind"] == "garch"
    with pytest.raises(ValueError):
        qp.var_forecasts(returns, volatility="egarch")
