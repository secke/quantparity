"""Cross-checks quantparity against QuantLib on parameter grids and writes validation/RESULTS.md.

    pip install quantparity QuantLib==1.44
    python validation/validate_quantlib.py

Every number in the README comes from this script. It is deterministic.
"""
import itertools
import json
import math
import pathlib
import sys
import time

import QuantLib as ql

import quantparity as qp

ROOT = pathlib.Path(__file__).resolve().parent.parent
TODAY = ql.Date(2, 1, 2026)
ql.Settings.instance().evaluationDate = TODAY
DC = ql.Actual365Fixed()
CAL = ql.NullCalendar()

rows = []  # (area, case count, quantity, max abs diff, max rel diff, note)


failures = []


def record(area, quantity, pairs, note="", limit=1e-10):
    """pairs: list of (quantparity value, QuantLib value); `limit` is the max abs diff accepted by --check."""
    abs_d = max(abs(a - b) for a, b in pairs)
    rel_d = max(abs(a - b) / max(abs(b), 1e-6) for a, b in pairs)
    rows.append((area, len(pairs), quantity, abs_d, rel_d, note))
    if not abs_d <= limit:
        failures.append(f"{area} / {quantity}: {abs_d:.2e} > {limit:.0e}")


def ql_type(t):
    return ql.Option.Call if t == "call" else ql.Option.Put


def flat(r):
    return ql.YieldTermStructureHandle(ql.FlatForward(TODAY, r, DC))


def process(s, r, q, vol):
    return ql.BlackScholesMertonProcess(
        ql.QuoteHandle(ql.SimpleQuote(s)), flat(q), flat(r),
        ql.BlackVolTermStructureHandle(ql.BlackConstantVol(TODAY, CAL, vol, DC)),
    )


def ql_option(t, s, k, r, q, vol, days, engine, american=False):
    ex = ql.AmericanExercise(TODAY, TODAY + days) if american else ql.EuropeanExercise(TODAY + days)
    opt = ql.VanillaOption(ql.PlainVanillaPayoff(ql_type(t), k), ex)
    opt.setPricingEngine(engine(process(s, r, q, vol)))
    return opt


# ------------------------------------------------------------------ 1. Black-Scholes-Merton and Greeks
grid = list(itertools.product(
    ["call", "put"], [80.0, 100.0, 125.0], [100.0], [0.0, 0.03, 0.08], [0.0, 0.02], [0.1, 0.25, 0.6],
    [30, 182, 365, 1825],
))
res = {k: [] for k in ["price", "delta", "gamma", "vega", "theta", "rho"]}
for t, s, k, r, q, vol, days in grid:
    T = days / 365
    g = qp.bsm(t, s, k, r, vol, T, dividend=q)
    o = ql_option(t, s, k, r, q, vol, days, ql.AnalyticEuropeanEngine)
    for name, ref in [("price", o.NPV()), ("delta", o.delta()), ("gamma", o.gamma()), ("vega", o.vega()),
                      ("theta", o.theta()), ("rho", o.rho())]:
        res[name].append((g[name], ref))
for name, pairs in res.items():
    record("European option (BSM)", name, pairs, "AnalyticEuropeanEngine" if name == "price" else "", limit=1e-11)

# ------------------------------------------------------------------ 2. Black-76
pairs = []
for t, f, k, r, vol, T in itertools.product(["call", "put"], [19.0, 78.5, 5100.0], [0.9, 1.0, 1.15],
                                             [0.01, 0.045, 0.10], [0.15, 0.35], [0.25, 0.75, 2.0]):
    pairs.append((qp.black76(t, f, f * k, r, vol, T)["price"],
                  ql.blackFormula(ql_type(t), f * k, f, vol * math.sqrt(T), math.exp(-r * T))))
record("Option on forward (Black-76)", "price", pairs, "blackFormula", limit=1e-11)

# ------------------------------------------------------------------ 3. Implied volatility
pairs = []
for t, s, k, r, vol, T in itertools.product(["call", "put"], [100.0], [70.0, 90.0, 100.0, 110.0, 140.0],
                                             [0.0, 0.03], [0.05, 0.2, 0.5, 1.0], [0.1, 1.0, 5.0]):
    p = qp.bsm(t, s, k, r, vol, T)["price"]
    fwd, df = s * math.exp(r * T), math.exp(-r * T)
    intrinsic = df * max((fwd - k) if t == "call" else (k - fwd), 0.0)
    if p - intrinsic < 1e-6:  # no time value left: volatility is not identifiable
        continue
    ref = ql.blackFormulaImpliedStdDev(ql_type(t), k, fwd, p, df, 0.0, 0.2, 1e-14, 1000) / math.sqrt(T)
    pairs.append((qp.implied_vol(t, p, s, k, r, T), ref))
record("Implied volatility", "vol", pairs, "blackFormulaImpliedStdDev")

# ------------------------------------------------------------------ 4. Fixed-rate bonds
res = {k: [] for k in ["price", "macaulay_duration", "modified_duration", "convexity"]}
for c, years, freq, y in itertools.product([0.0, 0.025, 0.06], [1, 5, 10, 30], [1, 2, 4], [0.01, 0.035, 0.08]):
    m = qp.bond(c, years, y, frequency=freq)
    qf = {1: ql.Annual, 2: ql.Semiannual, 4: ql.Quarterly}[freq]
    sched = ql.Schedule(TODAY, TODAY + ql.Period(years, ql.Years), ql.Period(qf), CAL, ql.Unadjusted, ql.Unadjusted,
                        ql.DateGeneration.Backward, False)
    dc = ql.Thirty360(ql.Thirty360.BondBasis)
    b = ql.FixedRateBond(0, 100.0, sched, [c], dc)
    rate = ql.InterestRate(y, dc, ql.Compounded, qf)
    res["price"].append((m["price"], b.cleanPrice(y, dc, ql.Compounded, qf)))
    res["macaulay_duration"].append((m["macaulay_duration"], ql.BondFunctions.duration(b, rate, ql.Duration.Macaulay)))
    res["modified_duration"].append((m["modified_duration"], ql.BondFunctions.duration(b, rate, ql.Duration.Modified)))
    res["convexity"].append((m["convexity"], ql.BondFunctions.convexity(b, rate)))
for name, pairs in res.items():
    record("Fixed-rate bond", name, pairs, "FixedRateBond, BondFunctions" if name == "price" else "", limit=1e-10)


# ------------------------------------------------------------------ 5. CRR trees: where the 5e-5 comes from
def crr_ql_probability(t, s, k, r, q, vol, T, n, american):
    """CRR with QuantLib's probability p = 1/2 + 1/2 (r - q - sigma^2/2) dt / dx (log-linearised)."""
    dt = T / n
    dx = vol * math.sqrt(dt)
    p = 0.5 + 0.5 * (r - q - 0.5 * vol * vol) * dt / dx
    disc = math.exp(-r * dt)
    sign = 1.0 if t == "call" else -1.0
    v = [max(sign * (s * math.exp((2 * j - n) * dx) - k), 0.0) for j in range(n + 1)]
    for i in range(n - 1, -1, -1):
        v = [disc * (p * v[j + 1] + (1 - p) * v[j]) for j in range(i + 1)]
        if american:
            v = [max(v[j], sign * (s * math.exp((2 * j - i) * dx) - k)) for j in range(i + 1)]
    return v[0]


crr_cases = list(itertools.product(["call", "put"], [90.0, 100.0, 110.0], [0.02, 0.06], [0.0, 0.04], [0.2, 0.4],
                                   [365, 730]))
N = 500
exact, qlprob = [], []
for t, s, r, q, vol, days in crr_cases:
    T = days / 365
    ref = ql_option(t, s, 100.0, r, q, vol, days, lambda p: ql.BinomialVanillaEngine(p, "crr", N), True).NPV()
    exact.append((qp.crr(t, s, 100.0, r, vol, T, steps=N, dividend=q, american=True), ref))
    qlprob.append((crr_ql_probability(t, s, 100.0, r, q, vol, T, N, True), ref))
record("American option (CRR, 500 steps)", "price", exact, "exact martingale p = (e^{(r-q)dt} - d)/(u - d)", limit=2e-3)
record("American option (CRR, 500 steps)", "price", qlprob, "same tree with QuantLib's log-linearised p", limit=1e-9)

conv = []
for n in [50, 100, 200, 500, 1000, 2000]:
    bs = qp.bsm("put", 100.0, 100.0, 0.06, 0.4, 1.0)["price"]
    a = qp.crr("put", 100.0, 100.0, 0.06, 0.4, 1.0, steps=n)
    b = ql_option("put", 100.0, 100.0, 0.06, 0.0, 0.4, 365,
                  lambda p, n=n: ql.BinomialVanillaEngine(p, "crr", n)).NPV()
    conv.append((n, a - bs, b - bs))

# ------------------------------------------------------------------ 6. Multi-curve €STR / Euribor 6M
market = json.loads((ROOT / "examples/data/eur_market.json").read_text())
y, m, d = map(int, market["as_of"].split("-"))
asof = ql.Date(d, m, y)
ql.Settings.instance().evaluationDate = asof
qh = lambda r: ql.QuoteHandle(ql.SimpleQuote(r))
estr_h = ql.RelinkableYieldTermStructureHandle()
ois = ql.PiecewiseLogLinearDiscount(
    asof, [ql.OISRateHelper(2, ql.Period(i["tenor"]), qh(i["rate"]), ql.Estr(estr_h))
           for i in market["curves"]["EUR-ESTR"]["instruments"]], ql.Actual365Fixed())
ois.enableExtrapolation()
estr_h.linkTo(ois)
fixed_dc = ql.Thirty360(ql.Thirty360.BondBasis)
helpers = []
for i in market["curves"]["EUR-6M"]["instruments"]:
    if i["type"] == "deposit":
        helpers.append(ql.DepositRateHelper(qh(i["rate"]), ql.Euribor6M()))
    else:
        helpers.append(ql.SwapRateHelper(qh(i["rate"]), ql.Period(i["tenor"]), ql.TARGET(), ql.Annual,
                                         ql.ModifiedFollowing, fixed_dc, ql.Euribor6M(), ql.QuoteHandle(),
                                         ql.Period(0, ql.Days), estr_h))
proj = ql.PiecewiseLogLinearDiscount(asof, helpers, ql.Actual365Fixed())
proj.enableExtrapolation()
euribor = ql.Euribor6M(ql.YieldTermStructureHandle(proj))
engine = ql.DiscountingSwapEngine(estr_h)
mk = qp.Market(market)

df_pairs = []
for yrs in range(1, 31):
    dd = ql.TARGET().advance(asof, ql.Period(yrs, ql.Years))
    iso = f"{dd.year():04d}-{dd.month():02d}-{dd.dayOfMonth():02d}"
    df_pairs.append((mk.discount("EUR-ESTR", iso), ois.discount(dd)))
record("€STR OIS curve", "discount factor 1Y-30Y", df_pairs, "PiecewiseLogLinearDiscount + OISRateHelper")

npv, par = [], []
notional = 10_000_000
for tenor, k, pay in itertools.product([1, 2, 3, 5, 7, 10, 15, 20, 30], [0.015, 0.025, 0.035], [True, False]):
    s = ql.MakeVanillaSwap(ql.Period(tenor, ql.Years), euribor, k, ql.Period(0, ql.Days), nominal=notional,
                           receiveFixed=not pay, fixedLegTenor=ql.Period(1, ql.Years), fixedLegDayCount=fixed_dc,
                           fixedLegConvention=ql.ModifiedFollowing,
                           fixedLegTerminationDateConvention=ql.ModifiedFollowing, pricingEngine=engine)
    v = qp.price({"id": "s", "product": "vanilla_swap", "notional": notional, "fixed_rate": k, "tenor": f"{tenor}Y",
                  "pay_fixed": pay, "forward_curve": "EUR-6M"}, market)[0]
    npv.append((v["price"], s.NPV()))
    par.append((v["measures"]["par_rate"], s.fairRate()))
record("Euribor 6M swap, €STR discounting", "NPV (10M notional, EUR)", npv, "SwapRateHelper + DiscountingSwapEngine", limit=1e-5)
record("Euribor 6M swap, €STR discounting", "par rate", par, "")

# ------------------------------------------------------------------ 7. Speed
trades = json.loads((ROOT / "examples/data/portfolio.json").read_text())
hist = (ROOT / "examples/data/factor_history.csv").read_text().strip().splitlines()
cols = hist[0].split(",")[1:]
table = {c: [float(l.split(",")[i + 1]) for l in hist[1:]] for i, c in enumerate(cols)}
t0 = time.perf_counter()
out = qp.revalue(trades, table, market)
elapsed = time.perf_counter() - t0
n_sc = len(out["scenarios"])

# ------------------------------------------------------------------ report
def fmt(x):
    return "0" if x == 0 else f"{x:.1e}"


lines = [
    "# Validation against QuantLib",
    "",
    f"Generated by `validation/validate_quantlib.py` with quantparity {qp.__version__} and QuantLib {ql.__version__}.",
    "",
    "Relative differences are taken against max(|QuantLib value|, 1e-6), so that values close to zero",
    "(deep out-of-the-money options) do not inflate them.",
    "",
    "| Area | Cases | Quantity | Max abs diff | Max rel diff | QuantLib reference / note |",
    "|---|---:|---|---:|---:|---|",
]
for area, n, qty, a, r, note in rows:
    lines.append(f"| {area} | {n} | {qty} | {fmt(a)} | {fmt(r)} | {note} |")
lines += [
    "",
    "## CRR: European put, S = K = 100, r = 6%, σ = 40%, T = 1, error vs Black-Scholes",
    "",
    "| Steps | quantparity | QuantLib |",
    "|---:|---:|---:|",
]
lines += [f"| {n} | {a:+.2e} | {b:+.2e} |" for n, a, b in conv]
lines += [
    "",
    "## Full revaluation speed",
    "",
    f"{len(trades['trades'])} trades (options, American options, Black-76, Euribor swaps on bootstrapped €STR/6M "
    f"curves, bonds) × {n_sc} historical scenarios: {elapsed:.2f} s, curves re-bootstrapped in every scenario.",
    "",
]
out_path = ROOT / "validation/RESULTS.md"
out_path.write_text("\n".join(lines))
print("\n".join(lines))

if "--check" in sys.argv:
    if failures:
        print("\nFAILED:\n" + "\n".join(failures))
        sys.exit(1)
    print("\nAll checks within limits.")
