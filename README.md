# quantparity

**An independent reference engine to validate pricers and risk models, cross-checked against QuantLib on every commit.**

Model validation, independent price verification (IPV) and FRTB all need the same thing: a second implementation, written separately from the production library, that tells you whether a number is right. quantparity is that second implementation. It is a small, dependency-free Rust core with Python bindings, and every result is reconciled with QuantLib 1.44 in CI.

```python
import quantparity as qp

qp.bsm("call", spot=42, strike=40, rate=0.10, vol=0.20, maturity=0.5)
# {'price': 4.7594, 'delta': 0.7791, 'gamma': 0.0500, 'vega': 8.8134, 'theta': -4.5591, 'rho': 13.9820}
```

## Parity with QuantLib

Results of [`validation/validate_quantlib.py`](validation/validate_quantlib.py) (full table in [`validation/RESULTS.md`](validation/RESULTS.md)).

| Area | Cases | Max abs diff vs QuantLib |
|---|---:|---:|
| European options: price, delta, gamma, vega, theta, rho | 432 × 6 | 3.0e-13 |
| Black-76 | 324 | 1.4e-12 |
| Implied volatility | 208 | 3.4e-12 |
| Fixed-rate bonds: price, durations, convexity | 108 × 4 | 2.6e-12 |
| €STR OIS curve, discount factors 1Y–30Y | 30 | 1.8e-13 |
| Euribor 6M swaps with €STR discounting, NPV on 10M EUR | 54 | 2.0e-6 EUR |
| Euribor 6M swaps, par rate | 54 | 1.3e-14 |
| American options, CRR 500 steps | 96 | 1.1e-3 (explained below) |
| Same tree with QuantLib's probability | 96 | 9.0e-11 |

The swap NPV gap is two micro-euros on a ten-million notional, after an independent rebuild of TARGET calendars, Modified Following, backward schedules, Act/360 and 30/360 accruals, spot lag, OIS compounding and a two-curve bootstrap.

### Finding: QuantLib's CRR tree does not use the martingale probability

The only material gap is on binomial trees. quantparity uses the textbook Cox-Ross-Rubinstein probability, which makes the discounted stock price an exact martingale on the tree:

$$p = \frac{e^{(r-q)\Delta t} - d}{u - d}, \qquad u = e^{\sigma\sqrt{\Delta t}},\ d = 1/u$$

QuantLib's `CoxRossRubinstein` lattice uses a first-order expansion of the same quantity:

$$p_{QL} = \frac{1}{2} + \frac{1}{2}\,\frac{(r - q - \sigma^2/2)\,\Delta t}{\sigma\sqrt{\Delta t}}$$

Replacing only that line in an otherwise identical tree reproduces QuantLib to 9e-11 on all 96 cases, which isolates the cause. Both trees converge to the same limit and the gap shrinks as steps are added: about 5e-5 in relative price at 500 steps, 1e-3 in absolute terms on the worst case. Production checks that compare against QuantLib's binomial engine need a tolerance at that level, or this one-line change.

## What is inside

| Module | Content |
|---|---|
| Pricing | Black-Scholes-Merton with Greeks, Black-76, CRR trees (European and American), implied volatility (Newton with bisection fallback), fixed-rate bonds with durations and convexity |
| Curves | Single-curve bootstrap (deposits and swaps, log-linear discount factors). Dated multi-curve: €STR OIS discounting, Euribor projection curves, TARGET calendar, Modified Following, Act/360, Act/365F, 30/360, backward schedules |
| Benchmarking | Compare any candidate pricer to the reference, with per-product absolute and relative tolerances |
| Market risk | Factor shocks (`SPOT`, `VOL`, `CURVE`, `CURVE:<pillar>`, `YIELD`), full revaluation in parallel with curves re-bootstrapped per scenario, historical VaR and ES, bucketed sensitivities, stress tests with limits |
| FRTB | P&L attribution test: hypothetical vs risk-theoretical P&L, Spearman correlation and Kolmogorov-Smirnov distance, green/amber/red zones |
| Model validation | Kupiec, Christoffersen, Basel traffic light, PSI, AUC / Gini / KS, binomial and Hosmer-Lemeshow PD calibration |

## Install

From source (requires Rust 1.80+ and Python 3.9+):

```bash
git clone https://github.com/secke/quantparity && cd quantparity
pip install maturin && maturin develop --release
```

Rust only: add `quantparity = { git = "https://github.com/secke/quantparity" }` to `Cargo.toml`.

## Python examples

### Multi-curve swap

```python
import json, quantparity as qp

market = json.load(open("examples/data/eur_market.json"))   # €STR OIS + Euribor 6M quotes
swap = {"id": "IRS-10Y", "product": "vanilla_swap", "notional": 10_000_000, "fixed_rate": 0.0274,
        "tenor": "10Y", "pay_fixed": True, "forward_curve": "EUR-6M"}
qp.price(swap, market)[0]["measures"]
# {'annuity': 88123966.5, 'dv01': 8806.8, 'fixed_leg_pv': 2414596.7, 'float_leg_pv': 2416359.2, 'par_rate': 0.02742}

curves = qp.Market(market)
curves.discount("EUR-ESTR", "2036-01-15")
curves.report("EUR-6M")["max_repricing_error"]       # < 1e-15
```

### Portfolio risk and the FRTB PLA test

```python
import csv
portfolio = json.load(open("examples/data/portfolio.json"))   # options, American options, swaps, bonds
rows = list(csv.DictReader(open("examples/data/factor_history.csv")))
history = {k: [float(r[k]) for r in rows] for k in rows[0] if k != "day"}   # 300 days of factor moves

pnl = [s["pnl"] for s in qp.revalue(portfolio, history, market)["scenarios"]]
qp.var_es(pnl, var_confidence=0.99, es_confidence=0.975)
# {'var': 483280.3, 'es': 537315.5, 'observations': 300, ...}

qp.sensitivities(portfolio, market)      # 32 factors, DV01 per curve pillar
qp.pla_test(portfolio, history, market)["result"]
# {'spearman': 0.99997, 'ks': 0.0133, 'zone': 'green', ...}

stress = json.load(open("examples/data/stress.json"))["scenarios"]
qp.stress(portfolio, stress, market)[2]
# {'name': 'Sovereign crisis', 'pnl': -1922233.7, 'limit': 1500000.0, 'utilisation': 1.28, 'status': 'fail'}
```

Full revaluation of this portfolio under 300 scenarios, with both curves re-bootstrapped in every scenario, takes about 1.4 s on a 2-core machine.

### Checking a production pricer

```python
candidate = {"EQ-SX5E-C": 312.47, "IR-PAY-10Y": 8813.0}       # prices from the system under review
qp.compare(portfolio, candidate, market, tolerances={"default": {"abs": 1e-6, "rel": 1e-8}})
# per-trade gap, per-product summary, pass / warn / fail, lists missing and unexpected trades
```

### Validation tests

```python
qp.kupiec(observations=250, exceptions=12, alpha=0.99)        # proportion of failures, LR and p-value
qp.traffic_light(250, 7)                                       # {'zone': 'yellow', 'plus_factor': 0.65, ...}
qp.christoffersen(hits, alpha=0.99)                            # independence of exceptions
qp.psi(reference_scores, current_scores, bins=10)
qp.discrimination(scores, defaults)                            # AUC, Gini, KS
qp.calibration(grades, pds, defaults)                          # binomial per grade, Hosmer-Lemeshow
```

## Rust example

```rust
use quantparity::{bsm, crr, OptionType};

let g = bsm(OptionType::Call, 42.0, 40.0, 0.10, 0.0, 0.20, 0.5)?; // s, k, r, q, vol, t
let american_put = crr(OptionType::Put, 50.0, 50.0, 0.10, 0.0, 0.40, 5.0 / 12.0, 500, true)?;
```

Run `cargo run --example quickstart` for a complete program (pricing, multi-curve swap, VaR).

Trades, markets and scenarios deserialize from the same JSON as in Python (`serde`).

## Input schema

Trades carry `id`, optional `quantity` (default 1, negative for short) and `underlying` (the name used by risk factors), plus one product:

| `product` | Fields |
|---|---|
| `european_option`, `american_option` | `option_type`, `spot`, `strike`, `rate`, `dividend` (0), `vol`, `maturity` (years); American: `steps` |
| `black76` | `option_type`, `forward`, `strike`, `rate`, `vol`, `maturity` |
| `fixed_rate_bond` | `coupon_rate`, `maturity`, `frequency`, `yield_rate`, `face` (100) |
| `vanilla_swap` | `notional`, `fixed_rate`, `tenor` (`10Y`), `pay_fixed` (true), `forward_curve`, `discount_curve` (optional) |
| `interest_rate_swap`, `zero_coupon_bond` | single-curve versions, maturity in years, `curve` |

Markets: `{"as_of": "YYYY-MM-DD", "curves": {"<name>": {...}}}`. Curves with `"kind": "ois"` or `"kind": "ibor"` are dated; see [`examples/data/eur_market.json`](examples/data/eur_market.json).

## Reproduce

```bash
cargo test                                         # 46 Rust tests (Hull, Haug and closed-form values)
pip install pytest QuantLib==1.44 && pytest tests  # Python API
python validation/validate_quantlib.py --check     # QuantLib parity, fails above the limits
```

## Scope

quantparity is a validation engine, not a front-office library. It does not cover volatility surfaces, stochastic or local volatility, credit and FX products, xVA or Monte Carlo. Extensions follow one rule: nothing is added until it reconciles with an independent reference.

## License

Apache-2.0.
