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
| Student-t quantile and cdf, 3 to 30 degrees of freedom | 77 × 2 | 1.4e-13 |
| Expected shortfall multipliers, normal and Student-t (QuantLib densities, numerical integration) | 16 | 1.4e-12 |
| EWMA VaR forecasts with t5 innovations | 1,470 | 6.9e-18 |
| Normal quantile | 14 | 3.4e-9 (QuantLib's own error, 4.2e-15 against scipy) |
| GARCH / GJR-GARCH(1,1) conditional volatility, given parameters (against `arch`) | 48,000 | 3.5e-18 |
| GARCH / GJR-GARCH(1,1) fit: log-likelihood at the optimum (against `arch`) | 36 | 1.8e-10 |
| GARCH / GJR-GARCH(1,1) fit: parameters (ω × 1e4, α, γ, β) | 126 | 7.2e-7 |
| Z2 null distribution by simulation, P(Z2 ≤ -0.70) and P(Z2 ≤ -1.8), normal, t3, t5 (against an independent numpy / scipy Monte Carlo) | 6 | 1.2e-3 (Monte Carlo noise: standard error 8e-4) |
| IFRS 9 ECL, stages 1 to 3, bullet / linear / annuity, 1 to 12 periods a year (against an independent numpy implementation) | 600 | 9.8e-15 relative |
| Cumulative PDs from an 8-state rating migration matrix, 1 to 30 years (`numpy.linalg.matrix_power`) | 210 | 4.4e-16 |
| LGD / CCF backtest: t statistic and one-sided p-value (`scipy.stats.ttest_1samp`) | 12 × 2 | 8.8e-14 |
| LGD / CCF backtest: concordance (1 + Somers' D) / 2 (`scipy.stats.somersd`) | 12 | 0 |
| IFRS 9 staging from the SICR criteria, 20,000 facilities (the same rules in numpy) | 9 | 0 |

The normal quantile gap is QuantLib's: `InverseCumulativeNormal` is Acklam's approximation without a refinement step, accurate to about 1e-9; quantparity refines it and agrees with scipy to 4e-15. Filtered historical simulation and the Acerbi-Székely statistics are recomputed independently in the validation script (gaps of 1e-17 and 2e-15). The volatility models have no QuantLib counterpart and are reconciled with the Python [`arch`](https://github.com/bashtage/arch) package instead, on 36 simulated samples of 500 to 2,500 days.

### Finding: `arch` needs rescaled returns, and does not say so in its result

Fitted on daily returns of order 0.01, `arch` stops more than 1e-3 short of the likelihood maximum in 33 of the 36 samples, close to its starting values, and still reports convergence in all 36. By default it only emits a `DataScaleWarning` and fits the unscaled data anyway (`rescale=True` fixes it); with `rescale=False` it is silent. Fitted on 100 r, it reaches the same optimum as quantparity to 2e-10. quantparity rescales the returns internally, so the fit does not depend on the unit of the input.

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
| Volatility | GARCH(1,1) and GJR-GARCH(1,1) by Gaussian quasi-maximum likelihood, with the conventions of `arch` |
| VaR / ES models | One-day forecasts with an EWMA (RiskMetrics), GARCH or GJR volatility, refitted on a rolling or expanding window, and normal, unit-variance Student-t or filtered historical simulation innovations (Barone-Adesi, Hull-White): VaR, ES and the VaR at the ES level |
| Model validation | Kupiec, Christoffersen, Basel traffic light, Acerbi-Székely Z1 and Z2 expected shortfall backtests with simulated p-values, PSI, AUC / Gini / KS, binomial and Hosmer-Lemeshow PD calibration, LGD and CCF backtests (one-sided t-test by segment, concordance), IFRS 9 staging check |
| IFRS 9 credit | ECL of a facility in stage 1, 2 or 3: PD curves (flat hazard, cumulative PDs by year, powers of a rating migration matrix, cohort estimate of the matrix), exposure profiles (bullet, linear, annuity), discounting at the effective interest rate |

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
f = qp.var_forecasts(returns, innovations="t", nu=5)           # or "normal", "fhs"; f["var"], f["es"], f["es_var"]
g = qp.var_forecasts(returns, volatility="gjr", warmup=500, refit=250, innovations="fhs")  # f["fits"]
qp.garch_fit(returns, asymmetric=True)                         # omega, alpha, gamma, beta, loglik
r = returns[f["start"]:]
qp.var_backtest(r, f["var"], alpha=0.99)
b = qp.es_backtest(r, f["es_var"], f["es"], confidence=0.975)  # Z1, Z2, zone (-0.70 / -1.8)
qp.z2_simulation(returns, f, b["z2"], simulations=10000)       # p-value of Z2 under the forecast distribution
qp.psi(reference_scores, current_scores, bins=10)
qp.discrimination(scores, defaults)                            # AUC, Gini, KS
qp.calibration(grades, pds, defaults)                          # binomial per grade, Hosmer-Lemeshow
```

### IFRS 9 credit

```python
qp.ecl(stage=2, balance=10_000, lgd=0.45, eir=0.06, periods=48, pd_curve=[0.02, 0.045, 0.07])
# {'ecl_12m': ..., 'ecl_lifetime': ..., 'ecl': ..., 'pd_horizon': ...}   monthly periods, annuity
qp.migration_pd_curve(states, matrix, from_state=2, years=10)  # cumulative PDs from M^1 ... M^10
qp.parameter_backtest(lgd_estimated, lgd_realized, segment=grades)  # t-test per segment, concordance
c = qp.realized_ccf(limits, drawn_12m_before, ead)              # then parameter_backtest(..., parameter="ccf")
qp.staging_check(records, {"relative_threshold": 2.0, "absolute_threshold": 0.005, "low_credit_risk_pd": 0.003})
```

Conventions: a default in period k happens at its end, on the balance outstanding during the period, and is discounted from that date at the effective interest rate; stage 1 takes the defaults of the first 12 months, stages 2 and 3 those of the remaining life; a stage 3 facility loses LGD × EAD. Between two whole years of a PD curve the hazard rate is constant, and after its last year the conditional PD of that year repeats. The staging rules follow IFRS 9: more than 90 days past due or in default is stage 3; more than 30 days past due (5.5.11), a watch list or forbearance flag, or a significant increase of the PD since origination (relative and absolute thresholds, low credit risk exemption of 5.5.10) is stage 2.

### Finding: the LGD t-test keeps its level on U-shaped LGDs

Realized LGDs pile up near 0 and 1, far from the normal law behind a t-test. The validation script draws 4,000 samples of 20, 50 and 200 realized LGDs from Beta(0.4, 0.6) with a correct estimate: the one-sided test rejects 4.2%, 4.7% and 4.2% of them at 5%. It is slightly conservative, not too severe, even on 20 defaults.

### Finding: the Z2 thresholds hold down to about 5 degrees of freedom

Acerbi and Székely report that the 5% and 0.01% thresholds of their Z2 test, -0.70 and -1.8 on 250 days, hardly depend on the distribution. The validation script checks this by simulation, with 4,000 samples drawn from the forecast distribution itself:

| Distribution | P(Z2 < -0.70), nominal 5% | P(Z2 < -1.8), nominal 0.01% |
|---|---:|---:|
| normal | 4.3% | 0.00% |
| Student-t 10 | 4.5% | 0.00% |
| Student-t 5 | 5.0% | 0.03% |
| Student-t 3 | 6.7% | 0.43% |

With very heavy tails the test rejects a correct model more often than its thresholds say: a red Z2 zone with 3 degrees of freedom happens to a right model about once in 230 windows, not once in 10,000. A Z2 alone is weak evidence for such tails. (With 200,000 samples instead of 4,000, P(Z2 ≤ -0.70) is 5.0% for the normal, 5.7% for t5 and 7.1% for t3.)

The remedy is Acerbi and Székely's own: simulate Z2 from the forecast distribution and read a p-value (`z2_simulation`, [`validation::Z2Null`](crates/quantparity/src/validation/z2_simulation.rs)). The volatility cancels out of Z2, so only the standardized law of each day is drawn: normal, unit-variance Student-t, or the past standardized returns of that day for filtered historical simulation. On 1,000 samples generated by the forecast model itself, the simulated test rejects at its nominal level whatever the tail:

| Distribution | Rejected, Z2 < -0.70 | Rejected, simulated p < 5% |
|---|---:|---:|
| normal | 6.1% | 5.8% |
| Student-t 5 | 5.6% | 4.9% |
| Student-t 3 | 7.2% | 5.0% |

(1,000 samples: standard error 0.7%.)

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
cargo test                                         # 66 Rust tests (Hull, Haug and closed-form values)
pip install pytest QuantLib==1.44 && pytest tests  # Python API
pip install scipy && python validation/validate_quantlib.py --check   # QuantLib parity, fails above the limits
```

## Scope

quantparity is a validation engine, not a front-office library. It does not cover volatility surfaces, stochastic or local volatility, credit and FX products, xVA or Monte Carlo. Extensions follow one rule: nothing is added until it reconciles with an independent reference.

## License

Apache-2.0.
