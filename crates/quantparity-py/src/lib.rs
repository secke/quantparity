//! Python bindings for quantparity. Inputs and outputs are plain Python objects (dicts, lists,
//! floats); structured inputs follow the same JSON schema as the Rust crate.

// Python-facing signatures mirror the financial parameters one-to-one.
#![allow(clippy::too_many_arguments)]

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyAny;
use pythonize::{depythonize, pythonize};
use quantparity::{self as qp, benchmark, curves, ifrs9, risk, stats, validation, var_models, OptionType};
use serde::Serialize;
use serde_json::Value;

fn err<E: std::fmt::Display>(e: E) -> PyErr {
    PyValueError::new_err(e.to_string())
}

fn to_py<T: Serialize>(py: Python<'_>, v: &T) -> PyResult<PyObject> {
    Ok(pythonize(py, v).map_err(err)?.unbind())
}

fn from_py(obj: &Bound<'_, PyAny>) -> PyResult<Value> {
    depythonize(obj).map_err(err)
}

fn option_type(s: &str) -> PyResult<OptionType> {
    match s.to_ascii_lowercase().as_str() {
        "call" | "c" => Ok(OptionType::Call),
        "put" | "p" => Ok(OptionType::Put),
        _ => Err(err(format!("option_type must be 'call' or 'put' (got '{s}')"))),
    }
}

fn need<T>(v: Option<T>, what: &str) -> PyResult<T> {
    v.ok_or_else(|| err(format!("{what}: empty, mismatched or non-finite inputs")))
}

fn trades(obj: &Bound<'_, PyAny>) -> PyResult<Vec<qp::Trade>> {
    let v = from_py(obj)?;
    let v = if v.is_object() && v.get("trades").is_none() {
        Value::Array(vec![v])
    } else {
        v
    };
    benchmark::trades_from_value(&v).map_err(err)
}

fn market_value(obj: Option<&Bound<'_, PyAny>>) -> PyResult<Value> {
    match obj {
        Some(o) if !o.is_none() => from_py(o),
        _ => Ok(serde_json::json!({"curves": {}})),
    }
}

fn scenarios(obj: &Bound<'_, PyAny>) -> PyResult<Vec<risk::Scenario>> {
    let v = from_py(obj)?;
    if v.is_array() {
        serde_json::from_value(v).map_err(err)
    } else {
        risk::scenarios_from_table(&v, None).map_err(err)
    }
}

// ---------------------------------------------------------------- pricing

/// Black-Scholes-Merton price and Greeks (vega per 1.00 of vol, theta per year).
#[pyfunction]
#[pyo3(signature = (option_type, spot, strike, rate, vol, maturity, dividend=0.0))]
fn bsm(
    py: Python<'_>,
    option_type: &str,
    spot: f64,
    strike: f64,
    rate: f64,
    vol: f64,
    maturity: f64,
    dividend: f64,
) -> PyResult<PyObject> {
    let g = qp::bsm(
        self::option_type(option_type)?,
        spot,
        strike,
        rate,
        dividend,
        vol,
        maturity,
    )
    .map_err(err)?;
    to_py(py, &g)
}

/// Black (1976) price and Greeks of an option on a forward.
#[pyfunction]
#[pyo3(signature = (option_type, forward, strike, rate, vol, maturity))]
fn black76(
    py: Python<'_>,
    option_type: &str,
    forward: f64,
    strike: f64,
    rate: f64,
    vol: f64,
    maturity: f64,
) -> PyResult<PyObject> {
    let g = qp::black76(self::option_type(option_type)?, forward, strike, rate, vol, maturity).map_err(err)?;
    to_py(py, &g)
}

/// Cox-Ross-Rubinstein binomial tree (exact martingale probability); American if `american`.
#[pyfunction]
#[pyo3(signature = (option_type, spot, strike, rate, vol, maturity, steps=500, dividend=0.0, american=false))]
fn crr(
    option_type: &str,
    spot: f64,
    strike: f64,
    rate: f64,
    vol: f64,
    maturity: f64,
    steps: usize,
    dividend: f64,
    american: bool,
) -> PyResult<f64> {
    qp::crr(
        self::option_type(option_type)?,
        spot,
        strike,
        rate,
        dividend,
        vol,
        maturity,
        steps,
        american,
    )
    .map_err(err)
}

/// Black-Scholes-Merton implied volatility.
#[pyfunction]
#[pyo3(signature = (option_type, price, spot, strike, rate, maturity, dividend=0.0))]
fn implied_vol(
    option_type: &str,
    price: f64,
    spot: f64,
    strike: f64,
    rate: f64,
    maturity: f64,
    dividend: f64,
) -> PyResult<f64> {
    qp::implied_vol(
        self::option_type(option_type)?,
        price,
        spot,
        strike,
        rate,
        dividend,
        maturity,
    )
    .map_err(err)
}

/// Fixed-rate bond: price, Macaulay and modified duration, convexity.
#[pyfunction]
#[pyo3(signature = (coupon, maturity, yield_rate, frequency=1, face=100.0))]
fn bond(py: Python<'_>, coupon: f64, maturity: f64, yield_rate: f64, frequency: u32, face: f64) -> PyResult<PyObject> {
    let m = qp::fixed_rate_bond(face, coupon, maturity, frequency, yield_rate).map_err(err)?;
    to_py(py, &m)
}

/// Values one trade (dict) or a list of trades against an optional market (dict of curves).
/// Returns a list of valuations `{id, product, price, measures}`.
#[pyfunction]
#[pyo3(signature = (trades, market=None))]
fn price(py: Python<'_>, trades: &Bound<'_, PyAny>, market: Option<&Bound<'_, PyAny>>) -> PyResult<PyObject> {
    let ts = self::trades(trades)?;
    let m = curves::Market::from_value(&market_value(market)?).map_err(err)?;
    let out = py
        .allow_threads(|| ts.iter().map(|t| qp::value_in(t, &m)).collect::<Result<Vec<_>, _>>())
        .map_err(err)?;
    to_py(py, &out)
}

// ---------------------------------------------------------------- curves

/// Market data: named single curves and dated multi-curves, bootstrapped at construction.
#[pyclass(module = "quantparity", frozen)]
struct Market {
    inner: curves::Market,
}

#[pymethods]
impl Market {
    #[new]
    fn new(spec: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(Market {
            inner: curves::Market::from_value(&from_py(spec)?).map_err(err)?,
        })
    }

    /// Names of all curves.
    fn curves(&self) -> Vec<String> {
        self.inner
            .curves
            .keys()
            .chain(self.inner.dated.keys())
            .cloned()
            .collect()
    }

    /// Discount factor at `t`: a time in years (single curve) or a `YYYY-MM-DD` date (dated curve).
    fn discount(&self, curve: &str, t: &Bound<'_, PyAny>) -> PyResult<f64> {
        if let Ok(d) = t.extract::<String>() {
            let d = qp::dates::Date::parse(&d).map_err(err)?;
            Ok(self.inner.dated_curve(curve).map_err(err)?.discount(d))
        } else {
            Ok(self.inner.curve(curve).map_err(err)?.discount(t.extract()?))
        }
    }

    /// Continuously compounded zero rate at `t` (years or `YYYY-MM-DD`).
    fn zero_rate(&self, curve: &str, t: &Bound<'_, PyAny>) -> PyResult<f64> {
        if let Ok(d) = t.extract::<String>() {
            let d = qp::dates::Date::parse(&d).map_err(err)?;
            Ok(self.inner.dated_curve(curve).map_err(err)?.zero_rate(d))
        } else {
            Ok(self.inner.curve(curve).map_err(err)?.zero_rate(t.extract()?))
        }
    }

    /// Build report: repricing error of every input instrument and the term structure.
    #[pyo3(signature = (curve, tolerance=1e-10, step_years=1.0))]
    fn report(&self, py: Python<'_>, curve: &str, tolerance: f64, step_years: f64) -> PyResult<PyObject> {
        let r = curves::simple::curve_report(&self.inner, curve, tolerance, step_years).map_err(err)?;
        to_py(py, &r)
    }
}

// ---------------------------------------------------------------- benchmark

/// Compares candidate prices (`{id: price}` or `[{id, price}]`) to the reference engine.
#[pyfunction]
#[pyo3(signature = (trades, candidate, market=None, tolerances=None))]
fn compare(
    py: Python<'_>,
    trades: &Bound<'_, PyAny>,
    candidate: &Bound<'_, PyAny>,
    market: Option<&Bound<'_, PyAny>>,
    tolerances: Option<&Bound<'_, PyAny>>,
) -> PyResult<PyObject> {
    let ts = self::trades(trades)?;
    let m = curves::Market::from_value(&market_value(market)?).map_err(err)?;
    let cand = benchmark::prices_from_value(&from_py(candidate)?).map_err(err)?;
    let tol = match tolerances {
        Some(t) if !t.is_none() => from_py(t)?,
        _ => Value::Null,
    };
    let tol = benchmark::Tolerances::from_value(&tol).map_err(err)?;
    let reference = benchmark::reference_prices(&ts, &m).map_err(err)?;
    to_py(py, &benchmark::compare(&ts, &reference, &cand, &tol).map_err(err)?)
}

// ---------------------------------------------------------------- risk

/// Value of each position (unit price × quantity).
#[pyfunction]
#[pyo3(signature = (trades, market=None))]
fn position_values(trades: &Bound<'_, PyAny>, market: Option<&Bound<'_, PyAny>>) -> PyResult<Vec<f64>> {
    risk::values(&self::trades(trades)?, &market_value(market)?).map_err(err)
}

/// Full revaluation under scenarios. `scenarios` is `[{name, shifts}]` or a table `{factor: [shocks]}`.
#[pyfunction]
#[pyo3(signature = (trades, scenarios, market=None))]
fn revalue(
    py: Python<'_>,
    trades: &Bound<'_, PyAny>,
    scenarios: &Bound<'_, PyAny>,
    market: Option<&Bound<'_, PyAny>>,
) -> PyResult<PyObject> {
    let (ts, sc, mk) = (
        self::trades(trades)?,
        self::scenarios(scenarios)?,
        market_value(market)?,
    );
    let (base, pnl) = py.allow_threads(|| risk::revalue(&ts, &mk, &sc)).map_err(err)?;
    to_py(py, &serde_json::json!({"base": base, "scenarios": pnl}))
}

/// Historical VaR and expected shortfall of a P&L vector, as positive amounts.
#[pyfunction]
#[pyo3(signature = (pnl, var_confidence=0.99, es_confidence=0.975))]
fn var_es(py: Python<'_>, pnl: Vec<f64>, var_confidence: f64, es_confidence: f64) -> PyResult<PyObject> {
    to_py(py, &need(risk::var_es(&pnl, var_confidence, es_confidence), "var_es")?)
}

/// Central-difference sensitivities to every risk factor the portfolio is exposed to.
#[pyfunction]
#[pyo3(signature = (trades, market=None))]
fn sensitivities(py: Python<'_>, trades: &Bound<'_, PyAny>, market: Option<&Bound<'_, PyAny>>) -> PyResult<PyObject> {
    let (ts, mk) = (self::trades(trades)?, market_value(market)?);
    let s = py.allow_threads(|| risk::sensitivities(&ts, &mk)).map_err(err)?;
    to_py(py, &s)
}

/// Stress tests: loss per scenario against its optional `limit`.
#[pyfunction]
#[pyo3(signature = (trades, scenarios, market=None))]
fn stress(
    py: Python<'_>,
    trades: &Bound<'_, PyAny>,
    scenarios: &Bound<'_, PyAny>,
    market: Option<&Bound<'_, PyAny>>,
) -> PyResult<PyObject> {
    let (ts, sc, mk) = (
        self::trades(trades)?,
        self::scenarios(scenarios)?,
        market_value(market)?,
    );
    to_py(py, &py.allow_threads(|| risk::stress(&ts, &mk, &sc)).map_err(err)?)
}

/// FRTB P&L attribution test (Spearman correlation and Kolmogorov-Smirnov distance).
#[pyfunction]
#[pyo3(signature = (hpl, rtpl, min_observations=250))]
fn pnl_attribution(py: Python<'_>, hpl: Vec<f64>, rtpl: Vec<f64>, min_observations: usize) -> PyResult<PyObject> {
    to_py(
        py,
        &need(risk::pnl_attribution(&hpl, &rtpl, min_observations), "pnl_attribution")?,
    )
}

/// Full PLA workflow: hypothetical P&L by full revaluation vs risk-theoretical P&L from sensitivities.
#[pyfunction]
#[pyo3(signature = (trades, scenarios, market=None, min_observations=250))]
fn pla_test(
    py: Python<'_>,
    trades: &Bound<'_, PyAny>,
    scenarios: &Bound<'_, PyAny>,
    market: Option<&Bound<'_, PyAny>>,
    min_observations: usize,
) -> PyResult<PyObject> {
    let (ts, sc, mk) = (
        self::trades(trades)?,
        self::scenarios(scenarios)?,
        market_value(market)?,
    );
    let (hpl, rtpl) = py
        .allow_threads(|| -> Result<_, String> {
            let (_, pnl) = risk::revalue(&ts, &mk, &sc)?;
            let sens = risk::sensitivities(&ts, &mk)?;
            Ok((
                pnl.iter().map(|p| p.pnl).collect::<Vec<_>>(),
                risk::risk_theoretical_pnl(&sens, &sc),
            ))
        })
        .map_err(err)?;
    let pla = need(risk::pnl_attribution(&hpl, &rtpl, min_observations), "pla_test")?;
    to_py(py, &serde_json::json!({"hpl": hpl, "rtpl": rtpl, "result": pla}))
}

// ---------------------------------------------------------------- validation

/// Kupiec proportion-of-failures test.
#[pyfunction]
#[pyo3(signature = (observations, exceptions, alpha=0.99))]
fn kupiec(py: Python<'_>, observations: usize, exceptions: usize, alpha: f64) -> PyResult<PyObject> {
    to_py(py, &validation::kupiec_pof(observations, exceptions, alpha))
}

/// Christoffersen independence and conditional coverage tests on a sequence of exceptions.
#[pyfunction]
#[pyo3(signature = (hits, alpha=0.99))]
fn christoffersen(py: Python<'_>, hits: Vec<bool>, alpha: f64) -> PyResult<PyObject> {
    to_py(py, &need(validation::christoffersen(&hits, alpha), "christoffersen")?)
}

/// Basel traffic-light zone.
#[pyfunction]
#[pyo3(signature = (observations, exceptions, alpha=0.99))]
fn traffic_light(py: Python<'_>, observations: usize, exceptions: usize, alpha: f64) -> PyResult<PyObject> {
    to_py(py, &validation::traffic_light(observations, exceptions, alpha))
}

/// Full VaR backtest: `var[t]` (positive loss) is the forecast for `realized[t]`.
#[pyfunction]
#[pyo3(signature = (realized, var, alpha=0.99))]
fn var_backtest(py: Python<'_>, realized: Vec<f64>, var: Vec<f64>, alpha: f64) -> PyResult<PyObject> {
    to_py(
        py,
        &need(validation::var_backtest(&realized, &var, alpha), "var_backtest")?,
    )
}

/// Expected shortfall backtest (Acerbi-Székely Z1 and Z2): `es_var[t]` is the VaR at the ES
/// confidence level and `es[t]` the ES forecast for `realized[t]`, as positive losses.
#[pyfunction]
#[pyo3(signature = (realized, es_var, es, confidence=0.975))]
fn es_backtest(
    py: Python<'_>,
    realized: Vec<f64>,
    es_var: Vec<f64>,
    es: Vec<f64>,
    confidence: f64,
) -> PyResult<PyObject> {
    to_py(
        py,
        &need(
            validation::es_backtest(&realized, &es_var, &es, confidence),
            "es_backtest",
        )?,
    )
}

// ---------------------------------------------------------------- VaR / ES models

/// One-day VaR and ES forecasts. `volatility`: `"ewma"` (decay `lambda_`), `"garch"` or `"gjr"`
/// (fitted on the `warmup` first returns, refitted every `refit` days on the latest `fit_window`
/// returns, all of them if `None`). `innovations`: `"normal"`, `"t"` (Student-t with `nu` degrees
/// of freedom, unit variance) or `"fhs"` (filtered historical simulation on the latest `window`
/// standardized returns, all of them if `None`, once `min_residuals` are available).
#[pyfunction]
#[pyo3(signature = (returns, lambda_=0.94, warmup=30, var_confidence=0.99, es_confidence=0.975, innovations="normal", nu=None, window=None, min_residuals=250, volatility="ewma", refit=250, fit_window=None))]
#[allow(clippy::too_many_arguments)]
fn var_forecasts(
    py: Python<'_>,
    returns: Vec<f64>,
    lambda_: f64,
    warmup: usize,
    var_confidence: f64,
    es_confidence: f64,
    innovations: &str,
    nu: Option<f64>,
    window: Option<usize>,
    min_residuals: usize,
    volatility: &str,
    refit: usize,
    fit_window: Option<usize>,
) -> PyResult<PyObject> {
    let vol = match volatility {
        "ewma" => var_models::Volatility::Ewma { lambda: lambda_ },
        "garch" | "gjr" => var_models::Volatility::Garch {
            asymmetric: volatility == "gjr",
            refit,
            window: fit_window,
        },
        other => return Err(err(format!("volatility: 'ewma', 'garch' or 'gjr' (got '{other}')"))),
    };
    let innov = match innovations {
        "normal" => var_models::Innovations::Normal,
        "t" => var_models::Innovations::StudentT {
            nu: need(nu, "nu (degrees of freedom of the Student-t)")?,
        },
        "fhs" => var_models::Innovations::Filtered { window, min_residuals },
        other => return Err(err(format!("innovations: 'normal', 't' or 'fhs' (got '{other}')"))),
    };
    let f = var_models::var_forecasts_with(&returns, vol, warmup, var_confidence, es_confidence, innov).map_err(err)?;
    to_py(py, &f)
}

/// P-value of an observed Z2 by simulation from the forecast distribution: `forecasts` is the
/// output of `var_forecasts` on `returns`, the test covers returns `from_..to` (default: every
/// forecast day).
#[pyfunction]
#[pyo3(signature = (returns, forecasts, z2, from_=None, to=None, simulations=10000, seed=1))]
#[allow(clippy::too_many_arguments)]
fn z2_simulation(
    py: Python<'_>,
    returns: Vec<f64>,
    forecasts: &Bound<'_, PyAny>,
    z2: f64,
    from_: Option<usize>,
    to: Option<usize>,
    simulations: usize,
    seed: u64,
) -> PyResult<PyObject> {
    let f: var_models::VarForecasts = depythonize(forecasts).map_err(err)?;
    let null = validation::Z2Null::new(&returns, &f).map_err(err)?;
    let s = null
        .test(
            z2,
            from_.unwrap_or(f.start),
            to.unwrap_or(returns.len()),
            simulations,
            seed,
        )
        .map_err(err)?;
    to_py(py, &s)
}

/// Gaussian quasi-maximum likelihood fit of a GARCH(1,1), or GJR-GARCH(1,1) if `asymmetric`,
/// with zero mean and the conventions of the `arch` package.
#[pyfunction]
#[pyo3(signature = (returns, asymmetric=false))]
fn garch_fit(py: Python<'_>, returns: Vec<f64>, asymmetric: bool) -> PyResult<PyObject> {
    to_py(py, &qp::garch::garch_fit(&returns, asymmetric).map_err(err)?)
}

/// Gaussian log-likelihood and conditional variances (followed by the next-day forecast) of a
/// GARCH / GJR with the given parameters and backcast.
#[pyfunction]
fn garch_loglik(returns: Vec<f64>, omega: f64, alpha: f64, gamma: f64, beta: f64, backcast: f64) -> (f64, Vec<f64>) {
    let p = qp::garch::GarchParams {
        omega,
        alpha,
        gamma,
        beta,
    };
    (
        qp::garch::loglik(&returns, &p, backcast),
        qp::garch::variances(&returns, &p, backcast),
    )
}

/// Quantile of the standard normal distribution.
#[pyfunction]
fn norm_inv(p: f64) -> f64 {
    stats::norm_inv(p)
}

/// Quantile of Student's t distribution with `nu` degrees of freedom.
#[pyfunction]
fn t_inv(p: f64, nu: f64) -> f64 {
    stats::t_inv(p, nu)
}

/// Cumulative distribution function of Student's t distribution.
#[pyfunction]
fn t_cdf(x: f64, nu: f64) -> f64 {
    stats::t_cdf(x, nu)
}

/// Population Stability Index with bins from reference quantiles.
#[pyfunction]
#[pyo3(signature = (expected, actual, bins=10))]
fn psi(py: Python<'_>, expected: Vec<f64>, actual: Vec<f64>, bins: usize) -> PyResult<PyObject> {
    to_py(py, &need(validation::psi(&expected, &actual, bins), "psi")?)
}

/// Discriminatory power of a PD score: AUC, Gini, KS (a higher score means a higher risk).
#[pyfunction]
#[pyo3(signature = (score, default, min_gini_pass=0.4, min_gini_warn=0.3))]
fn discrimination(
    py: Python<'_>,
    score: Vec<f64>,
    default: Vec<bool>,
    min_gini_pass: f64,
    min_gini_warn: f64,
) -> PyResult<PyObject> {
    to_py(
        py,
        &need(
            validation::discrimination(&score, &default, min_gini_pass, min_gini_warn),
            "discrimination",
        )?,
    )
}

/// PD calibration by grade: binomial test per grade and Hosmer-Lemeshow overall.
#[pyfunction]
fn calibration(py: Python<'_>, grade: Vec<f64>, pd: Vec<f64>, default: Vec<bool>) -> PyResult<PyObject> {
    to_py(
        py,
        &need(validation::calibration(&grade, &pd, &default), "calibration")?,
    )
}

/// IFRS 9 ECL of a facility: `pd_curve` lists the cumulative PDs at the end of each year.
#[pyfunction]
#[pyo3(signature = (stage, balance, lgd, eir, periods, pd_curve, periods_per_year=12, repayment="annuity"))]
fn ecl(
    py: Python<'_>,
    stage: u8,
    balance: f64,
    lgd: f64,
    eir: f64,
    periods: usize,
    pd_curve: Vec<f64>,
    periods_per_year: u32,
    repayment: &str,
) -> PyResult<PyObject> {
    let pd = ifrs9::PdCurve::new(pd_curve).map_err(err)?;
    let f = ifrs9::Facility {
        stage,
        balance,
        lgd,
        eir,
        periods,
        repayment: ifrs9::Repayment::parse(repayment).map_err(err)?,
    };
    to_py(py, &f.ecl(&pd, periods_per_year).map_err(err)?)
}

/// IFRS 9 ECL from period-by-period cumulative PDs, LGDs and exposures.
#[pyfunction]
#[pyo3(signature = (stage, cumulative_pd, lgd, ead, eir, periods_per_year=12))]
fn ecl_profile(
    py: Python<'_>,
    stage: u8,
    cumulative_pd: Vec<f64>,
    lgd: Vec<f64>,
    ead: Vec<f64>,
    eir: f64,
    periods_per_year: u32,
) -> PyResult<PyObject> {
    let input = ifrs9::EclInput {
        stage,
        cumulative_pd,
        lgd,
        ead,
        eir,
    };
    to_py(py, &ifrs9::ecl(&input, periods_per_year).map_err(err)?)
}

/// Cumulative PDs over `years` years from state `from_state` of an annual migration matrix whose
/// last state is default.
#[pyfunction]
fn migration_pd_curve(
    states: Vec<String>,
    matrix: Vec<Vec<f64>>,
    from_state: usize,
    years: usize,
) -> PyResult<Vec<f64>> {
    let m = ifrs9::MigrationMatrix::new(states, matrix).map_err(err)?;
    Ok(m.pd_curve(from_state, years).map_err(err)?.cumulative)
}

/// Exposure profile: balance outstanding in each remaining period.
#[pyfunction]
#[pyo3(signature = (balance, periods, eir, periods_per_year=12, repayment="annuity"))]
fn exposure_profile(
    balance: f64,
    periods: usize,
    eir: f64,
    periods_per_year: u32,
    repayment: &str,
) -> PyResult<Vec<f64>> {
    Ok(ifrs9::Repayment::parse(repayment)
        .map_err(err)?
        .profile(balance, periods, eir, periods_per_year))
}

/// LGD or CCF backtest: one-sided t-test by segment and overall, concordance of the ranking.
#[pyfunction]
#[pyo3(signature = (estimated, realized, segment=None, parameter="lgd", min_concordance=0.6))]
fn parameter_backtest(
    py: Python<'_>,
    estimated: Vec<f64>,
    realized: Vec<f64>,
    segment: Option<Vec<f64>>,
    parameter: &str,
    min_concordance: f64,
) -> PyResult<PyObject> {
    let b = validation::parameter_backtest(parameter, &estimated, &realized, segment.as_deref(), min_concordance);
    to_py(py, &need(b, "parameter_backtest")?)
}

/// (1 + Somers' D(estimated | realized)) / 2.
#[pyfunction]
fn concordance(estimated: Vec<f64>, realized: Vec<f64>) -> Option<f64> {
    validation::concordance(&estimated, &realized)
}

/// Realized CCFs, (EAD − drawn) / (limit − drawn), for an undrawn amount above `min_undrawn`.
#[pyfunction]
#[pyo3(signature = (limit, drawn, ead, min_undrawn=0.0))]
fn realized_ccf(
    py: Python<'_>,
    limit: Vec<f64>,
    drawn: Vec<f64>,
    ead: Vec<f64>,
    min_undrawn: f64,
) -> PyResult<PyObject> {
    to_py(
        py,
        &need(
            validation::realized_ccf(&limit, &drawn, &ead, min_undrawn),
            "realized_ccf",
        )?,
    )
}

/// IFRS 9 staging check: `records` is a list of dicts (reported_stage, defaulted, dpd,
/// pd_origination, pd_current, flag, defaulted_next_12m), `rules` a dict of StagingRules fields.
#[pyfunction]
#[pyo3(signature = (records, rules=None))]
fn staging_check(py: Python<'_>, records: &Bound<'_, PyAny>, rules: Option<&Bound<'_, PyAny>>) -> PyResult<PyObject> {
    let mut v = from_py(records)?;
    for r in v.as_array_mut().into_iter().flatten() {
        if let Some(o) = r.as_object_mut() {
            for (k, d) in [
                ("defaulted", Value::Bool(false)),
                ("dpd", 0.0.into()),
                ("flag", Value::Bool(false)),
            ] {
                o.entry(k).or_insert(d);
            }
            o.entry("defaulted_next_12m").or_insert(Value::Null);
        }
    }
    let records: Vec<validation::StagingRecord> = serde_json::from_value(v).map_err(err)?;
    let mut r = serde_json::to_value(validation::StagingRules::default()).unwrap();
    if let Some(o) = rules.filter(|o| !o.is_none()) {
        for (k, x) in from_py(o)?.as_object().into_iter().flatten() {
            r[k] = x.clone();
        }
    }
    let rules: validation::StagingRules = serde_json::from_value(r).map_err(err)?;
    to_py(py, &validation::staging_check(&records, &rules).map_err(err)?)
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_class::<Market>()?;
    m.add_function(wrap_pyfunction!(bsm, m)?)?;
    m.add_function(wrap_pyfunction!(black76, m)?)?;
    m.add_function(wrap_pyfunction!(crr, m)?)?;
    m.add_function(wrap_pyfunction!(implied_vol, m)?)?;
    m.add_function(wrap_pyfunction!(bond, m)?)?;
    m.add_function(wrap_pyfunction!(price, m)?)?;
    m.add_function(wrap_pyfunction!(compare, m)?)?;
    m.add_function(wrap_pyfunction!(position_values, m)?)?;
    m.add_function(wrap_pyfunction!(revalue, m)?)?;
    m.add_function(wrap_pyfunction!(var_es, m)?)?;
    m.add_function(wrap_pyfunction!(sensitivities, m)?)?;
    m.add_function(wrap_pyfunction!(stress, m)?)?;
    m.add_function(wrap_pyfunction!(pnl_attribution, m)?)?;
    m.add_function(wrap_pyfunction!(pla_test, m)?)?;
    m.add_function(wrap_pyfunction!(kupiec, m)?)?;
    m.add_function(wrap_pyfunction!(christoffersen, m)?)?;
    m.add_function(wrap_pyfunction!(traffic_light, m)?)?;
    m.add_function(wrap_pyfunction!(var_backtest, m)?)?;
    m.add_function(wrap_pyfunction!(es_backtest, m)?)?;
    m.add_function(wrap_pyfunction!(var_forecasts, m)?)?;
    m.add_function(wrap_pyfunction!(garch_fit, m)?)?;
    m.add_function(wrap_pyfunction!(z2_simulation, m)?)?;
    m.add_function(wrap_pyfunction!(garch_loglik, m)?)?;
    m.add_function(wrap_pyfunction!(norm_inv, m)?)?;
    m.add_function(wrap_pyfunction!(t_inv, m)?)?;
    m.add_function(wrap_pyfunction!(t_cdf, m)?)?;
    m.add_function(wrap_pyfunction!(psi, m)?)?;
    m.add_function(wrap_pyfunction!(discrimination, m)?)?;
    m.add_function(wrap_pyfunction!(calibration, m)?)?;
    m.add_function(wrap_pyfunction!(ecl, m)?)?;
    m.add_function(wrap_pyfunction!(ecl_profile, m)?)?;
    m.add_function(wrap_pyfunction!(migration_pd_curve, m)?)?;
    m.add_function(wrap_pyfunction!(exposure_profile, m)?)?;
    m.add_function(wrap_pyfunction!(parameter_backtest, m)?)?;
    m.add_function(wrap_pyfunction!(concordance, m)?)?;
    m.add_function(wrap_pyfunction!(realized_ccf, m)?)?;
    m.add_function(wrap_pyfunction!(staging_check, m)?)?;
    Ok(())
}
