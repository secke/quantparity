//! Python bindings for quantparity. Inputs and outputs are plain Python objects (dicts, lists,
//! floats); structured inputs follow the same JSON schema as the Rust crate.

// Python-facing signatures mirror the financial parameters one-to-one.
#![allow(clippy::too_many_arguments)]

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyAny;
use pythonize::{depythonize, pythonize};
use quantparity::{self as qp, benchmark, curves, risk, validation, OptionType};
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
    m.add_function(wrap_pyfunction!(psi, m)?)?;
    m.add_function(wrap_pyfunction!(discrimination, m)?)?;
    m.add_function(wrap_pyfunction!(calibration, m)?)?;
    Ok(())
}
