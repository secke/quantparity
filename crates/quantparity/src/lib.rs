//! # quantparity
//!
//! An independent reference engine for validating production pricers and risk models.
//!
//! - **Pricing**: Black-Scholes-Merton with Greeks, Black-76, Cox-Ross-Rubinstein trees (European and
//!   American), fixed-rate bonds, implied volatility, single-curve and dated multi-curve swaps
//!   (€STR OIS discounting, Euribor projection, TARGET calendar, Modified Following).
//! - **Benchmarking**: compare a candidate pricer to the reference with per-product tolerances.
//! - **Market risk**: factor shocks, full revaluation, historical VaR/ES, bucketed sensitivities,
//!   stress tests and the FRTB P&L attribution test (Spearman / Kolmogorov-Smirnov).
//! - **Model validation**: Kupiec, Christoffersen, Basel traffic light, PSI, AUC/Gini/KS,
//!   binomial and Hosmer-Lemeshow calibration tests.
//!
//! Every result is checked against QuantLib in the repository's `validation/` suite.

use serde::{Deserialize, Serialize};

pub mod benchmark;
pub mod curves;
pub mod dates;
mod instruments;
pub mod risk;
pub mod stats;
pub mod validation;

pub use instruments::*;

/// Outcome of a test or a check.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[serde(rename_all = "lowercase")]
pub enum Status {
    Pass,
    Warn,
    Fail,
}

impl Status {
    /// Status of a hypothesis test: rejection at 1% = fail, at 5% = warn.
    pub fn from_p_value(p: f64) -> Self {
        if p < 0.01 {
            Status::Fail
        } else if p < 0.05 {
            Status::Warn
        } else {
            Status::Pass
        }
    }
}
