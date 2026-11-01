//! Expected shortfall backtesting: the Z1 and Z2 tests of Acerbi and Székely ("Backtesting
//! Expected Shortfall", Risk, December 2014).
//!
//! With returns X_t, the VaR at the ES level (VaR_t) and the ES (ES_t) as positive losses, and
//! I_t = 1{X_t < -VaR_t}:
//!
//! - Z2 = Σ X_t I_t / (T (1-α) ES_t) + 1 tests the ES unconditionally; E(Z2) = 0 when the ES is
//!   right, Z2 < 0 when the losses beyond the VaR are larger or more frequent than forecast.
//! - Z1 = Σ X_t I_t / ES_t / N_T + 1 tests the size of the losses given an exception (N_T of them).
//!
//! Acerbi and Székely find the Z2 significance thresholds stable across distributions on a 250-day
//! window at 97.5%: -0.70 at 5% and -1.8 at 0.01%, the same levels as the Basel traffic light.
//! They are used here to give a green / yellow / red zone.

use crate::Status;
use serde::{Deserialize, Serialize};

/// Z2 at the 5% significance level (yellow below) and at 0.01% (red below).
pub const Z2_YELLOW: f64 = -0.70;
pub const Z2_RED: f64 = -1.8;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EsBacktest {
    pub confidence: f64,
    pub observations: usize,
    /// Days with a loss beyond the VaR at the ES level.
    pub exceptions: usize,
    pub expected_exceptions: f64,
    /// Mean ratio of the loss to the forecast ES on exception days (1 when the ES is right).
    pub severity: Option<f64>,
    pub z1: Option<f64>,
    pub z2: f64,
    pub zone: String,
    pub status: Status,
}

/// ES backtest of forecasts `es_var[t]` and `es[t]` (positive losses) for `realized[t]`.
pub fn es_backtest(realized: &[f64], es_var: &[f64], es: &[f64], confidence: f64) -> Option<EsBacktest> {
    let n = realized.len();
    if n == 0 || es_var.len() != n || es.len() != n || !(confidence > 0.0 && confidence < 1.0) {
        return None;
    }
    if es.iter().chain(es_var).chain(realized).any(|x| !x.is_finite()) || es.iter().any(|e| *e <= 0.0) {
        return None;
    }
    let (mut z2_sum, mut ratio_sum, mut x) = (0.0, 0.0, 0usize);
    for t in 0..n {
        if realized[t] < -es_var[t] {
            x += 1;
            z2_sum += realized[t] / es[t];
            ratio_sum += -realized[t] / es[t];
        }
    }
    let z2 = z2_sum / (n as f64 * (1.0 - confidence)) + 1.0;
    let severity = (x > 0).then(|| ratio_sum / x as f64);
    let (zone, status) = if z2 >= Z2_YELLOW {
        ("green", Status::Pass)
    } else if z2 >= Z2_RED {
        ("yellow", Status::Warn)
    } else {
        ("red", Status::Fail)
    };
    Some(EsBacktest {
        confidence,
        observations: n,
        exceptions: x,
        expected_exceptions: n as f64 * (1.0 - confidence),
        severity,
        z1: severity.map(|s| 1.0 - s),
        z2,
        zone: zone.into(),
        status,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn z_statistics_on_a_known_sample() {
        // 200 days, VaR 0.02 and ES 0.03 every day; 5 exceptions of -0.03 = the forecast ES.
        let mut r = vec![0.001; 200];
        for i in [10, 50, 90, 130, 170] {
            r[i] = -0.03;
        }
        let b = es_backtest(&r, &[0.02; 200], &[0.03; 200], 0.975).unwrap();
        assert_eq!(b.exceptions, 5);
        assert!((b.expected_exceptions - 5.0).abs() < 1e-12);
        assert!(b.z2.abs() < 1e-12 && b.z1.unwrap().abs() < 1e-12);
        assert_eq!(b.zone, "green");
        // Losses twice the ES: Z1 = -1, Z2 = -1 (yellow).
        let worse: Vec<f64> = r.iter().map(|x| if *x < 0.0 { -0.06 } else { *x }).collect();
        let b = es_backtest(&worse, &[0.02; 200], &[0.03; 200], 0.975).unwrap();
        assert!((b.z1.unwrap() + 1.0).abs() < 1e-12 && (b.z2 + 1.0).abs() < 1e-12);
        assert_eq!((b.zone.as_str(), b.status), ("yellow", Status::Warn));
        // No exception: Z2 = 1, Z1 undefined.
        let calm = es_backtest(&[0.001; 50], &[0.02; 50], &[0.03; 50], 0.975).unwrap();
        assert_eq!((calm.z2, calm.z1), (1.0, None));
        assert!(es_backtest(&r, &[0.02; 10], &[0.03; 200], 0.975).is_none());
    }
}
