//! Data drift: Population Stability Index.

use crate::Status;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Psi {
    pub bins: usize,
    pub psi: f64,
    pub expected_share: Vec<f64>,
    pub actual_share: Vec<f64>,
    pub status: Status,
}

/// PSI between a reference sample and a current sample, bins from reference quantiles.
/// Usual thresholds: < 0.10 stable, < 0.25 monitor, otherwise significant drift.
pub fn psi(expected: &[f64], actual: &[f64], bins: usize) -> Option<Psi> {
    if expected.is_empty() || actual.is_empty() || bins < 2 {
        return None;
    }
    let mut sorted = expected.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mut edges: Vec<f64> = (1..bins)
        .map(|i| sorted[(i * sorted.len() / bins).min(sorted.len() - 1)])
        .collect();
    edges.dedup();
    let k = edges.len() + 1;
    let share = |xs: &[f64]| {
        let mut c = vec![0f64; k];
        xs.iter().for_each(|x| c[edges.partition_point(|e| e < x)] += 1.0);
        c.iter().map(|v| (v / xs.len() as f64).max(1e-6)).collect::<Vec<_>>()
    };
    let (e, a) = (share(expected), share(actual));
    let value: f64 = e.iter().zip(&a).map(|(e, a)| (a - e) * (a / e).ln()).sum();
    let status = if value < 0.10 {
        Status::Pass
    } else if value < 0.25 {
        Status::Warn
    } else {
        Status::Fail
    };
    Some(Psi {
        bins: k,
        psi: value,
        expected_share: e,
        actual_share: a,
        status,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_samples_are_stable() {
        let x: Vec<f64> = (0..1000).map(|i| i as f64).collect();
        let p = psi(&x, &x, 10).unwrap();
        assert!(p.psi.abs() < 1e-12 && p.status == Status::Pass);
    }

    #[test]
    fn shifted_sample_drifts() {
        let x: Vec<f64> = (0..1000).map(|i| i as f64).collect();
        let y: Vec<f64> = x.iter().map(|v| v + 400.0).collect();
        assert_eq!(psi(&x, &y, 10).unwrap().status, Status::Fail);
    }
}
