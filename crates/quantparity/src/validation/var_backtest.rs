//! VaR model validation: Kupiec, Christoffersen, Basel traffic light.

use crate::stats::{binom_cdf, chi2_sf, xlny};
use crate::Status;
use serde::{Deserialize, Serialize};

/// Historical VaR (positive loss) at confidence level `alpha` (e.g. 0.99):
/// k-th worst return, k = ceil(n·(1-α)).
pub fn historical_var(returns: &[f64], alpha: f64) -> Option<f64> {
    if returns.is_empty() || !(0.0..1.0).contains(&alpha) || returns.iter().any(|x| !x.is_finite()) {
        return None;
    }
    let mut r = returns.to_vec();
    r.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let k = ((1.0 - alpha) * r.len() as f64 - 1e-9).ceil().max(1.0) as usize;
    Some(-r[(k - 1).min(r.len() - 1)])
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Kupiec {
    pub observations: usize,
    pub exceptions: usize,
    pub expected_rate: f64,
    pub observed_rate: f64,
    pub lr_pof: f64,
    pub p_value: f64,
    /// 95% critical value of χ²(1).
    pub critical_value: f64,
    pub passed: bool,
    pub status: Status,
}

/// Kupiec test (proportion of failures), H0: exception rate = 1 - α.
pub fn kupiec_pof(observations: usize, exceptions: usize, alpha: f64) -> Kupiec {
    let (n, x, p) = (observations as f64, exceptions as f64, 1.0 - alpha);
    let ll = |q: f64| xlny(n - x, 1.0 - q) + xlny(x, q);
    let phat = if n > 0.0 { x / n } else { 0.0 };
    let lr = (-2.0 * (ll(p) - ll(phat))).max(0.0);
    let p_value = chi2_sf(lr, 1.0);
    Kupiec {
        observations,
        exceptions,
        expected_rate: p,
        observed_rate: phat,
        lr_pof: lr,
        p_value,
        critical_value: 3.841_458_820_694_124,
        passed: p_value >= 0.05,
        status: Status::from_p_value(p_value),
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Christoffersen {
    pub n00: usize,
    pub n01: usize,
    pub n10: usize,
    pub n11: usize,
    pub pi01: f64,
    pub pi11: f64,
    pub lr_ind: f64,
    pub p_ind: f64,
    pub lr_cc: f64,
    pub p_cc: f64,
    pub status: Status,
}

/// Christoffersen test: independence of exceptions (first-order Markov chain)
/// and conditional coverage (LR_cc = LR_pof + LR_ind, χ²(2)).
pub fn christoffersen(hits: &[bool], alpha: f64) -> Option<Christoffersen> {
    if hits.len() < 2 {
        return None;
    }
    let (mut n00, mut n01, mut n10, mut n11) = (0usize, 0usize, 0usize, 0usize);
    for w in hits.windows(2) {
        match (w[0], w[1]) {
            (false, false) => n00 += 1,
            (false, true) => n01 += 1,
            (true, false) => n10 += 1,
            (true, true) => n11 += 1,
        }
    }
    let f = |x: usize| x as f64;
    let pi01 = if n00 + n01 > 0 { f(n01) / f(n00 + n01) } else { 0.0 };
    let pi11 = if n10 + n11 > 0 { f(n11) / f(n10 + n11) } else { 0.0 };
    let pi = f(n01 + n11) / f(n00 + n01 + n10 + n11);
    let l0 = xlny(f(n00 + n10), 1.0 - pi) + xlny(f(n01 + n11), pi);
    let l1 = xlny(f(n00), 1.0 - pi01) + xlny(f(n01), pi01) + xlny(f(n10), 1.0 - pi11) + xlny(f(n11), pi11);
    let lr_ind = (-2.0 * (l0 - l1)).max(0.0);
    let pof = kupiec_pof(hits.len(), hits.iter().filter(|h| **h).count(), alpha);
    let lr_cc = pof.lr_pof + lr_ind;
    let (p_ind, p_cc) = (chi2_sf(lr_ind, 1.0), chi2_sf(lr_cc, 2.0));
    Some(Christoffersen {
        n00,
        n01,
        n10,
        n11,
        pi01,
        pi11,
        lr_ind,
        p_ind,
        lr_cc,
        p_cc,
        status: Status::from_p_value(p_ind).max(Status::from_p_value(p_cc)),
    })
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TrafficLight {
    pub observations: usize,
    pub exceptions: usize,
    /// P(X ≤ exceptions) under H0.
    pub cumulative_probability: f64,
    pub zone: String,
    /// Multiplier add-on (regulatory 250-day window at 99%).
    pub plus_factor: Option<f64>,
    pub status: Status,
}

/// Basel Committee traffic-light approach: green if P(X ≤ x) < 95%,
/// amber if < 99.99%, red otherwise.
pub fn traffic_light(observations: usize, exceptions: usize, alpha: f64) -> TrafficLight {
    let cum = binom_cdf(exceptions as u64, observations as u64, 1.0 - alpha);
    let (zone, status) = if cum < 0.95 {
        ("green", Status::Pass)
    } else if cum < 0.9999 {
        ("yellow", Status::Warn)
    } else {
        ("red", Status::Fail)
    };
    let regulatory = observations == 250 && (alpha - 0.99).abs() < 1e-12;
    let plus_factor = regulatory.then_some(match exceptions {
        0..=4 => 0.0,
        5 => 0.40,
        6 => 0.50,
        7 => 0.65,
        8 => 0.75,
        9 => 0.85,
        _ => 1.0,
    });
    TrafficLight {
        observations,
        exceptions,
        cumulative_probability: cum,
        zone: zone.into(),
        plus_factor,
        status,
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct VarBacktest {
    pub alpha: f64,
    pub observations: usize,
    pub exceptions: usize,
    pub kupiec: Kupiec,
    pub christoffersen: Option<Christoffersen>,
    pub traffic_light: TrafficLight,
    pub status: Status,
}

/// Full backtest: `var[t]` (positive loss) is the forecast for `realized[t]`.
pub fn var_backtest(realized: &[f64], var: &[f64], alpha: f64) -> Option<VarBacktest> {
    if realized.is_empty() || realized.len() != var.len() {
        return None;
    }
    let hits: Vec<bool> = realized.iter().zip(var).map(|(r, v)| *r < -*v).collect();
    let n = hits.len();
    let x = hits.iter().filter(|h| **h).count();
    let kupiec = kupiec_pof(n, x, alpha);
    let christoffersen = christoffersen(&hits, alpha);
    let traffic_light = traffic_light(n, x, alpha);
    let status = kupiec
        .status
        .max(traffic_light.status)
        .max(christoffersen.as_ref().map_or(Status::Pass, |c| c.status));
    Some(VarBacktest {
        alpha,
        observations: n,
        exceptions: x,
        kupiec,
        christoffersen,
        traffic_light,
        status,
    })
}

/// Rolling backtest of the historical VaR estimated over `window` days.
pub fn rolling_backtest(returns: &[f64], window: usize, alpha: f64) -> Option<VarBacktest> {
    if window == 0 || returns.len() <= window {
        return None;
    }
    let var: Option<Vec<f64>> = (window..returns.len())
        .map(|t| historical_var(&returns[t - window..t], alpha))
        .collect();
    var_backtest(&returns[window..], &var?, alpha)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn var_on_known_sample() {
        let r: Vec<f64> = (1..=100).map(|i| -(i as f64)).collect();
        assert_eq!(historical_var(&r, 0.99), Some(100.0));
        assert_eq!(historical_var(&[f64::NAN], 0.99), None);
    }

    #[test]
    fn kupiec_accepts_expected_rate() {
        let k = kupiec_pof(1000, 10, 0.99);
        assert!(k.passed && k.lr_pof < 1e-9 && (k.p_value - 1.0).abs() < 1e-6);
    }

    #[test]
    fn kupiec_rejects_too_many_exceptions() {
        let k = kupiec_pof(1000, 30, 0.99);
        assert!(!k.passed && k.status == Status::Fail);
    }

    #[test]
    fn kupiec_handles_zero_exceptions() {
        assert!(kupiec_pof(250, 0, 0.99).lr_pof.is_finite());
    }

    #[test]
    fn traffic_light_zones_match_basel() {
        let z = |x| traffic_light(250, x, 0.99).zone;
        assert_eq!(
            (z(4), z(5), z(9), z(10)),
            ("green".into(), "yellow".into(), "yellow".into(), "red".into())
        );
        assert_eq!(traffic_light(250, 7, 0.99).plus_factor, Some(0.65));
    }

    #[test]
    fn christoffersen_detects_clustering() {
        // 10 clustered exceptions vs. 10 scattered exceptions over 1000 days.
        let mut clustered = vec![false; 1000];
        clustered[500..510].iter_mut().for_each(|h| *h = true);
        let mut spread = vec![false; 1000];
        (0..10).for_each(|i| spread[i * 100 + 50] = true);
        let c = christoffersen(&clustered, 0.99).unwrap();
        let s = christoffersen(&spread, 0.99).unwrap();
        assert_eq!(c.status, Status::Fail);
        assert_eq!(s.status, Status::Pass);
    }

    #[test]
    fn full_backtest_aggregates_status() {
        let realized = vec![-0.05; 20];
        let var = vec![0.01; 20];
        let b = var_backtest(&realized, &var, 0.99).unwrap();
        assert_eq!((b.exceptions, b.status), (20, Status::Fail));
    }
}
