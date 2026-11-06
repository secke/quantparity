//! Backtests of LGD and CCF estimates against realized values on defaulted facilities.
//!
//! Calibration: one-sided t-test of the mean of realized − estimated, H0: the estimate is not too
//! low (the test of the ECB instructions for reporting the validation results of internal models,
//! 2019, for LGD and CCF), by segment and overall. Ranking: the concordance of estimated and realized
//! values, (1 + Somers' D) / 2, the generalized AUC: 0.5 for no ranking power, 1 for a perfect one.

use crate::stats::t_cdf;
use crate::Status;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SegmentBacktest {
    pub segment: Option<i64>,
    pub observations: usize,
    pub mean_estimated: f64,
    pub mean_realized: f64,
    /// Mean of realized − estimated.
    pub mean_error: f64,
    pub mean_absolute_error: f64,
    pub t_statistic: f64,
    /// P(T ≥ t), Student-t with n − 1 degrees of freedom.
    pub p_value: f64,
    pub status: Status,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ParameterBacktest {
    pub parameter: String,
    pub overall: SegmentBacktest,
    pub segments: Vec<SegmentBacktest>,
    /// Concordance of estimated and realized values, `None` when every realized value is equal.
    pub concordance: Option<f64>,
    pub min_concordance: f64,
    pub status: Status,
}

fn segment_test(segment: Option<i64>, est: &[f64], real: &[f64]) -> SegmentBacktest {
    let n = est.len();
    let nf = n as f64;
    let e: Vec<f64> = real.iter().zip(est).map(|(r, e)| r - e).collect();
    let mean = e.iter().sum::<f64>() / nf;
    let (t, p) = if n < 2 {
        (f64::NAN, f64::NAN)
    } else {
        let sd = (e.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (nf - 1.0)).sqrt();
        if sd == 0.0 {
            // Constant error: its sign is certain.
            match mean.partial_cmp(&0.0) {
                Some(std::cmp::Ordering::Greater) => (f64::INFINITY, 0.0),
                Some(std::cmp::Ordering::Less) => (f64::NEG_INFINITY, 1.0),
                _ => (0.0, 1.0),
            }
        } else {
            let t = mean / (sd / nf.sqrt());
            (t, 1.0 - t_cdf(t, nf - 1.0))
        }
    };
    SegmentBacktest {
        segment,
        observations: n,
        mean_estimated: est.iter().sum::<f64>() / nf,
        mean_realized: real.iter().sum::<f64>() / nf,
        mean_error: mean,
        mean_absolute_error: e.iter().map(|x| x.abs()).sum::<f64>() / nf,
        t_statistic: t,
        p_value: p,
        status: if p.is_nan() {
            Status::Warn
        } else {
            Status::from_p_value(p)
        },
    }
}

/// (1 + Somers' D(estimated | realized)) / 2 over the pairs with different realized values, ties
/// in the estimate counting one half. O(n log n): a Fenwick tree over the ranks of the estimates.
pub fn concordance(estimated: &[f64], realized: &[f64]) -> Option<f64> {
    let n = estimated.len();
    if n != realized.len() || n < 2 {
        return None;
    }
    let mut ranks: Vec<f64> = estimated.to_vec();
    ranks.sort_by(|a, b| a.partial_cmp(b).unwrap());
    ranks.dedup();
    let rank = |x: f64| ranks.partition_point(|r| *r < x);
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| realized[a].partial_cmp(&realized[b]).unwrap());
    let mut tree = vec![0u64; ranks.len() + 1];
    let below = |tree: &[u64], mut i: usize| {
        // Number of inserted estimates with rank < i.
        let mut s = 0;
        while i > 0 {
            s += tree[i];
            i &= i - 1;
        }
        s
    };
    let (mut concordant, mut discordant, mut pairs, mut inserted) = (0u64, 0u64, 0u64, 0u64);
    let mut i = 0;
    while i < n {
        let mut j = i;
        while j + 1 < n && realized[idx[j + 1]] == realized[idx[i]] {
            j += 1;
        }
        for &k in &idx[i..=j] {
            let r = rank(estimated[k]);
            let lower = below(&tree, r);
            concordant += lower;
            discordant += inserted - below(&tree, r + 1);
            pairs += inserted;
        }
        for &k in &idx[i..=j] {
            let mut r = rank(estimated[k]) + 1;
            while r < tree.len() {
                tree[r] += 1;
                r += r & r.wrapping_neg();
            }
            inserted += 1;
        }
        i = j + 1;
    }
    (pairs > 0).then(|| 0.5 + (concordant as f64 - discordant as f64) / (2.0 * pairs as f64))
}

/// Backtest of an estimated parameter (`"lgd"`, `"ccf"`) on defaulted facilities.
pub fn parameter_backtest(
    parameter: &str,
    estimated: &[f64],
    realized: &[f64],
    segment: Option<&[f64]>,
    min_concordance: f64,
) -> Option<ParameterBacktest> {
    let n = estimated.len();
    if n == 0 || realized.len() != n || segment.is_some_and(|s| s.len() != n) {
        return None;
    }
    if estimated.iter().chain(realized).any(|x| !x.is_finite()) {
        return None;
    }
    let overall = segment_test(None, estimated, realized);
    let mut groups: BTreeMap<i64, (Vec<f64>, Vec<f64>)> = BTreeMap::new();
    if let Some(s) = segment {
        for i in 0..n {
            let g = groups.entry(s[i].round() as i64).or_default();
            g.0.push(estimated[i]);
            g.1.push(realized[i]);
        }
    }
    let segments: Vec<SegmentBacktest> = groups
        .into_iter()
        .map(|(g, (e, r))| segment_test(Some(g), &e, &r))
        .collect();
    let concordance = concordance(estimated, realized);
    let ranking = match concordance {
        Some(c) if c < min_concordance => Status::Warn,
        _ => Status::Pass,
    };
    let status = segments
        .iter()
        .map(|s| s.status)
        .fold(overall.status.max(ranking), Status::max);
    Some(ParameterBacktest {
        parameter: parameter.into(),
        overall,
        segments,
        concordance,
        min_concordance,
        status,
    })
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RealizedCcf {
    /// Realized CCF of each facility kept, (EAD − drawn) / (limit − drawn).
    pub ccf: Vec<f64>,
    /// Indices of the facilities kept: undrawn amount above `min_undrawn` at the reference date.
    pub kept: Vec<usize>,
    pub excluded: usize,
    /// Realized CCFs below 0 or above 1 (kept as they are).
    pub outside_unit_interval: usize,
}

/// Realized CCFs from the limit and the drawn amount at the reference date (12 months before
/// default) and the exposure at default.
pub fn realized_ccf(limit: &[f64], drawn: &[f64], ead: &[f64], min_undrawn: f64) -> Option<RealizedCcf> {
    let n = limit.len();
    if drawn.len() != n || ead.len() != n || limit.iter().chain(drawn).chain(ead).any(|x| !x.is_finite()) {
        return None;
    }
    let (mut ccf, mut kept) = (vec![], vec![]);
    for i in 0..n {
        let undrawn = limit[i] - drawn[i];
        if undrawn > min_undrawn {
            ccf.push((ead[i] - drawn[i]) / undrawn);
            kept.push(i);
        }
    }
    let outside_unit_interval = ccf.iter().filter(|c| !(0.0..=1.0).contains(*c)).count();
    Some(RealizedCcf {
        excluded: n - kept.len(),
        ccf,
        kept,
        outside_unit_interval,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn brute(est: &[f64], real: &[f64]) -> f64 {
        let (mut c, mut pairs) = (0.0, 0.0);
        for i in 0..est.len() {
            for j in 0..est.len() {
                if real[i] > real[j] {
                    pairs += 1.0;
                    c += if est[i] > est[j] {
                        1.0
                    } else if est[i] == est[j] {
                        0.5
                    } else {
                        0.0
                    };
                }
            }
        }
        c / pairs
    }

    #[test]
    fn concordance_matches_pairwise_count() {
        let est = [0.2, 0.4, 0.4, 0.1, 0.9, 0.5, 0.5, 0.3];
        let real = [0.0, 0.5, 0.2, 0.2, 1.0, 0.5, 0.1, 0.0];
        assert!((concordance(&est, &real).unwrap() - brute(&est, &real)).abs() < 1e-15);
        assert_eq!(concordance(&[1.0, 2.0, 3.0], &[0.1, 0.2, 0.3]), Some(1.0));
        assert_eq!(concordance(&[1.0, 2.0], &[0.5, 0.5]), None);
    }

    #[test]
    fn t_test_flags_underestimation() {
        let est = vec![0.3; 200];
        let fine: Vec<f64> = (0..200).map(|i| 0.3 + if i % 2 == 0 { 0.2 } else { -0.2 }).collect();
        let low: Vec<f64> = fine.iter().map(|x| x + 0.08).collect();
        let a = parameter_backtest("lgd", &est, &fine, None, 0.0).unwrap();
        assert!(a.overall.mean_error.abs() < 1e-12 && a.overall.status == Status::Pass);
        let b = parameter_backtest("lgd", &est, &low, None, 0.0).unwrap();
        // t = 0.08 / (0.2005 / sqrt(200)) = 5.6.
        assert!(b.overall.t_statistic > 5.0 && b.status == Status::Fail);
    }

    #[test]
    fn ccf_from_limits() {
        let r = realized_ccf(&[100.0, 100.0, 50.0], &[40.0, 100.0, 10.0], &[70.0, 100.0, 60.0], 0.0).unwrap();
        assert_eq!(r.kept, vec![0, 2]);
        assert_eq!(r.ccf, vec![0.5, 1.25]);
        assert_eq!((r.excluded, r.outside_unit_interval), (1, 1));
    }
}
