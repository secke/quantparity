//! PD model validation: discriminatory power and calibration.

use crate::stats::{binom_sf, chi2_sf};
use crate::Status;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Discrimination {
    pub observations: usize,
    pub defaults: usize,
    pub auc: f64,
    pub gini: f64,
    pub ks: f64,
    pub min_gini_pass: f64,
    pub min_gini_warn: f64,
    pub status: Status,
}

/// AUC (Mann-Whitney, average ranks for ties), Gini = 2·AUC − 1, Kolmogorov-Smirnov.
/// Convention: a higher score means a higher risk.
pub fn discrimination(
    score: &[f64],
    default: &[bool],
    min_gini_pass: f64,
    min_gini_warn: f64,
) -> Option<Discrimination> {
    let n = score.len();
    if n == 0 || n != default.len() || score.iter().any(|s| !s.is_finite()) {
        return None;
    }
    let nd = default.iter().filter(|d| **d).count();
    let nn = n - nd;
    if nd == 0 || nn == 0 {
        return None;
    }
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| score[a].partial_cmp(&score[b]).unwrap());

    // Average ranks and KS, walking through groups of ties.
    let (mut rank_sum_def, mut cum_d, mut cum_n, mut ks) = (0.0, 0.0, 0.0, 0f64);
    let mut i = 0;
    while i < n {
        let mut j = i;
        while j + 1 < n && score[idx[j + 1]] == score[idx[i]] {
            j += 1;
        }
        let avg_rank = (i + j) as f64 / 2.0 + 1.0;
        for &k in &idx[i..=j] {
            if default[k] {
                rank_sum_def += avg_rank;
                cum_d += 1.0;
            } else {
                cum_n += 1.0;
            }
        }
        ks = ks.max((cum_d / nd as f64 - cum_n / nn as f64).abs());
        i = j + 1;
    }
    let (ndf, nnf) = (nd as f64, nn as f64);
    let auc = (rank_sum_def - ndf * (ndf + 1.0) / 2.0) / (ndf * nnf);
    let gini = 2.0 * auc - 1.0;
    let status = if gini >= min_gini_pass {
        Status::Pass
    } else if gini >= min_gini_warn {
        Status::Warn
    } else {
        Status::Fail
    };
    Some(Discrimination {
        observations: n,
        defaults: nd,
        auc,
        gini,
        ks,
        min_gini_pass,
        min_gini_warn,
        status,
    })
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GradeCalibration {
    pub grade: i64,
    pub obligors: usize,
    pub defaults: usize,
    pub mean_pd: f64,
    pub observed_dr: f64,
    /// One-sided binomial test, H0: the PD is not underestimated.
    pub p_value: f64,
    pub status: Status,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Calibration {
    pub grades: Vec<GradeCalibration>,
    pub hosmer_lemeshow: f64,
    pub hl_df: usize,
    pub hl_p_value: f64,
    pub status: Status,
}

/// Calibration by rating grade: binomial test per grade and overall Hosmer-Lemeshow
/// (out-of-sample validation: degrees of freedom = number of grades).
pub fn calibration(grade: &[f64], pd: &[f64], default: &[bool]) -> Option<Calibration> {
    if grade.is_empty() || grade.len() != pd.len() || pd.len() != default.len() {
        return None;
    }
    if pd.iter().any(|p| !(0.0..=1.0).contains(p)) {
        return None;
    }
    let mut groups: BTreeMap<i64, (usize, usize, f64)> = BTreeMap::new();
    for i in 0..grade.len() {
        let g = groups.entry(grade[i].round() as i64).or_insert((0, 0, 0.0));
        g.0 += 1;
        g.1 += default[i] as usize;
        g.2 += pd[i];
    }
    let mut hl = 0.0;
    let grades: Vec<GradeCalibration> = groups
        .into_iter()
        .map(|(g, (n, d, sum_pd))| {
            let p = sum_pd / n as f64;
            let expected = n as f64 * p;
            if p > 0.0 && p < 1.0 {
                hl += (d as f64 - expected).powi(2) / (expected * (1.0 - p));
            }
            let p_value = binom_sf(d as u64, n as u64, p);
            GradeCalibration {
                grade: g,
                obligors: n,
                defaults: d,
                mean_pd: p,
                observed_dr: d as f64 / n as f64,
                p_value,
                status: Status::from_p_value(p_value),
            }
        })
        .collect();
    let df = grades.len();
    let hl_p_value = chi2_sf(hl, df as f64);
    let status = grades
        .iter()
        .map(|g| g.status)
        .fold(Status::from_p_value(hl_p_value), Status::max);
    Some(Calibration {
        grades,
        hosmer_lemeshow: hl,
        hl_df: df,
        hl_p_value,
        status,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn perfect_and_random_ranking() {
        let s = [0.1, 0.2, 0.3, 0.4];
        let perfect = discrimination(&s, &[false, false, true, true], 0.4, 0.25).unwrap();
        assert!((perfect.gini - 1.0).abs() < 1e-12 && (perfect.ks - 1.0).abs() < 1e-12);
        let ties = discrimination(&[0.5; 4], &[false, true, false, true], 0.4, 0.25).unwrap();
        assert!(ties.gini.abs() < 1e-12 && ties.status == Status::Fail);
    }

    #[test]
    fn calibration_flags_underestimated_grade() {
        // Grade 1: PD 1%, 10 defaults out of 1000 (in line); grade 2: PD 1%, 40 defaults out of 1000.
        let mut grade = vec![];
        let mut def = vec![];
        for g in [1.0, 2.0] {
            let d = if g == 1.0 { 10 } else { 40 };
            for i in 0..1000 {
                grade.push(g);
                def.push(i < d);
            }
        }
        let c = calibration(&grade, &vec![0.01; 2000], &def).unwrap();
        assert_eq!(c.grades[0].status, Status::Pass);
        assert_eq!(c.grades[1].status, Status::Fail);
        assert_eq!(c.status, Status::Fail);
    }
}
