//! IFRS 9 staging: recomputes each facility's stage from the SICR criteria and the default
//! definition, compares it with the reported stage, and measures how well stage 2 anticipated
//! the defaults that followed.
//!
//! Rules, in order: stage 3 if the facility is in default or more than `dpd_stage3` days past due;
//! stage 2 if it is more than `dpd_stage2` days past due (the IFRS 9 backstop, 5.5.11), on a
//! watch list or forborne, or if its PD has increased significantly since origination: the ratio
//! of the current to the origination PD at least `relative_threshold` and the increase at least
//! `absolute_threshold`, unless the current PD is at most `low_credit_risk_pd` (5.5.10); stage 1
//! otherwise.

use crate::Status;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct StagingRules {
    pub dpd_stage2: f64,
    pub dpd_stage3: f64,
    pub relative_threshold: f64,
    pub absolute_threshold: f64,
    pub low_credit_risk_pd: Option<f64>,
    /// Share of understaged facilities above which the check fails.
    pub tolerance: f64,
}

impl Default for StagingRules {
    fn default() -> Self {
        Self {
            dpd_stage2: 30.0,
            dpd_stage3: 90.0,
            relative_threshold: 2.0,
            absolute_threshold: 0.0,
            low_credit_risk_pd: None,
            tolerance: 0.0,
        }
    }
}

/// One facility; `flag` is a watch list or forbearance flag, `defaulted_next_12m` the outcome.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct StagingRecord {
    pub reported_stage: u8,
    pub defaulted: bool,
    pub dpd: f64,
    pub pd_origination: f64,
    pub pd_current: f64,
    pub flag: bool,
    pub defaulted_next_12m: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Mismatch {
    pub index: usize,
    pub reported: u8,
    pub expected: u8,
    pub reasons: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct StageEffectiveness {
    /// Facilities in each reported stage (1, 2) at the reporting date.
    pub facilities: [usize; 2],
    /// Defaults of the following 12 months, by reported stage (1, 2).
    pub defaults: [usize; 2],
    pub default_rate: [f64; 2],
    /// Share of the defaults that were in stage 2 beforehand.
    pub stage2_hit_rate: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct StagingCheck {
    pub facilities: usize,
    /// `confusion[i][j]`: reported stage i + 1, recomputed stage j + 1.
    pub confusion: [[usize; 3]; 3],
    pub agreement: f64,
    pub understaged: usize,
    pub overstaged: usize,
    /// Facilities meeting each criterion: defaulted, dpd > stage 3 threshold, then, for the others,
    /// dpd > stage 2 threshold, flag, PD increase.
    pub criteria: Vec<(String, usize)>,
    /// The first 50 mismatches.
    pub mismatches: Vec<Mismatch>,
    pub effectiveness: Option<StageEffectiveness>,
    pub rules: StagingRules,
    pub status: Status,
}

/// Recomputed stage of a facility, with the criteria it meets.
pub fn expected_stage(r: &StagingRecord, rules: &StagingRules) -> (u8, Vec<&'static str>) {
    let mut reasons = vec![];
    if r.defaulted {
        reasons.push("defaulted");
    }
    if r.dpd > rules.dpd_stage3 {
        reasons.push("dpd_stage3");
    }
    if !reasons.is_empty() {
        return (3, reasons);
    }
    if r.dpd > rules.dpd_stage2 {
        reasons.push("dpd_stage2");
    }
    if r.flag {
        reasons.push("flag");
    }
    let low_risk = rules.low_credit_risk_pd.is_some_and(|x| r.pd_current <= x);
    if !low_risk
        && r.pd_current >= rules.relative_threshold * r.pd_origination
        && r.pd_current - r.pd_origination >= rules.absolute_threshold
        && r.pd_current > r.pd_origination
    {
        reasons.push("pd_increase");
    }
    (if reasons.is_empty() { 1 } else { 2 }, reasons)
}

pub fn staging_check(records: &[StagingRecord], rules: &StagingRules) -> Result<StagingCheck, String> {
    if records.is_empty() {
        return Err("no facility".into());
    }
    let mut confusion = [[0usize; 3]; 3];
    let (mut under, mut over, mut mismatches) = (0, 0, vec![]);
    let names = ["defaulted", "dpd_stage3", "dpd_stage2", "flag", "pd_increase"];
    let mut counts = [0usize; 5];
    for (i, r) in records.iter().enumerate() {
        if !(1..=3).contains(&r.reported_stage) {
            return Err(format!("facility {i}: stage {} (1, 2 or 3)", r.reported_stage));
        }
        if ![r.dpd, r.pd_origination, r.pd_current].iter().all(|x| x.is_finite()) {
            return Err(format!("facility {i}: non-finite days past due or PD"));
        }
        let (e, reasons) = expected_stage(r, rules);
        for (k, n) in names.iter().enumerate() {
            counts[k] += reasons.contains(n) as usize;
        }
        confusion[r.reported_stage as usize - 1][e as usize - 1] += 1;
        if r.reported_stage != e {
            if r.reported_stage < e {
                under += 1;
            } else {
                over += 1;
            }
            if mismatches.len() < 50 {
                mismatches.push(Mismatch {
                    index: i,
                    reported: r.reported_stage,
                    expected: e,
                    reasons: reasons.iter().map(|s| s.to_string()).collect(),
                });
            }
        }
    }
    let n = records.len();
    let agreement = (0..3).map(|k| confusion[k][k]).sum::<usize>() as f64 / n as f64;
    let performing: Vec<&StagingRecord> = records.iter().filter(|r| r.reported_stage < 3).collect();
    let effectiveness = if !performing.is_empty() && performing.iter().all(|r| r.defaulted_next_12m.is_some()) {
        let (mut fac, mut def) = ([0usize; 2], [0usize; 2]);
        for r in &performing {
            let s = r.reported_stage as usize - 1;
            fac[s] += 1;
            def[s] += r.defaulted_next_12m.unwrap() as usize;
        }
        let rate = |k: usize| {
            if fac[k] == 0 {
                f64::NAN
            } else {
                def[k] as f64 / fac[k] as f64
            }
        };
        let total = def[0] + def[1];
        Some(StageEffectiveness {
            facilities: fac,
            defaults: def,
            default_rate: [rate(0), rate(1)],
            stage2_hit_rate: if total == 0 {
                f64::NAN
            } else {
                def[1] as f64 / total as f64
            },
        })
    } else {
        None
    };
    let status = if under as f64 > rules.tolerance * n as f64 {
        Status::Fail
    } else if under + over > 0 {
        Status::Warn
    } else {
        Status::Pass
    };
    Ok(StagingCheck {
        facilities: n,
        confusion,
        agreement,
        understaged: under,
        overstaged: over,
        criteria: names.iter().zip(counts).map(|(s, c)| (s.to_string(), c)).collect(),
        mismatches,
        effectiveness,
        rules: rules.clone(),
        status,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rec(stage: u8, dpd: f64, pd0: f64, pd: f64) -> StagingRecord {
        StagingRecord {
            reported_stage: stage,
            dpd,
            pd_origination: pd0,
            pd_current: pd,
            ..Default::default()
        }
    }

    #[test]
    fn rules_and_mismatches() {
        let rules = StagingRules {
            absolute_threshold: 0.005,
            low_credit_risk_pd: Some(0.003),
            ..Default::default()
        };
        let records = vec![
            rec(1, 0.0, 0.01, 0.015),  // ratio 1.5: stage 1
            rec(1, 0.0, 0.01, 0.025),  // ratio 2.5, +1.5%: stage 2, understaged
            rec(1, 0.0, 0.001, 0.003), // ratio 3, but low credit risk: stage 1
            rec(1, 0.0, 0.002, 0.004), // ratio 2, +0.2% only: stage 1
            rec(2, 31.0, 0.01, 0.01),  // backstop: stage 2
            rec(2, 91.0, 0.01, 0.01),  // > 90 days: stage 3, understaged
            rec(3, 10.0, 0.01, 0.01),  // reported 3, nothing: overstaged
        ];
        let c = staging_check(&records, &rules).unwrap();
        assert_eq!((c.understaged, c.overstaged), (2, 1));
        assert_eq!(c.mismatches.iter().map(|m| m.index).collect::<Vec<_>>(), vec![1, 5, 6]);
        assert_eq!(c.mismatches[0].reasons, vec!["pd_increase"]);
        assert_eq!(c.confusion[1][2], 1);
        assert_eq!(c.status, Status::Fail);
        let lenient = StagingRules {
            tolerance: 0.5,
            ..rules
        };
        assert_eq!(staging_check(&records, &lenient).unwrap().status, Status::Warn);
    }

    #[test]
    fn effectiveness_counts_defaults_by_stage() {
        let mut r = vec![rec(1, 0.0, 0.01, 0.01); 8];
        r.extend(vec![rec(2, 40.0, 0.01, 0.01); 2]);
        for (i, x) in r.iter_mut().enumerate() {
            x.defaulted_next_12m = Some(i == 0 || i >= 8);
        }
        let e = staging_check(&r, &StagingRules::default())
            .unwrap()
            .effectiveness
            .unwrap();
        assert_eq!((e.facilities, e.defaults), ([8, 2], [1, 2]));
        assert!((e.stage2_hit_rate - 2.0 / 3.0).abs() < 1e-15);
    }
}
