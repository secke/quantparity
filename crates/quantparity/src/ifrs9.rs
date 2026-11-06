//! IFRS 9 expected credit loss: PD term structures, exposure profiles and the ECL of a facility.
//!
//! Conventions: periods of `1 / periods_per_year` years from the reporting date; a default in
//! period k (k = 1, 2, …) happens at its end, with the exposure and LGD of that period, and the loss
//! is discounted from that date at the effective interest rate: DF_k = (1 + eir)^(-k / periods_per_year).
//! Stage 1 takes the defaults of the first 12 months, stages 2 and 3 those of the remaining life;
//! a stage 3 facility is in default, its ECL is LGD × EAD of the first period, undiscounted.

use serde::{Deserialize, Serialize};

/// Cumulative PDs at the end of each whole year from the reporting date: `cumulative[y - 1]` is
/// P(default by year y). Between two years the hazard rate is constant; after the last year the
/// conditional PD of the last year applies to every following year.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PdCurve {
    pub cumulative: Vec<f64>,
}

impl PdCurve {
    pub fn new(cumulative: Vec<f64>) -> Result<Self, String> {
        if cumulative.is_empty() {
            return Err("empty PD curve".into());
        }
        let mut prev = 0.0;
        for &c in &cumulative {
            if !(prev..=1.0).contains(&c) {
                return Err(format!(
                    "cumulative PDs must be non-decreasing in [0, 1], got {cumulative:?}"
                ));
            }
            prev = c;
        }
        Ok(Self { cumulative })
    }

    /// A constant hazard rate that gives `pd_12m` over one year.
    pub fn flat(pd_12m: f64) -> Result<Self, String> {
        Self::new(vec![pd_12m])
    }

    /// Survival probability at `t` years.
    pub fn survival(&self, t: f64) -> f64 {
        if t <= 0.0 {
            return 1.0;
        }
        let s = |y: usize| if y == 0 { 1.0 } else { 1.0 - self.cumulative[y - 1] };
        let n = self.cumulative.len();
        let y = t.floor() as usize;
        if y < n {
            let (a, b) = (s(y), s(y + 1));
            if a == 0.0 {
                return 0.0;
            }
            a * (b / a).powf(t - y as f64)
        } else {
            // Beyond the curve: the survival rate of the last year, every year.
            let (a, b) = (s(n - 1), s(n));
            if a == 0.0 {
                return 0.0;
            }
            b * (b / a).powf(t - n as f64)
        }
    }

    /// Cumulative PD at the end of each of `periods` periods.
    pub fn cumulative_periods(&self, periods: usize, periods_per_year: u32) -> Vec<f64> {
        (1..=periods)
            .map(|k| 1.0 - self.survival(k as f64 / periods_per_year as f64))
            .collect()
    }
}

/// An annual rating migration matrix whose last state is default (absorbing).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MigrationMatrix {
    pub states: Vec<String>,
    pub p: Vec<Vec<f64>>,
}

impl MigrationMatrix {
    pub fn new(states: Vec<String>, p: Vec<Vec<f64>>) -> Result<Self, String> {
        let n = states.len();
        if n < 2 || p.len() != n || p.iter().any(|r| r.len() != n) {
            return Err(format!("a migration matrix over {n} states must be {n} × {n}"));
        }
        for (i, r) in p.iter().enumerate() {
            let sum: f64 = r.iter().sum();
            if r.iter().any(|x| !(0.0..=1.0).contains(x)) || (sum - 1.0).abs() > 1e-6 {
                return Err(format!(
                    "row `{}` is not a probability distribution (sum {sum})",
                    states[i]
                ));
            }
        }
        if (p[n - 1][n - 1] - 1.0).abs() > 1e-12 {
            return Err(format!("the last state, `{}`, must be absorbing", states[n - 1]));
        }
        Ok(Self { states, p })
    }

    /// Cohort estimate from observed one-year transitions (state indices, the last one is default).
    pub fn estimate(states: Vec<String>, from: &[usize], to: &[usize]) -> Result<Self, String> {
        let n = states.len();
        if from.len() != to.len() || from.iter().chain(to).any(|&s| s >= n) {
            return Err("transitions must be pairs of state indices".into());
        }
        let mut counts = vec![vec![0.0; n]; n];
        for (&a, &b) in from.iter().zip(to) {
            counts[a][b] += 1.0;
        }
        counts[n - 1] = (0..n).map(|j| (j == n - 1) as u8 as f64).collect();
        for (i, r) in counts.iter_mut().enumerate() {
            let total: f64 = r.iter().sum();
            if total == 0.0 {
                return Err(format!("no transition observed from `{}`", states[i]));
            }
            r.iter_mut().for_each(|x| *x /= total);
        }
        Self::new(states, counts)
    }

    /// Cumulative PDs over `years` years of a facility in state `from`: the default column of M^y.
    pub fn pd_curve(&self, from: usize, years: usize) -> Result<PdCurve, String> {
        let n = self.states.len();
        if from >= n - 1 {
            return Err(format!("no PD curve from state {from}"));
        }
        let mut row: Vec<f64> = (0..n).map(|j| (j == from) as u8 as f64).collect();
        let mut cumulative = Vec::with_capacity(years);
        for _ in 0..years {
            row = (0..n).map(|j| (0..n).map(|i| row[i] * self.p[i][j]).sum()).collect();
            cumulative.push(row[n - 1].min(1.0));
        }
        PdCurve::new(cumulative)
    }
}

/// How the exposure runs off until maturity.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Repayment {
    /// The whole balance until maturity.
    Bullet,
    /// Equal principal repayments.
    Linear,
    /// Equal instalments at the effective interest rate.
    Annuity,
}

impl Repayment {
    pub fn parse(s: &str) -> Result<Self, String> {
        match s {
            "bullet" => Ok(Self::Bullet),
            "linear" => Ok(Self::Linear),
            "annuity" => Ok(Self::Annuity),
            _ => Err(format!("unknown repayment `{s}` (bullet, linear, annuity)")),
        }
    }

    /// Exposure in each of the `periods` remaining periods: the balance outstanding at the start of
    /// the period, before its repayment (a default in a period loses the balance owed during it).
    pub fn profile(self, balance: f64, periods: usize, eir: f64, periods_per_year: u32) -> Vec<f64> {
        let n = periods as f64;
        match self {
            Self::Bullet => vec![balance; periods],
            Self::Linear => (0..periods).map(|k| balance * (1.0 - k as f64 / n)).collect(),
            Self::Annuity => {
                let r = (1.0 + eir).powf(1.0 / periods_per_year as f64) - 1.0;
                if r.abs() < 1e-14 {
                    return Self::Linear.profile(balance, periods, eir, periods_per_year);
                }
                // Balance after k instalments: B (1 - (1 + r)^-(n - k)) / (1 - (1 + r)^-n).
                let a = |k: f64| 1.0 - (1.0 + r).powf(-(n - k));
                (0..periods).map(|k| balance * a(k as f64) / a(0.0)).collect()
            }
        }
    }
}

/// Inputs of one facility, period by period from the reporting date.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EclInput {
    pub stage: u8,
    /// Cumulative PD at the end of each remaining period.
    pub cumulative_pd: Vec<f64>,
    /// LGD in each period, or one value for all.
    pub lgd: Vec<f64>,
    /// Exposure at default in each period, or one value for all.
    pub ead: Vec<f64>,
    /// Effective interest rate, annual compounding.
    pub eir: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Ecl {
    pub stage: u8,
    pub ecl_12m: f64,
    pub ecl_lifetime: f64,
    /// The ECL of the stage: 12 months in stage 1, lifetime in stages 2 and 3.
    pub ecl: f64,
    /// PD over the horizon of the stage.
    pub pd_horizon: f64,
}

/// ECL of a facility (conventions in the module documentation).
pub fn ecl(input: &EclInput, periods_per_year: u32) -> Result<Ecl, String> {
    let n = input.cumulative_pd.len();
    let per = |v: &[f64], what: &str| -> Result<Vec<f64>, String> {
        match v.len() {
            1 => Ok(vec![v[0]; n.max(1)]),
            l if l == n => Ok(v.to_vec()),
            l => Err(format!("{what}: {l} values for {n} periods")),
        }
    };
    let (lgd, ead) = (per(&input.lgd, "lgd")?, per(&input.ead, "ead")?);
    if !(1..=3).contains(&input.stage) {
        return Err(format!("stage {} (1, 2 or 3)", input.stage));
    }
    if periods_per_year == 0 || !input.eir.is_finite() || input.eir <= -1.0 {
        return Err("invalid periods per year or effective interest rate".into());
    }
    if lgd.iter().any(|x| !x.is_finite()) || ead.iter().any(|x| !x.is_finite() || *x < 0.0) {
        return Err("LGD and EAD must be finite, EAD non-negative".into());
    }
    if input.stage == 3 {
        let loss = lgd[0] * ead[0];
        return Ok(Ecl {
            stage: 3,
            ecl_12m: loss,
            ecl_lifetime: loss,
            ecl: loss,
            pd_horizon: 1.0,
        });
    }
    if n == 0 {
        return Err("no remaining period".into());
    }
    let mut prev = 0.0;
    for &c in &input.cumulative_pd {
        if !(prev..=1.0).contains(&c) {
            return Err("cumulative PDs must be non-decreasing in [0, 1]".into());
        }
        prev = c;
    }
    let h12 = (periods_per_year as usize).min(n);
    let (mut ecl_12m, mut ecl_lifetime, mut prev) = (0.0, 0.0, 0.0);
    for k in 0..n {
        let c = input.cumulative_pd[k];
        let df = (1.0 + input.eir).powf(-((k + 1) as f64) / periods_per_year as f64);
        let loss = (c - prev) * lgd[k] * ead[k] * df;
        prev = c;
        ecl_lifetime += loss;
        if k < h12 {
            ecl_12m += loss;
        }
    }
    let (ecl, pd_horizon) = if input.stage == 1 {
        (ecl_12m, input.cumulative_pd[h12 - 1])
    } else {
        (ecl_lifetime, input.cumulative_pd[n - 1])
    };
    Ok(Ecl {
        stage: input.stage,
        ecl_12m,
        ecl_lifetime,
        ecl,
        pd_horizon,
    })
}

/// A facility described by its balance, remaining term and a PD curve.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Facility {
    pub stage: u8,
    pub balance: f64,
    pub lgd: f64,
    pub eir: f64,
    /// Remaining periods to maturity.
    pub periods: usize,
    pub repayment: Repayment,
}

impl Facility {
    pub fn ecl(&self, pd: &PdCurve, periods_per_year: u32) -> Result<Ecl, String> {
        let periods = if self.stage == 3 {
            self.periods.max(1)
        } else {
            self.periods
        };
        ecl(
            &EclInput {
                stage: self.stage,
                cumulative_pd: pd.cumulative_periods(periods, periods_per_year),
                lgd: vec![self.lgd],
                ead: self
                    .repayment
                    .profile(self.balance, periods, self.eir, periods_per_year),
                eir: self.eir,
            },
            periods_per_year,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flat_hazard_and_closed_form() {
        // Bullet, annual periods, constant PD p: ECL_lifetime = LGD·B·Σ (1-p)^(k-1) p / (1+r)^k.
        let (p, lgd, b, r, n) = (0.02, 0.45, 1000.0, 0.05, 5);
        let pd = PdCurve::flat(p).unwrap();
        let f = Facility {
            stage: 2,
            balance: b,
            lgd,
            eir: r,
            periods: n,
            repayment: Repayment::Bullet,
        };
        let e = f.ecl(&pd, 1).unwrap();
        let exact: f64 = (1..=n)
            .map(|k| (1.0 - p).powi(k as i32 - 1) * p * lgd * b / (1.0 + r).powi(k as i32))
            .sum();
        assert!((e.ecl - exact).abs() < 1e-10, "{} {exact}", e.ecl);
        assert!((e.ecl_12m - p * lgd * b / (1.0 + r)).abs() < 1e-10);
        // Monthly periods: the 12-month PD is the same.
        let m = Facility {
            periods: 12 * n,
            ..f.clone()
        }
        .ecl(&pd, 12)
        .unwrap();
        assert!((pd.cumulative_periods(12, 12)[11] - p).abs() < 1e-14);
        // Defaults spread over the year are discounted less than at its end.
        assert!(m.ecl_12m > e.ecl_12m && m.ecl_12m < p * lgd * b);
        let s3 = Facility { stage: 3, ..f }.ecl(&pd, 12).unwrap();
        assert_eq!(s3.ecl, lgd * b);
    }

    #[test]
    fn pd_curve_interpolates_and_extrapolates() {
        let pd = PdCurve::new(vec![0.01, 0.03]).unwrap();
        assert!((1.0 - pd.survival(1.0) - 0.01).abs() < 1e-15);
        assert!((1.0 - pd.survival(2.0) - 0.03).abs() < 1e-15);
        // Year 3: the conditional PD of year 2, 1 - 0.97/0.99.
        let s3 = 0.97 * 0.97 / 0.99;
        assert!((pd.survival(3.0) - s3).abs() < 1e-15);
        assert!(PdCurve::new(vec![0.02, 0.01]).is_err());
    }

    #[test]
    fn migration_matrix_powers() {
        let s = |v: &[&str]| v.iter().map(|x| x.to_string()).collect::<Vec<_>>();
        let m = MigrationMatrix::new(
            s(&["A", "B", "D"]),
            vec![vec![0.9, 0.08, 0.02], vec![0.1, 0.8, 0.1], vec![0.0, 0.0, 1.0]],
        )
        .unwrap();
        let c = m.pd_curve(0, 2).unwrap().cumulative;
        assert!((c[0] - 0.02).abs() < 1e-15);
        assert!((c[1] - (0.02 + 0.9 * 0.02 + 0.08 * 0.1)).abs() < 1e-15);
        let e = MigrationMatrix::estimate(s(&["A", "B", "D"]), &[0, 0, 0, 1, 1], &[0, 1, 2, 1, 2]).unwrap();
        assert_eq!(e.p[0], vec![1.0 / 3.0; 3]);
        assert_eq!(e.p[1], vec![0.0, 0.5, 0.5]);
        assert!(MigrationMatrix::new(s(&["A", "D"]), vec![vec![0.9, 0.1], vec![0.1, 0.9]]).is_err());
    }

    #[test]
    fn annuity_profile() {
        // 12 monthly instalments at 12% a year: the balance before the last one is one instalment's principal.
        let p = Repayment::Annuity.profile(1000.0, 12, 0.12, 12);
        let r = 1.12f64.powf(1.0 / 12.0) - 1.0;
        let instalment = 1000.0 * r / (1.0 - (1.0 + r).powi(-12));
        assert!((p[0] - 1000.0).abs() < 1e-12);
        assert!((p[11] * (1.0 + r) - instalment).abs() < 1e-9);
        assert_eq!(
            Repayment::Linear.profile(100.0, 4, 0.0, 4),
            vec![100.0, 75.0, 50.0, 25.0]
        );
    }
}
