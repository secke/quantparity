//! Single-curve yield curves and interest rate swaps.
//!
//! Sequential bootstrap on deposits (simple rate) and par swaps (fixed leg paid at
//! `fixed_frequency`), log-linear interpolation of discount factors (piecewise-constant forwards),
//! flat-forward extrapolation. Times are in years and coupon accrual fractions are 1/frequency
//! (no calendar, no day count).

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum CurveInstrument {
    Deposit {
        maturity: f64,
        rate: f64,
    },
    Swap {
        maturity: f64,
        rate: f64,
        fixed_frequency: u32,
    },
}

impl CurveInstrument {
    pub fn maturity(&self) -> f64 {
        match self {
            CurveInstrument::Deposit { maturity, .. } | CurveInstrument::Swap { maturity, .. } => *maturity,
        }
    }
    pub fn quote(&self) -> f64 {
        match self {
            CurveInstrument::Deposit { rate, .. } | CurveInstrument::Swap { rate, .. } => *rate,
        }
    }
    pub fn label(&self) -> String {
        match self {
            CurveInstrument::Deposit { maturity, .. } => format!("DEP {maturity}Y"),
            CurveInstrument::Swap { maturity, .. } => format!("IRS {maturity}Y"),
        }
    }
    fn bumped(&self, d: f64) -> Self {
        let mut x = self.clone();
        match &mut x {
            CurveInstrument::Deposit { rate, .. } | CurveInstrument::Swap { rate, .. } => *rate += d,
        }
        x
    }
}

fn two() -> u32 {
    2
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CurveSpec {
    pub instruments: Vec<CurveInstrument>,
    /// Floating leg frequency (informational in a single-curve setting).
    #[serde(default = "two")]
    pub float_frequency: u32,
}

#[derive(Debug, Clone)]
pub struct Curve {
    times: Vec<f64>,
    log_df: Vec<f64>,
    pub spec: CurveSpec,
}

/// Integer number of periods `t × f`.
pub fn periods(t: f64, f: u32) -> Result<usize, String> {
    if f == 0 {
        return Err("zero frequency".into());
    }
    let n = (t * f as f64).round();
    if n < 1.0 || (n - t * f as f64).abs() > 1e-9 {
        return Err(format!("maturity {t} is not a multiple of the period 1/{f}"));
    }
    Ok(n as usize)
}

impl Curve {
    pub fn discount(&self, t: f64) -> f64 {
        if t <= 0.0 {
            return 1.0;
        }
        let n = self.times.len();
        let i = self.times.partition_point(|x| *x < t).clamp(1, n - 1);
        let (t0, t1, l0, l1) = (self.times[i - 1], self.times[i], self.log_df[i - 1], self.log_df[i]);
        (l0 + (l1 - l0) * (t - t0) / (t1 - t0)).exp()
    }

    /// Continuously compounded zero rate.
    pub fn zero_rate(&self, t: f64) -> f64 {
        if t <= 0.0 {
            return self.forward_rate(0.0, 1e-6);
        }
        -self.discount(t).ln() / t
    }

    /// Continuously compounded forward rate between `t1` and `t2`.
    pub fn forward_rate(&self, t1: f64, t2: f64) -> f64 {
        (self.discount(t1).ln() - self.discount(t2).ln()) / (t2 - t1)
    }

    /// Annuity Σ αᵢ P(tᵢ) of a fixed leg with frequency `f` up to `t`.
    pub fn annuity(&self, t: f64, f: u32) -> Result<f64, String> {
        let n = periods(t, f)?;
        Ok((1..=n).map(|i| self.discount(i as f64 / f as f64)).sum::<f64>() / f as f64)
    }

    /// Par swap rate (spot start, single curve).
    pub fn par_swap_rate(&self, t: f64, f: u32) -> Result<f64, String> {
        Ok((1.0 - self.discount(t)) / self.annuity(t, f)?)
    }

    fn model_rate(&self, inst: &CurveInstrument) -> Result<f64, String> {
        match inst {
            CurveInstrument::Deposit { maturity, .. } => Ok((1.0 / self.discount(*maturity) - 1.0) / maturity),
            CurveInstrument::Swap {
                maturity,
                fixed_frequency,
                ..
            } => self.par_swap_rate(*maturity, *fixed_frequency),
        }
    }

    pub fn bootstrap(spec: &CurveSpec) -> Result<Self, String> {
        let mut inst = spec.instruments.clone();
        if inst.is_empty() {
            return Err("curve has no instruments".into());
        }
        inst.sort_by(|a, b| a.maturity().partial_cmp(&b.maturity()).unwrap());
        let mut c = Curve {
            times: vec![0.0],
            log_df: vec![0.0],
            spec: spec.clone(),
        };
        for k in &inst {
            let t = k.maturity();
            if !(t.is_finite() && t > *c.times.last().unwrap()) {
                return Err(format!("{}: maturities not strictly increasing", k.label()));
            }
            if let CurveInstrument::Swap { fixed_frequency, .. } = k {
                periods(t, *fixed_frequency)?;
            }
            c.times.push(t);
            c.log_df.push(0.0);
            // Bisection on the zero rate of the new pillar: the model rate is increasing in z.
            let (mut lo, mut hi) = (-0.5f64, 1.0f64);
            let err_at = |c: &mut Curve, z: f64| -> Result<f64, String> {
                *c.log_df.last_mut().unwrap() = -z * t;
                Ok(c.model_rate(k)? - k.quote())
            };
            if err_at(&mut c, lo)? > 0.0 || err_at(&mut c, hi)? < 0.0 {
                return Err(format!("{}: quote outside the solvable range", k.label()));
            }
            for _ in 0..200 {
                let mid = 0.5 * (lo + hi);
                if err_at(&mut c, mid)? > 0.0 {
                    hi = mid
                } else {
                    lo = mid
                }
                if hi - lo < 1e-16 {
                    break;
                }
            }
            err_at(&mut c, 0.5 * (lo + hi))?;
        }
        Ok(c)
    }

    /// Curve rebuilt after shifting every quote by `bp` basis points.
    pub fn bumped(&self, bp: f64) -> Result<Self, String> {
        let spec = CurveSpec {
            instruments: self.spec.instruments.iter().map(|i| i.bumped(bp * 1e-4)).collect(),
            float_frequency: self.spec.float_frequency,
        };
        Self::bootstrap(&spec)
    }

    /// Repricing of the input instruments: (instrument, maturity, quote, model rate).
    pub fn repricing(&self) -> Result<Vec<(String, f64, f64, f64)>, String> {
        self.spec
            .instruments
            .iter()
            .map(|k| Ok((k.label(), k.maturity(), k.quote(), self.model_rate(k)?)))
            .collect()
    }

    pub fn pillars(&self) -> &[f64] {
        &self.times[1..]
    }
}

/// Named market curves: simple curves (time in years) and dated multi-curves.
#[derive(Debug, Clone, Default)]
pub struct Market {
    pub curves: BTreeMap<String, Curve>,
    pub as_of: Option<crate::dates::Date>,
    pub dated: BTreeMap<String, crate::curves::multicurve::DatedCurve>,
    dated_specs: BTreeMap<String, crate::curves::multicurve::DatedCurveSpec>,
}

impl Market {
    /// `{"as_of": "YYYY-MM-DD", "curves": {"<name>": {...}}}`. A curve with `kind` (`ois`, `ibor`)
    /// is dated and requires `as_of`; otherwise it is a simple curve (maturities in years).
    pub fn from_value(v: &Value) -> Result<Self, String> {
        let raw: BTreeMap<String, Value> = serde_json::from_value(v.get("curves").cloned().unwrap_or(Value::Null))
            .map_err(|e| format!("invalid market: {e}"))?;
        let mut curves = BTreeMap::new();
        let mut dated_specs = BTreeMap::new();
        for (n, spec) in raw {
            if spec.get("kind").is_some() {
                let s: crate::curves::multicurve::DatedCurveSpec =
                    serde_json::from_value(spec).map_err(|e| format!("curve {n}: {e}"))?;
                dated_specs.insert(n, s);
            } else {
                let s: CurveSpec = serde_json::from_value(spec).map_err(|e| format!("curve {n}: {e}"))?;
                curves.insert(n.clone(), Curve::bootstrap(&s).map_err(|e| format!("curve {n}: {e}"))?);
            }
        }
        let as_of = match v.get("as_of").and_then(Value::as_str) {
            Some(d) => Some(crate::dates::Date::parse(d)?),
            None if !dated_specs.is_empty() => return Err("`as_of` is required for dated curves".into()),
            None => None,
        };
        let dated = match as_of {
            Some(d) => crate::curves::multicurve::build_all(d, &dated_specs)?,
            None => BTreeMap::new(),
        };
        Ok(Self {
            curves,
            as_of,
            dated,
            dated_specs,
        })
    }

    /// Market rebuilt after shifting every dated quote by `bp` basis points.
    pub fn bumped_dated(&self, bp: f64) -> Result<Self, String> {
        let specs: BTreeMap<_, _> = self
            .dated_specs
            .iter()
            .map(|(k, s)| (k.clone(), s.bumped(bp * 1e-4)))
            .collect();
        let as_of = self.as_of.ok_or("market has no date")?;
        Ok(Self {
            curves: self.curves.clone(),
            as_of: self.as_of,
            dated: crate::curves::multicurve::build_all(as_of, &specs)?,
            dated_specs: specs,
        })
    }

    pub fn dated_curve(&self, name: &str) -> Result<&crate::curves::multicurve::DatedCurve, String> {
        self.dated
            .get(name)
            .ok_or_else(|| format!("dated curve `{name}` not found in market"))
    }

    pub fn curve(&self, name: &str) -> Result<&Curve, String> {
        self.curves
            .get(name)
            .ok_or_else(|| format!("curve `{name}` not found in market"))
    }
}

/// Curve build report: instrument repricing and term structure.
pub fn curve_report(market: &Market, name: &str, tolerance: f64, step_years: f64) -> Result<Value, String> {
    use serde_json::json;
    let status = |e: f64| {
        if e <= tolerance {
            "pass"
        } else if e <= 100.0 * tolerance {
            "warn"
        } else {
            "fail"
        }
    };
    if let Some(c) = market.dated.get(name) {
        let disc = c.discount_curve_name().map(|d| market.dated_curve(d)).transpose()?;
        let rep = c.repricing(disc)?;
        let max_err = rep.iter().map(|(_, _, q, m)| (q - m).abs()).fold(0.0, f64::max);
        let pillars: Vec<Value> = rep
            .iter()
            .map(|(l, d, q, m)| {
                json!({"instrument": l, "maturity": d.to_string(), "t": c.time(*d), "quote": q, "model_rate": m,
                       "repricing_error": (q - m).abs(), "discount_factor": c.discount(*d), "zero_rate": c.zero_rate(*d)})
            })
            .collect();
        let last = *c.pillars().last().ok_or("empty curve")?;
        let months = (step_years * 12.0).round().max(1.0) as i64;
        let mut grid = vec![];
        let mut k = 1;
        loop {
            let d = c.as_of.add_months(k * months);
            if d > last {
                break;
            }
            let prev = c.as_of.add_months((k - 1) * months);
            let fwd = (c.discount(prev) / c.discount(d)).ln() / (c.time(d) - c.time(prev));
            grid.push(
                json!({"t": c.time(d), "date": d.to_string(), "discount_factor": c.discount(d),
                             "zero_rate": c.zero_rate(d), "forward_rate": fwd}),
            );
            k += 1;
        }
        let negative = grid
            .iter()
            .filter(|g| g["forward_rate"].as_f64().unwrap_or(0.0) < 0.0)
            .count();
        return Ok(
            json!({"curve": name, "as_of": c.as_of.to_string(), "spot": c.spot().to_string(),
                         "discount_curve": c.discount_curve_name(), "pillars": pillars,
                         "max_repricing_error": max_err, "tolerance": tolerance, "negative_forwards": negative,
                         "term_structure": grid, "status": status(max_err)}),
        );
    }
    let c = market.curve(name)?;
    let rep = c.repricing()?;
    let max_err = rep.iter().map(|(_, _, q, m)| (q - m).abs()).fold(0.0, f64::max);
    let pillars: Vec<Value> = rep
        .iter()
        .map(|(l, t, q, m)| {
            json!({"instrument": l, "maturity": t, "quote": q, "model_rate": m, "repricing_error": (q - m).abs(),
                   "discount_factor": c.discount(*t), "zero_rate": c.zero_rate(*t)})
        })
        .collect();
    let last = c.pillars().last().copied().unwrap_or(0.0);
    let step = step_years.max(1e-3);
    let grid: Vec<Value> = (1..=(last / step).round() as usize)
        .map(|i| {
            let t = i as f64 * step;
            json!({"t": t, "discount_factor": c.discount(t), "zero_rate": c.zero_rate(t),
                   "forward_rate": c.forward_rate(t - step, t)})
        })
        .collect();
    let negative = grid
        .iter()
        .filter(|g| g["forward_rate"].as_f64().unwrap_or(0.0) < 0.0)
        .count();
    Ok(
        json!({"curve": name, "pillars": pillars, "max_repricing_error": max_err, "tolerance": tolerance,
              "negative_forwards": negative, "term_structure": grid, "status": status(max_err)}),
    )
}

/// NPV of a spot-starting vanilla interest rate swap, paying or receiving fixed.
pub fn swap_npv(
    c: &Curve,
    notional: f64,
    fixed_rate: f64,
    maturity: f64,
    f: u32,
    pay_fixed: bool,
) -> Result<f64, String> {
    let float_leg = notional * (1.0 - c.discount(maturity));
    let fixed_leg = notional * fixed_rate * c.annuity(maturity, f)?;
    Ok(if pay_fixed {
        float_leg - fixed_leg
    } else {
        fixed_leg - float_leg
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec() -> CurveSpec {
        serde_json::from_value(serde_json::json!({"instruments": [
            {"type": "deposit", "maturity": 0.25, "rate": 0.031},
            {"type": "deposit", "maturity": 0.5, "rate": 0.0315},
            {"type": "swap", "maturity": 1, "rate": 0.0322, "fixed_frequency": 1},
            {"type": "swap", "maturity": 2, "rate": 0.0330, "fixed_frequency": 1},
            {"type": "swap", "maturity": 5, "rate": 0.0345, "fixed_frequency": 1},
            {"type": "swap", "maturity": 10, "rate": 0.0362, "fixed_frequency": 1},
            {"type": "swap", "maturity": 30, "rate": 0.0371, "fixed_frequency": 1}
        ]}))
        .unwrap()
    }

    #[test]
    fn bootstrap_reprices_inputs() {
        let c = Curve::bootstrap(&spec()).unwrap();
        for (label, _, q, m) in c.repricing().unwrap() {
            assert!((q - m).abs() < 1e-13, "{label}: {q} vs {m}");
        }
        assert_eq!(c.pillars().len(), 7);
    }

    #[test]
    fn par_swap_has_zero_npv_and_dv01_sign() {
        let c = Curve::bootstrap(&spec()).unwrap();
        let k = c.par_swap_rate(7.0, 1).unwrap();
        assert!(swap_npv(&c, 1e7, k, 7.0, 1, true).unwrap().abs() < 1e-6);
        let up = c.bumped(1.0).unwrap();
        let dv01 = swap_npv(&up, 1e7, k, 7.0, 1, true).unwrap() - swap_npv(&c, 1e7, k, 7.0, 1, true).unwrap();
        // Fixed payer: gains when rates rise, ~ notional × annuity × 1bp.
        let approx = 1e7 * c.annuity(7.0, 1).unwrap() * 1e-4;
        assert!(dv01 > 0.0 && (dv01 / approx - 1.0).abs() < 0.05, "{dv01} vs {approx}");
    }

    #[test]
    fn flat_curve_closed_form() {
        // 1Y deposit at simple rate r, then annual swaps at the same rate: flat curve, annual compounding.
        let r: f64 = 0.04;
        let s: CurveSpec = serde_json::from_value(serde_json::json!({"instruments": [
            {"type": "deposit", "maturity": 1, "rate": r},
            {"type": "swap", "maturity": 2, "rate": r, "fixed_frequency": 1},
            {"type": "swap", "maturity": 3, "rate": r, "fixed_frequency": 1}
        ]}))
        .unwrap();
        let c = Curve::bootstrap(&s).unwrap();
        for t in [1.0, 2.0, 3.0] {
            assert!((c.discount(t) - (1.0 + r).powf(-t)).abs() < 1e-14);
        }
        // Log-linear interpolation = constant forward between pillars.
        assert!((c.discount(1.5) - (1.0 + r).powf(-1.5)).abs() < 1e-14);
    }

    #[test]
    fn rejects_bad_curves() {
        let bad = |j| Curve::bootstrap(&serde_json::from_value::<CurveSpec>(j).unwrap()).is_err();
        assert!(bad(serde_json::json!({"instruments": []})));
        assert!(bad(serde_json::json!({"instruments": [
            {"type": "deposit", "maturity": 1, "rate": 0.03}, {"type": "deposit", "maturity": 1, "rate": 0.03}]})));
        assert!(bad(
            serde_json::json!({"instruments": [{"type": "swap", "maturity": 1.3, "rate": 0.03, "fixed_frequency": 1}]})
        ));
    }
}
