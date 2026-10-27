//! Dated multi-curve framework: an OIS (€STR) curve for discounting and Euribor projection curves
//! discounted on OIS. Market conventions: calendar, spot lag, Modified Following, Act/360 and
//! 30/360 day counts, schedules generated backward. Curve time is Act/365F from the valuation
//! date; discount factors are interpolated log-linearly.

use crate::dates::{backward_schedule, Calendar, Convention, Date, DayCount, Tenor};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

fn two() -> i64 {
    2
}
fn one() -> u32 {
    1
}
fn act360() -> DayCount {
    DayCount::Act360
}
fn thirty360() -> DayCount {
    DayCount::Thirty360
}
fn mf() -> Convention {
    Convention::ModifiedFollowing
}
fn target() -> Calendar {
    Calendar::Target
}
fn six_m() -> String {
    "6M".into()
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Quote {
    pub tenor: String,
    pub rate: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum IborInstrument {
    Deposit { tenor: String, rate: f64 },
    Swap { tenor: String, rate: f64 },
}

impl IborInstrument {
    fn tenor(&self) -> &str {
        match self {
            IborInstrument::Deposit { tenor, .. } | IborInstrument::Swap { tenor, .. } => tenor,
        }
    }
    fn rate(&self) -> f64 {
        match self {
            IborInstrument::Deposit { rate, .. } | IborInstrument::Swap { rate, .. } => *rate,
        }
    }
    fn bump(&mut self, d: f64) {
        match self {
            IborInstrument::Deposit { rate, .. } | IborInstrument::Swap { rate, .. } => *rate += d,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum DatedCurveSpec {
    /// Par OIS swaps: annual fixed leg (single period if ≤ 1Y), compounded floating leg.
    Ois {
        #[serde(default = "target")]
        calendar: Calendar,
        #[serde(default = "two")]
        settlement_days: i64,
        #[serde(default = "act360")]
        day_count: DayCount,
        instruments: Vec<Quote>,
    },
    /// Projection curve of an Ibor index, discounted on `discount_curve`.
    Ibor {
        discount_curve: String,
        #[serde(default = "target")]
        calendar: Calendar,
        #[serde(default = "two")]
        settlement_days: i64,
        #[serde(default = "six_m")]
        index_tenor: String,
        #[serde(default = "act360")]
        float_day_count: DayCount,
        #[serde(default = "one")]
        fixed_frequency: u32,
        #[serde(default = "thirty360")]
        fixed_day_count: DayCount,
        #[serde(default = "mf")]
        convention: Convention,
        instruments: Vec<IborInstrument>,
    },
}

impl DatedCurveSpec {
    pub fn bumped(&self, d: f64) -> Self {
        let mut s = self.clone();
        match &mut s {
            DatedCurveSpec::Ois { instruments, .. } => instruments.iter_mut().for_each(|q| q.rate += d),
            DatedCurveSpec::Ibor { instruments, .. } => instruments.iter_mut().for_each(|q| q.bump(d)),
        }
        s
    }

    fn calendar_and_lag(&self) -> (Calendar, i64) {
        match self {
            DatedCurveSpec::Ois {
                calendar,
                settlement_days,
                ..
            }
            | DatedCurveSpec::Ibor {
                calendar,
                settlement_days,
                ..
            } => (*calendar, *settlement_days),
        }
    }
}

#[derive(Debug, Clone)]
pub struct DatedCurve {
    pub as_of: Date,
    dates: Vec<Date>,
    times: Vec<f64>,
    log_df: Vec<f64>,
    pub spec: DatedCurveSpec,
}

impl DatedCurve {
    pub fn time(&self, d: Date) -> f64 {
        DayCount::Act365F.year_fraction(self.as_of, d)
    }

    pub fn discount(&self, d: Date) -> f64 {
        let t = self.time(d);
        if t <= 0.0 {
            return 1.0;
        }
        let n = self.times.len();
        let i = self.times.partition_point(|x| *x < t).clamp(1, n - 1);
        let (t0, t1, l0, l1) = (self.times[i - 1], self.times[i], self.log_df[i - 1], self.log_df[i]);
        (l0 + (l1 - l0) * (t - t0) / (t1 - t0)).exp()
    }

    pub fn zero_rate(&self, d: Date) -> f64 {
        let t = self.time(d);
        if t <= 0.0 {
            0.0
        } else {
            -self.discount(d).ln() / t
        }
    }

    pub fn pillars(&self) -> &[Date] {
        &self.dates[1..]
    }

    pub fn spot(&self) -> Date {
        let (cal, lag) = self.spec.calendar_and_lag();
        cal.advance_business_days(self.as_of, lag)
    }
}

/// Legs of a spot-starting Ibor-vs-fixed swap: (annuity Σ τ·P, floating leg value, maturity).
#[allow(clippy::too_many_arguments)]
pub fn ibor_swap_legs(
    fwd: &DatedCurve,
    disc: &DatedCurve,
    spot: Date,
    tenor: Tenor,
    cal: Calendar,
    conv: Convention,
    fixed_months: i64,
    fixed_dc: DayCount,
    float_months: i64,
    float_dc: DayCount,
) -> Result<(f64, f64, Date), String> {
    let end = tenor.add_to(spot);
    let fixed = backward_schedule(spot, end, fixed_months, cal, conv, conv)?;
    let float = backward_schedule(spot, end, float_months, cal, conv, conv)?;
    let annuity: f64 = fixed
        .windows(2)
        .map(|w| fixed_dc.year_fraction(w[0], w[1]) * disc.discount(w[1]))
        .sum();
    let float_pv: f64 = float
        .windows(2)
        .map(|w| {
            let tau = float_dc.year_fraction(w[0], w[1]);
            let fwd_rate = (fwd.discount(w[0]) / fwd.discount(w[1]) - 1.0) / tau;
            fwd_rate * tau * disc.discount(w[1])
        })
        .sum();
    Ok((annuity, float_pv, *fixed.last().unwrap()))
}

struct Pillar {
    label: String,
    maturity: Date,
    quote: f64,
}

/// Model rate of a calibration instrument (`c` is the curve being built).
fn model_rate(c: &DatedCurve, disc: Option<&DatedCurve>, k: usize) -> Result<f64, String> {
    let spot = c.spot();
    match &c.spec {
        DatedCurveSpec::Ois {
            calendar,
            day_count,
            instruments,
            ..
        } => {
            let tenor = Tenor::parse(&instruments[k].tenor)?;
            let end = tenor.add_to(spot);
            let months = tenor.months().unwrap_or(1);
            let sched = if months <= 12 {
                vec![spot, calendar.adjust(end, Convention::ModifiedFollowing)]
            } else {
                backward_schedule(
                    spot,
                    end,
                    12,
                    *calendar,
                    Convention::ModifiedFollowing,
                    Convention::ModifiedFollowing,
                )?
            };
            let annuity: f64 = sched
                .windows(2)
                .map(|w| day_count.year_fraction(w[0], w[1]) * c.discount(w[1]))
                .sum();
            Ok((c.discount(spot) - c.discount(*sched.last().unwrap())) / annuity)
        }
        DatedCurveSpec::Ibor {
            calendar,
            index_tenor,
            float_day_count,
            fixed_frequency,
            fixed_day_count,
            convention,
            instruments,
            ..
        } => {
            let idx = Tenor::parse(index_tenor)?.months()?;
            match &instruments[k] {
                IborInstrument::Deposit { tenor, .. } => {
                    let end = index_maturity(*calendar, *convention, spot, Tenor::parse(tenor)?);
                    let tau = float_day_count.year_fraction(spot, end);
                    Ok((c.discount(spot) / c.discount(end) - 1.0) / tau)
                }
                IborInstrument::Swap { tenor, .. } => {
                    let d = disc.ok_or("missing discount curve")?;
                    let (a, f, _) = ibor_swap_legs(
                        c,
                        d,
                        spot,
                        Tenor::parse(tenor)?,
                        *calendar,
                        *convention,
                        12 / *fixed_frequency as i64,
                        *fixed_day_count,
                        idx,
                        *float_day_count,
                    )?;
                    Ok(f / a)
                }
            }
        }
    }
}

/// Maturity of a deposit / Ibor fixing (end-of-month rule included).
fn index_maturity(cal: Calendar, conv: Convention, start: Date, tenor: Tenor) -> Date {
    let raw = tenor.add_to(start);
    if matches!(tenor, Tenor::Months(_)) && cal.is_end_of_month(start) {
        cal.end_of_month(raw)
    } else {
        cal.adjust(raw, conv)
    }
}

fn pillars_of(c: &DatedCurve) -> Result<Vec<Pillar>, String> {
    let spot = c.spot();
    let (cal, _) = c.spec.calendar_and_lag();
    let mut out = vec![];
    match &c.spec {
        DatedCurveSpec::Ois { instruments, .. } => {
            for q in instruments {
                let end = cal.adjust(Tenor::parse(&q.tenor)?.add_to(spot), Convention::ModifiedFollowing);
                out.push(Pillar {
                    label: format!("OIS {}", q.tenor),
                    maturity: end,
                    quote: q.rate,
                });
            }
        }
        DatedCurveSpec::Ibor {
            instruments,
            convention,
            ..
        } => {
            for q in instruments {
                let t = Tenor::parse(q.tenor())?;
                let (label, end) = match q {
                    IborInstrument::Deposit { .. } => {
                        (format!("DEP {}", q.tenor()), index_maturity(cal, *convention, spot, t))
                    }
                    IborInstrument::Swap { .. } => {
                        (format!("IRS {}", q.tenor()), cal.adjust(t.add_to(spot), *convention))
                    }
                };
                out.push(Pillar {
                    label,
                    maturity: end,
                    quote: q.rate(),
                });
            }
        }
    }
    Ok(out)
}

impl DatedCurve {
    pub fn bootstrap(as_of: Date, spec: &DatedCurveSpec, disc: Option<&DatedCurve>) -> Result<Self, String> {
        let mut c = DatedCurve {
            as_of,
            dates: vec![as_of],
            times: vec![0.0],
            log_df: vec![0.0],
            spec: spec.clone(),
        };
        let mut pillars: Vec<(usize, Pillar)> = pillars_of(&c)?.into_iter().enumerate().collect();
        if pillars.is_empty() {
            return Err("curve has no instruments".into());
        }
        pillars.sort_by_key(|(_, p)| p.maturity);
        for (k, p) in &pillars {
            if p.maturity <= *c.dates.last().unwrap() {
                return Err(format!("{}: maturities not strictly increasing", p.label));
            }
            let t = c.time(p.maturity);
            c.dates.push(p.maturity);
            c.times.push(t);
            c.log_df.push(0.0);
            let err_at = |c: &mut DatedCurve, z: f64| -> Result<f64, String> {
                *c.log_df.last_mut().unwrap() = -z * t;
                Ok(model_rate(c, disc, *k)? - p.quote)
            };
            let (mut lo, mut hi) = (-0.5f64, 1.0f64);
            if err_at(&mut c, lo)? > 0.0 || err_at(&mut c, hi)? < 0.0 {
                return Err(format!("{}: quote outside the solvable range", p.label));
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

    /// Repricing: (instrument, maturity, quote, model rate).
    pub fn repricing(&self, disc: Option<&DatedCurve>) -> Result<Vec<(String, Date, f64, f64)>, String> {
        pillars_of(self)?
            .into_iter()
            .enumerate()
            .map(|(k, p)| Ok((p.label, p.maturity, p.quote, model_rate(self, disc, k)?)))
            .collect()
    }

    pub fn discount_curve_name(&self) -> Option<&str> {
        match &self.spec {
            DatedCurveSpec::Ibor { discount_curve, .. } => Some(discount_curve),
            DatedCurveSpec::Ois { .. } => None,
        }
    }
}

/// Builds dated curves in dependency order (OIS before the curves discounted on it).
pub fn build_all(
    as_of: Date,
    specs: &BTreeMap<String, DatedCurveSpec>,
) -> Result<BTreeMap<String, DatedCurve>, String> {
    let mut built: BTreeMap<String, DatedCurve> = BTreeMap::new();
    while built.len() < specs.len() {
        let before = built.len();
        for (name, spec) in specs {
            if built.contains_key(name) {
                continue;
            }
            let disc = match spec {
                DatedCurveSpec::Ois { .. } => None,
                DatedCurveSpec::Ibor { discount_curve, .. } => match built.get(discount_curve) {
                    Some(d) => Some(d.clone()),
                    None if specs.contains_key(discount_curve) => continue,
                    None => return Err(format!("curve {name}: discount curve `{discount_curve}` not found")),
                },
            };
            let c = DatedCurve::bootstrap(as_of, spec, disc.as_ref()).map_err(|e| format!("curve {name}: {e}"))?;
            built.insert(name.clone(), c);
        }
        if built.len() == before {
            return Err("circular dependencies between curves".into());
        }
    }
    Ok(built)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn market() -> BTreeMap<String, DatedCurveSpec> {
        serde_json::from_value(json!({
            "ESTR": {"kind": "ois", "instruments": [
                {"tenor": "1M", "rate": 0.0190}, {"tenor": "6M", "rate": 0.0192}, {"tenor": "1Y", "rate": 0.0195},
                {"tenor": "2Y", "rate": 0.0201}, {"tenor": "5Y", "rate": 0.0222}, {"tenor": "10Y", "rate": 0.0248}]},
            "EUR6M": {"kind": "ibor", "discount_curve": "ESTR", "instruments": [
                {"type": "deposit", "tenor": "6M", "rate": 0.0218}, {"type": "swap", "tenor": "1Y", "rate": 0.0221},
                {"type": "swap", "tenor": "2Y", "rate": 0.0226}, {"type": "swap", "tenor": "5Y", "rate": 0.0249},
                {"type": "swap", "tenor": "10Y", "rate": 0.0274}]}
        }))
        .unwrap()
    }

    #[test]
    fn multicurve_bootstrap_reprices() {
        let as_of = Date::parse("2026-01-13").unwrap();
        let curves = build_all(as_of, &market()).unwrap();
        let estr = &curves["ESTR"];
        let eur = &curves["EUR6M"];
        assert_eq!(estr.spot().to_string(), "2026-01-15");
        for (l, _, q, m) in estr
            .repricing(None)
            .unwrap()
            .into_iter()
            .chain(eur.repricing(Some(estr)).unwrap())
        {
            assert!((q - m).abs() < 1e-13, "{l}: {q} vs {m}");
        }
        // Positive Euribor-€STR basis: the projection curve sits above OIS.
        let d10 = Date::parse("2036-01-15").unwrap();
        assert!(eur.zero_rate(d10) > estr.zero_rate(d10));
    }

    #[test]
    fn missing_or_circular_dependencies() {
        let as_of = Date::parse("2026-01-13").unwrap();
        let mut m = market();
        m.remove("ESTR");
        assert!(build_all(as_of, &m).unwrap_err().contains("not found"));
    }
}
