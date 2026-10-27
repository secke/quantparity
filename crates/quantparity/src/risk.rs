//! Portfolio market risk: factor shocks, full revaluation under scenarios, historical VaR and
//! ES, sensitivities, stress tests and the FRTB P&L attribution test.
//!
//! Risk factors and shock units:
//! - `SPOT:<underlying>`: relative change of the spot or forward (0.01 = +1%);
//! - `VOL:<underlying>`: absolute change of the volatility (0.01 = +1 vol point);
//! - `CURVE:<curve>`: parallel shift of every quote of the curve, in basis points;
//! - `CURVE:<curve>:<pillar>`: shift of a single quote (`5Y`, `6M`…), in basis points;
//! - `YIELD:<issuer>`: shift of the yield of the issuer's bonds, in basis points.

use crate::curves::Market;
use crate::Status;
use crate::{price_in, Instrument, Trade};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};

pub type Shifts = BTreeMap<String, f64>;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Scenario {
    pub name: String,
    pub shifts: Shifts,
    /// Maximum allowed loss (positive amount), for stress tests.
    #[serde(default)]
    pub limit: Option<f64>,
}

fn trim_num(x: f64) -> String {
    let s = format!("{x}");
    s.strip_suffix(".0").map(str::to_string).unwrap_or(s)
}

/// Pillar of a curve instrument: tenor (dated curve) or maturity in years (`5Y`).
fn bucket(inst: &Value) -> Option<String> {
    if let Some(t) = inst.get("tenor").and_then(Value::as_str) {
        return Some(t.to_ascii_uppercase());
    }
    inst.get("maturity")
        .and_then(Value::as_f64)
        .map(|m| format!("{}Y", trim_num(m)))
}

fn shock_trade(t: &mut Trade, kind: &str, v: f64) {
    match (&mut t.instrument, kind) {
        (Instrument::EuropeanOption { spot, .. } | Instrument::AmericanOption { spot, .. }, "SPOT") => *spot *= 1.0 + v,
        (Instrument::Black76 { forward, .. }, "SPOT") => *forward *= 1.0 + v,
        (
            Instrument::EuropeanOption { vol, .. }
            | Instrument::AmericanOption { vol, .. }
            | Instrument::Black76 { vol, .. },
            "VOL",
        ) => *vol = (*vol + v).max(1e-4),
        (Instrument::FixedRateBond { yield_rate, .. }, "YIELD") => *yield_rate += v * 1e-4,
        _ => {}
    }
}

/// Applies shocks to positions and market data (market JSON).
pub fn apply(trades: &[Trade], market: &Value, shifts: &Shifts) -> Result<(Vec<Trade>, Value), String> {
    let mut ts = trades.to_vec();
    let mut mk = market.clone();
    for (factor, &v) in shifts {
        if !v.is_finite() {
            return Err(format!("factor `{factor}`: non-numeric shock"));
        }
        if v == 0.0 {
            continue;
        }
        let (kind, rest) = factor
            .split_once(':')
            .ok_or_else(|| format!("factor `{factor}`: expected TYPE:name"))?;
        match kind {
            "SPOT" | "VOL" | "YIELD" => {
                for t in ts.iter_mut().filter(|t| t.underlying.as_deref() == Some(rest)) {
                    shock_trade(t, kind, v);
                }
            }
            "CURVE" => {
                let (curve, pillar) = match rest.split_once(':') {
                    Some((c, b)) => (c, Some(b.to_ascii_uppercase())),
                    None => (rest, None),
                };
                let insts = mk
                    .get_mut("curves")
                    .and_then(|c| c.get_mut(curve))
                    .and_then(|c| c.get_mut("instruments"))
                    .and_then(Value::as_array_mut)
                    .ok_or_else(|| format!("factor `{factor}`: curve `{curve}` not found in market"))?;
                let mut hit = false;
                for i in insts.iter_mut() {
                    if pillar.is_none() || bucket(i) == pillar {
                        if let Some(r) = i.get("rate").and_then(Value::as_f64) {
                            i["rate"] = json!(r + v * 1e-4);
                            hit = true;
                        }
                    }
                }
                if !hit {
                    return Err(format!("factor `{factor}`: pillar not found in curve"));
                }
            }
            _ => return Err(format!("facteur `{factor}`: unknown type (SPOT, VOL, CURVE, YIELD)")),
        }
    }
    Ok((ts, mk))
}

fn market_of(v: &Value) -> Result<Market, String> {
    if v.is_null() {
        Ok(Market::default())
    } else {
        Market::from_value(v)
    }
}

/// Value of each position (unit price × quantity).
pub fn values(trades: &[Trade], market: &Value) -> Result<Vec<f64>, String> {
    let m = market_of(market)?;
    trades
        .iter()
        .map(|t| {
            price_in(t, &m)
                .map(|p| p * t.quantity)
                .map_err(|e| format!("{}: {e}", t.id))
        })
        .collect()
}

/// Runs `f` on every element in parallel (result order preserved).
fn par_map<T: Sync, R: Send>(items: &[T], f: impl Fn(&T) -> Result<R, String> + Sync) -> Result<Vec<R>, String> {
    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .min(items.len().max(1));
    let chunk = items.len().div_ceil(threads).max(1);
    let parts: Vec<Result<Vec<R>, String>> = std::thread::scope(|s| {
        let handles: Vec<_> = items
            .chunks(chunk)
            .map(|c| s.spawn(|| c.iter().map(&f).collect::<Result<Vec<R>, String>>()))
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().unwrap_or_else(|_| Err("computation interrupted".into())))
            .collect()
    });
    let mut out = Vec::with_capacity(items.len());
    for p in parts {
        out.extend(p?);
    }
    Ok(out)
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ScenarioPnl {
    pub name: String,
    pub pnl: f64,
    pub by_trade: Vec<f64>,
}

/// Full revaluation: P&L of each scenario relative to the base value.
pub fn revalue(
    trades: &[Trade],
    market: &Value,
    scenarios: &[Scenario],
) -> Result<(Vec<f64>, Vec<ScenarioPnl>), String> {
    let base = values(trades, market)?;
    let pnl = par_map(scenarios, |sc| {
        let (ts, mk) = apply(trades, market, &sc.shifts).map_err(|e| format!("{}: {e}", sc.name))?;
        let v = values(&ts, &mk).map_err(|e| format!("{}: {e}", sc.name))?;
        let by_trade: Vec<f64> = v.iter().zip(&base).map(|(a, b)| a - b).collect();
        Ok(ScenarioPnl {
            name: sc.name.clone(),
            pnl: by_trade.iter().sum(),
            by_trade,
        })
    })?;
    Ok((base, pnl))
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct VarEs {
    pub observations: usize,
    pub var_confidence: f64,
    pub var: f64,
    pub es_confidence: f64,
    pub es: f64,
}

/// VaR (k-th worst P&L, k = ⌈n(1-α)⌉) and ES (mean of the losses beyond it), as positive amounts.
pub fn var_es(pnl: &[f64], var_confidence: f64, es_confidence: f64) -> Option<VarEs> {
    if pnl.is_empty() || pnl.iter().any(|x| !x.is_finite()) {
        return None;
    }
    let mut s = pnl.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let k = |a: f64| (((1.0 - a) * s.len() as f64 - 1e-9).ceil().max(1.0) as usize).min(s.len());
    let kv = k(var_confidence);
    let ke = k(es_confidence);
    Some(VarEs {
        observations: s.len(),
        var_confidence,
        var: -s[kv - 1],
        es_confidence,
        es: -s[..ke].iter().sum::<f64>() / ke as f64,
    })
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Sensitivity {
    pub factor: String,
    /// Unit shock size (0.01 for SPOT and VOL, 1bp for CURVE and YIELD).
    pub bump: f64,
    pub unit: String,
    /// Portfolio value change for a unit shock (central difference).
    pub value: f64,
}

/// Risk factors the portfolio is exposed to.
pub fn risk_factors(trades: &[Trade], market: &Value) -> Vec<(String, f64, &'static str)> {
    let mut f: BTreeSet<(String, u8)> = BTreeSet::new();
    for t in trades {
        let Some(u) = &t.underlying else { continue };
        match t.instrument {
            Instrument::EuropeanOption { .. } | Instrument::AmericanOption { .. } | Instrument::Black76 { .. } => {
                f.insert((format!("SPOT:{u}"), 0));
                f.insert((format!("VOL:{u}"), 1));
            }
            Instrument::FixedRateBond { .. } => {
                f.insert((format!("YIELD:{u}"), 2));
            }
            _ => {}
        }
    }
    if let Some(curves) = market.get("curves").and_then(Value::as_object) {
        for (c, spec) in curves {
            for i in spec.get("instruments").and_then(Value::as_array).into_iter().flatten() {
                if let Some(b) = bucket(i) {
                    f.insert((format!("CURVE:{c}:{b}"), 2));
                }
            }
        }
    }
    f.into_iter()
        .map(|(name, k)| match k {
            0 => (name, 0.01, "per +1% of the underlying"),
            1 => (name, 0.01, "per +1 vol point"),
            _ => (name, 1.0, "per +1 basis point"),
        })
        .collect()
}

/// Portfolio sensitivities by central difference on every factor.
pub fn sensitivities(trades: &[Trade], market: &Value) -> Result<Vec<Sensitivity>, String> {
    let factors = risk_factors(trades, market);
    par_map(&factors, |(name, bump, unit)| {
        let v = |h: f64| -> Result<f64, String> {
            let (ts, mk) = apply(trades, market, &Shifts::from([(name.clone(), h)]))?;
            Ok(values(&ts, &mk)?.iter().sum())
        };
        Ok(Sensitivity {
            factor: name.clone(),
            bump: *bump,
            unit: unit.to_string(),
            value: (v(*bump)? - v(-*bump)?) / 2.0,
        })
    })
}

/// Risk-theoretical P&L: sensitivities × shocks (linear approximation).
/// A parallel shock `CURVE:<c>` applies to every pillar `CURVE:<c>:<p>`.
pub fn risk_theoretical_pnl(sens: &[Sensitivity], scenarios: &[Scenario]) -> Vec<f64> {
    scenarios
        .iter()
        .map(|sc| {
            sc.shifts
                .iter()
                .map(|(f, m)| {
                    let parallel = f.starts_with("CURVE:") && f.matches(':').count() == 1;
                    sens.iter()
                        .filter(|s| {
                            if parallel {
                                s.factor.starts_with(&format!("{f}:"))
                            } else {
                                &s.factor == f
                            }
                        })
                        .map(|s| s.value * m / s.bump)
                        .sum::<f64>()
                })
                .sum()
        })
        .collect()
}

fn ranks(x: &[f64]) -> Vec<f64> {
    let mut idx: Vec<usize> = (0..x.len()).collect();
    idx.sort_by(|&a, &b| x[a].partial_cmp(&x[b]).unwrap());
    let mut r = vec![0.0; x.len()];
    let mut i = 0;
    while i < idx.len() {
        let mut j = i;
        while j + 1 < idx.len() && x[idx[j + 1]] == x[idx[i]] {
            j += 1;
        }
        let avg = (i + j) as f64 / 2.0 + 1.0;
        idx[i..=j].iter().for_each(|&k| r[k] = avg);
        i = j + 1;
    }
    r
}

/// Spearman rank correlation.
pub fn spearman(a: &[f64], b: &[f64]) -> Option<f64> {
    if a.len() != b.len() || a.len() < 3 {
        return None;
    }
    let (ra, rb) = (ranks(a), ranks(b));
    let n = a.len() as f64;
    let (ma, mb) = (ra.iter().sum::<f64>() / n, rb.iter().sum::<f64>() / n);
    let cov: f64 = ra.iter().zip(&rb).map(|(x, y)| (x - ma) * (y - mb)).sum();
    let va: f64 = ra.iter().map(|x| (x - ma).powi(2)).sum();
    let vb: f64 = rb.iter().map(|y| (y - mb).powi(2)).sum();
    (va > 0.0 && vb > 0.0).then(|| cov / (va * vb).sqrt())
}

/// Two-sample Kolmogorov-Smirnov statistic.
pub fn ks_statistic(a: &[f64], b: &[f64]) -> Option<f64> {
    if a.is_empty() || b.is_empty() {
        return None;
    }
    let mut x = a.to_vec();
    let mut y = b.to_vec();
    x.sort_by(|p, q| p.partial_cmp(q).unwrap());
    y.sort_by(|p, q| p.partial_cmp(q).unwrap());
    let (mut i, mut j, mut d) = (0usize, 0usize, 0f64);
    while i < x.len() && j < y.len() {
        let v = x[i].min(y[j]);
        while i < x.len() && x[i] <= v {
            i += 1;
        }
        while j < y.len() && y[j] <= v {
            j += 1;
        }
        d = d.max((i as f64 / x.len() as f64 - j as f64 / y.len() as f64).abs());
    }
    Some(d)
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PnlAttribution {
    pub observations: usize,
    pub spearman: f64,
    pub ks: f64,
    /// `green`, `amber` or `red` (FRTB thresholds: ρ ≥ 0.80 and KS ≤ 0.09; ρ ≥ 0.70 and KS ≤ 0.12).
    pub zone: String,
    pub min_observations: usize,
    pub unexplained_std: f64,
    pub hpl_std: f64,
    pub status: Status,
}

/// P&L attribution test: hypothetical P&L (full revaluation) vs risk-theoretical P&L.
pub fn pnl_attribution(hpl: &[f64], rtpl: &[f64], min_observations: usize) -> Option<PnlAttribution> {
    let rho = spearman(hpl, rtpl)?;
    let ks = ks_statistic(hpl, rtpl)?;
    let (zone, mut status) = if rho >= 0.80 && ks <= 0.09 {
        ("green", Status::Pass)
    } else if rho >= 0.70 && ks <= 0.12 {
        ("amber", Status::Warn)
    } else {
        ("red", Status::Fail)
    };
    if hpl.len() < min_observations {
        status = status.max(Status::Warn);
    }
    let std = |v: &[f64]| {
        let m = v.iter().sum::<f64>() / v.len() as f64;
        (v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / v.len() as f64).sqrt()
    };
    let diff: Vec<f64> = hpl.iter().zip(rtpl).map(|(a, b)| a - b).collect();
    Some(PnlAttribution {
        observations: hpl.len(),
        spearman: rho,
        ks,
        zone: zone.into(),
        min_observations,
        unexplained_std: std(&diff),
        hpl_std: std(hpl),
        status,
    })
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct StressResult {
    pub name: String,
    pub pnl: f64,
    pub limit: Option<f64>,
    /// Loss relative to the limit.
    pub utilisation: Option<f64>,
    pub status: Status,
}

/// Stress tests: `fail` if the loss exceeds the limit, `warn` above 80% of the limit.
pub fn stress(trades: &[Trade], market: &Value, scenarios: &[Scenario]) -> Result<Vec<StressResult>, String> {
    let (_, pnl) = revalue(trades, market, scenarios)?;
    Ok(scenarios
        .iter()
        .zip(pnl)
        .map(|(sc, p)| {
            let utilisation = sc.limit.filter(|l| *l > 0.0).map(|l| (-p.pnl).max(0.0) / l);
            let status = match utilisation {
                Some(u) if u > 1.0 => Status::Fail,
                Some(u) if u > 0.8 => Status::Warn,
                _ => Status::Pass,
            };
            StressResult {
                name: sc.name.clone(),
                pnl: p.pnl,
                limit: sc.limit,
                utilisation,
                status,
            }
        })
        .collect())
}

/// Historical scenarios from a table `{factor: [shocks…]}` (one row per date).
pub fn scenarios_from_table(table: &Value, label: Option<&str>) -> Result<Vec<Scenario>, String> {
    let obj = table.as_object().ok_or("expected a scenario table ({factor: [...]})")?;
    let n = obj.values().next().and_then(Value::as_array).map_or(0, Vec::len);
    let mut out = vec![];
    for i in 0..n {
        let mut shifts = Shifts::new();
        let mut name = format!("scenario {}", i + 1);
        for (k, col) in obj {
            let v = col
                .get(i)
                .and_then(Value::as_f64)
                .ok_or_else(|| format!("column `{k}`: missing value at row {}", i + 1))?;
            if Some(k.as_str()) == label {
                name = format!("day {}", trim_num(v));
            } else {
                shifts.insert(k.clone(), v);
            }
        }
        out.push(Scenario {
            name,
            shifts,
            limit: None,
        });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bsm;
    use crate::OptionType::Call;

    fn book() -> (Vec<Trade>, Value) {
        let trades: Vec<Trade> = serde_json::from_value(json!([
            {"id": "C1", "product": "european_option", "underlying": "IDX", "quantity": 10, "option_type": "call",
             "spot": 100, "strike": 100, "rate": 0.02, "vol": 0.2, "maturity": 1},
            {"id": "B1", "product": "fixed_rate_bond", "underlying": "GOV", "quantity": -2, "coupon_rate": 0.03,
             "maturity": 5, "frequency": 1, "yield_rate": 0.03},
            {"id": "Z1", "product": "zero_coupon_bond", "face": 1000, "maturity": 3, "curve": "EUR"}
        ]))
        .unwrap();
        let market = json!({"curves": {"EUR": {"instruments": [
            {"type": "deposit", "maturity": 1, "rate": 0.02},
            {"type": "swap", "maturity": 2, "rate": 0.022, "fixed_frequency": 1},
            {"type": "swap", "maturity": 5, "rate": 0.025, "fixed_frequency": 1}]}}});
        (trades, market)
    }

    #[test]
    fn shocks_apply_to_the_right_factors() {
        let (t, m) = book();
        let (ts, mk) = apply(
            &t,
            &m,
            &Shifts::from([("SPOT:IDX".into(), -0.1), ("CURVE:EUR:2Y".into(), 10.0)]),
        )
        .unwrap();
        match ts[0].instrument {
            Instrument::EuropeanOption { spot, .. } => assert!((spot - 90.0).abs() < 1e-12),
            _ => unreachable!(),
        }
        assert!((mk["curves"]["EUR"]["instruments"][1]["rate"].as_f64().unwrap() - 0.023).abs() < 1e-12);
        assert_eq!(mk["curves"]["EUR"]["instruments"][0]["rate"], json!(0.02));
        assert!(apply(&t, &m, &Shifts::from([("FX:EURUSD".into(), 0.01)])).is_err());
        assert!(apply(&t, &m, &Shifts::from([("CURVE:EUR:7Y".into(), 1.0)])).is_err());
    }

    #[test]
    fn var_es_on_known_sample() {
        let pnl: Vec<f64> = (1..=100).map(|i| -(i as f64)).collect();
        let r = var_es(&pnl, 0.99, 0.975).unwrap();
        assert_eq!(r.var, 100.0);
        assert!((r.es - (100.0 + 99.0 + 98.0) / 3.0).abs() < 1e-12);
    }

    #[test]
    fn delta_matches_black_scholes() {
        let (t, m) = book();
        let s = sensitivities(&t, &m).unwrap();
        let d = s.iter().find(|x| x.factor == "SPOT:IDX").unwrap().value;
        let bs = bsm(Call, 100.0, 100.0, 0.02, 0.0, 0.2, 1.0).unwrap();
        assert!((d - 10.0 * bs.delta * 1.0).abs() < 1e-3, "{d}");
        let vega = s.iter().find(|x| x.factor == "VOL:IDX").unwrap().value;
        assert!((vega - 10.0 * bs.vega * 0.01).abs() < 1e-4, "{vega}");
        // Short bond position: gains when yields rise.
        assert!(s.iter().find(|x| x.factor == "YIELD:GOV").unwrap().value > 0.0);
        assert!(s.iter().any(|x| x.factor == "CURVE:EUR:5Y"));
    }

    #[test]
    fn theoretical_pnl_tracks_full_revaluation_for_small_moves() {
        let (t, m) = book();
        let sens = sensitivities(&t, &m).unwrap();
        let sc = vec![Scenario {
            name: "small shock".into(),
            shifts: Shifts::from([
                ("SPOT:IDX".into(), 0.002),
                ("CURVE:EUR".into(), 1.0),
                ("YIELD:GOV".into(), -0.5),
            ]),
            limit: None,
        }];
        let (_, full) = revalue(&t, &m, &sc).unwrap();
        let lin = risk_theoretical_pnl(&sens, &sc);
        assert!(
            (full[0].pnl - lin[0]).abs() < 0.01 * full[0].pnl.abs().max(1e-6),
            "{} vs {}",
            full[0].pnl,
            lin[0]
        );
    }

    #[test]
    fn pla_zones() {
        let h: Vec<f64> = (0..300).map(|i| ((i * 37 % 101) as f64 - 50.0) / 10.0).collect();
        let same = pnl_attribution(&h, &h, 250).unwrap();
        assert_eq!((same.zone.as_str(), same.status), ("green", Status::Pass));
        let noise: Vec<f64> = (0..300).map(|i| ((i * 53 % 97) as f64 - 48.0) / 10.0).collect();
        assert_eq!(pnl_attribution(&h, &noise, 250).unwrap().zone, "red");
        assert_eq!(pnl_attribution(&h[..100], &h[..100], 250).unwrap().status, Status::Warn);
        assert!((spearman(&[1.0, 2.0, 3.0], &[10.0, 20.0, 30.0]).unwrap() - 1.0).abs() < 1e-12);
        assert_eq!(ks_statistic(&[1.0, 2.0], &[3.0, 4.0]), Some(1.0));
    }

    #[test]
    fn stress_limits() {
        let (t, m) = book();
        let sc = vec![
            Scenario {
                name: "krach".into(),
                shifts: Shifts::from([("SPOT:IDX".into(), -0.3)]),
                limit: Some(50.0),
            },
            Scenario {
                name: "mild".into(),
                shifts: Shifts::from([("SPOT:IDX".into(), -0.01)]),
                limit: Some(50.0),
            },
        ];
        let r = stress(&t, &m, &sc).unwrap();
        assert_eq!((r[0].status, r[1].status), (Status::Fail, Status::Pass));
    }
}
