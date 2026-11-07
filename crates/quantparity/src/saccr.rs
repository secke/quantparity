//! Standardised approach for counterparty credit risk (SA-CCR): the exposure at default of a
//! netting set of derivatives, as in the Basel framework (CRE52, from BCBS 279) and its
//! transpositions (CRR2 article 274 and following, 12 CFR 217.132(c)).
//!
//! EAD = α (RC + multiplier × AddOn). The replacement cost RC is max(V − C, 0), and for a
//! margined netting set max(V − C, TH + MTA − NICA, 0). The add-on is the sum, by asset class,
//! of the add-ons of the hedging sets, built from the effective notional of each trade
//! D = δ × d × MF: supervisory delta, adjusted notional (the notional times the supervisory
//! duration for interest rate and credit trades) and maturity factor. The multiplier,
//! min(1, 5% + 95% exp((V − C) / (2 × 95% × AddOn))), recognises excess collateral and
//! out-of-the-money trades. The EAD of a margined netting set is capped at its unmargined EAD.
//!
//! Amounts are in one reporting currency; times are in years, and a year has 250 business days
//! for the maturity floor and the margin period of risk.

use crate::norm_cdf;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Business days in a year (maturity floor, margin period of risk).
pub const BUSINESS_DAYS: f64 = 250.0;
/// Shortest maturity recognised for an unmargined trade: 10 business days.
const MATURITY_FLOOR: f64 = 10.0 / BUSINESS_DAYS;
const FLOOR: f64 = 0.05;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[serde(rename_all = "lowercase")]
pub enum AssetClass {
    #[serde(alias = "interest_rate")]
    Ir,
    Fx,
    Credit,
    Equity,
    Commodity,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum OptionKind {
    Call,
    Put,
}

/// An option, or a swaption: the delta is that of a Black option on the underlying price P with
/// strike K, shifted by λ when rates can be negative: Φ(±d1), d1 = (ln((P + λ) / (K + λ)) + σ² T / 2) / (σ √T).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct OptionData {
    pub kind: OptionKind,
    pub bought: bool,
    pub price: f64,
    pub strike: f64,
    /// Latest exercise date, in years (T).
    pub exercise: f64,
    /// Supervisory volatility of the asset class by default.
    pub volatility: Option<f64>,
    #[serde(default)]
    pub shift: f64,
}

/// Attachment and detachment points of a CDO tranche: δ = ±15 / ((1 + 14 A)(1 + 14 D)).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Tranche {
    pub attachment: f64,
    pub detachment: f64,
}

/// One derivative of a netting set.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Trade {
    pub id: String,
    pub asset_class: AssetClass,
    /// Currency (interest rate), currency pair (FX), reference entity or index (credit, equity),
    /// commodity type (commodity: trades of one type offset each other fully).
    pub qualifier: String,
    /// Credit single name: AAA, AA, A, BBB, BB, B or CCC.
    pub rating: Option<String>,
    /// Credit or equity index.
    #[serde(default)]
    pub index: bool,
    /// Credit index: investment grade (true) or speculative grade.
    pub investment_grade: Option<bool>,
    /// Commodity: electricity, oil_gas, metals, agricultural or other.
    pub sector: Option<String>,
    /// Notional in the reporting currency: of the foreign leg (FX), price × units (equity, commodity).
    pub notional: f64,
    /// Long in the primary risk factor (pay fixed, protection bought, long the asset). Not used for options.
    pub long: Option<bool>,
    pub market_value: f64,
    /// Latest date the contract may be active, in years (M).
    pub maturity: f64,
    /// Interest rate and credit: start and end of the underlying period (S, E; default 0 and M).
    pub start: Option<f64>,
    pub end: Option<f64>,
    pub option: Option<OptionData>,
    pub tranche: Option<Tranche>,
    /// Basis transaction (two floating risk factors in one asset class): own hedging set, half the factor.
    #[serde(default)]
    pub basis: bool,
    /// Volatility transaction: own hedging set, five times the factor.
    #[serde(default)]
    pub volatility_trade: bool,
}

/// Margin agreement of a netting set.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Margin {
    #[serde(default)]
    pub threshold: f64,
    #[serde(default)]
    pub mta: f64,
    /// Net independent collateral amount: held minus posted (segregated) independent collateral.
    #[serde(default)]
    pub nica: f64,
    /// Margin period of risk in business days; otherwise floor + remargining period − 1.
    pub mpor_days: Option<f64>,
    #[serde(default = "one")]
    pub remargin_days: f64,
    /// 10 business days for bilateral trades, 5 for client cleared trades, 20 for large or disputed sets.
    #[serde(default = "ten")]
    pub mpor_floor_days: f64,
}

fn one() -> f64 {
    1.0
}
fn ten() -> f64 {
    10.0
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct NettingSet {
    pub id: String,
    pub trades: Vec<Trade>,
    /// Haircut value of the net collateral held (C), variation and independent.
    #[serde(default)]
    pub collateral: f64,
    pub margin: Option<Margin>,
}

/// Supervisory factor, correlation and option volatility of a trade (CRE52.72).
struct Supervisory {
    factor: f64,
    correlation: f64,
    volatility: f64,
}

fn supervisory(t: &Trade) -> Result<Supervisory, String> {
    let s = |factor, correlation, volatility| Supervisory {
        factor,
        correlation,
        volatility,
    };
    let base = match t.asset_class {
        AssetClass::Ir => s(0.005, 0.0, 0.5),
        AssetClass::Fx => s(0.04, 0.0, 0.15),
        AssetClass::Credit if t.index => match t.investment_grade {
            Some(true) => s(0.0038, 0.8, 0.8),
            Some(false) => s(0.0106, 0.8, 0.8),
            None => return Err(format!("trade {}: a credit index needs `investment_grade`", t.id)),
        },
        AssetClass::Credit => {
            let r = t
                .rating
                .as_deref()
                .ok_or(format!("trade {}: a single-name credit trade needs a `rating`", t.id))?;
            let f = match r.to_ascii_uppercase().as_str() {
                "AAA" | "AA" => 0.0038,
                "A" => 0.0042,
                "BBB" => 0.0054,
                "BB" => 0.0106,
                "B" => 0.016,
                "CCC" => 0.06,
                _ => return Err(format!("trade {}: unknown rating `{r}` (AAA to CCC)", t.id)),
            };
            s(f, 0.5, 1.0)
        }
        AssetClass::Equity if t.index => s(0.2, 0.8, 0.75),
        AssetClass::Equity => s(0.32, 0.5, 1.2),
        AssetClass::Commodity => match sector(t)? {
            "electricity" => s(0.4, 0.4, 1.5),
            _ => s(0.18, 0.4, 0.7),
        },
    };
    let k = if t.basis {
        0.5
    } else if t.volatility_trade {
        5.0
    } else {
        1.0
    };
    Ok(Supervisory {
        factor: base.factor * k,
        ..base
    })
}

fn sector(t: &Trade) -> Result<&str, String> {
    match t.sector.as_deref() {
        Some(s @ ("electricity" | "oil_gas" | "metals" | "agricultural" | "other")) => Ok(s),
        Some(s) => Err(format!(
            "trade {}: unknown commodity sector `{s}` (electricity, oil_gas, metals, agricultural, other)",
            t.id
        )),
        None => Err(format!("trade {}: a commodity trade needs a `sector`", t.id)),
    }
}

/// Supervisory duration of an interest rate or credit trade: (e^(−0.05 S) − e^(−0.05 E)) / 0.05.
pub fn supervisory_duration(start: f64, end: f64) -> f64 {
    ((-0.05 * start).exp() - (-0.05 * end).exp()) / 0.05
}

/// Supervisory delta of an option bought or sold.
pub fn option_delta(o: &OptionData, volatility: f64) -> Result<f64, String> {
    let (p, k) = (o.price + o.shift, o.strike + o.shift);
    if !(p > 0.0 && k > 0.0 && o.exercise > 0.0 && volatility > 0.0) {
        return Err("an option needs a positive (shifted) price and strike, exercise date and volatility".into());
    }
    let sd = volatility * o.exercise.sqrt();
    let d1 = ((p / k).ln() + 0.5 * sd * sd) / sd;
    let sign = if o.bought { 1.0 } else { -1.0 };
    Ok(match o.kind {
        OptionKind::Call => sign * norm_cdf(d1),
        OptionKind::Put => -sign * norm_cdf(-d1),
    })
}

/// A trade's part in the add-on.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TradeDetail {
    pub id: String,
    pub asset_class: AssetClass,
    pub hedging_set: String,
    pub qualifier: String,
    /// Interest rate maturity bucket: 1 (E < 1 year), 2 (1 to 5 years), 3 (beyond 5 years).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bucket: Option<u8>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub supervisory_duration: Option<f64>,
    pub adjusted_notional: f64,
    pub delta: f64,
    pub maturity_factor: f64,
    pub supervisory_factor: f64,
    pub effective_notional: f64,
    pub market_value: f64,
}

/// Add-on of one hedging set, with its effective notional (interest rate, FX) or its
/// components: reference entities (credit, equity) or commodity types.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct HedgingSet {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub effective_notional: Option<f64>,
    /// Interest rate: net effective notional of the three maturity buckets.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub buckets: Option<[f64; 3]>,
    /// Credit, equity, commodity: signed add-on of each entity or commodity type.
    #[serde(skip_serializing_if = "BTreeMap::is_empty", default)]
    pub components: BTreeMap<String, f64>,
    pub add_on: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ClassAddOn {
    pub asset_class: AssetClass,
    pub add_on: f64,
    pub hedging_sets: Vec<HedgingSet>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Exposure {
    pub netting_set: String,
    pub market_value: f64,
    pub collateral: f64,
    pub replacement_cost: f64,
    pub multiplier: f64,
    pub add_on: f64,
    pub pfe: f64,
    pub ead: f64,
    pub margined: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mpor_days: Option<f64>,
    /// Margined netting set: EAD computed as if unmargined, the cap of the margined EAD.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub unmargined_ead: Option<f64>,
    pub capped: bool,
    pub asset_classes: Vec<ClassAddOn>,
    pub trades: Vec<TradeDetail>,
}

fn check(t: &Trade) -> Result<(), String> {
    let finite = [t.notional, t.market_value, t.maturity].iter().all(|x| x.is_finite());
    if !finite || t.notional < 0.0 || t.maturity < 0.0 {
        return Err(format!(
            "trade {}: notional and maturity must be finite and non-negative, the market value finite",
            t.id
        ));
    }
    if t.option.is_none() && t.long.is_none() {
        return Err(format!("trade {}: give `long` (or `option`)", t.id));
    }
    if t.option.is_some() && t.tranche.is_some() {
        return Err(format!("trade {}: an option cannot be a tranche", t.id));
    }
    if t.basis && t.volatility_trade {
        return Err(format!(
            "trade {}: a trade is a basis or a volatility transaction, not both",
            t.id
        ));
    }
    if let Some(tr) = &t.tranche {
        if t.asset_class != AssetClass::Credit
            || !(0.0..1.0).contains(&tr.attachment)
            || tr.detachment <= tr.attachment
            || tr.detachment > 1.0
        {
            return Err(format!(
                "trade {}: a tranche is a credit trade with 0 <= A < D <= 1",
                t.id
            ));
        }
    }
    Ok(())
}

/// Effective notional of each trade, with an unmargined (`mpor = None`) or a margined maturity factor.
fn details(trades: &[Trade], mpor_days: Option<f64>) -> Result<Vec<TradeDetail>, String> {
    trades
        .iter()
        .map(|t| {
            check(t)?;
            let sv = supervisory(t)?;
            let (mut duration, mut bucket) = (None, None);
            let d = match t.asset_class {
                AssetClass::Ir | AssetClass::Credit => {
                    let s = t.start.unwrap_or(0.0).max(0.0);
                    let e = t.end.unwrap_or(t.maturity);
                    if e < s || e.is_nan() {
                        return Err(format!("trade {}: the end date is before the start date", t.id));
                    }
                    let sd = supervisory_duration(s, e);
                    duration = Some(sd);
                    if t.asset_class == AssetClass::Ir {
                        bucket = Some(if e < 1.0 {
                            1
                        } else if e <= 5.0 {
                            2
                        } else {
                            3
                        });
                    }
                    t.notional * sd
                }
                _ => t.notional,
            };
            let delta = if let Some(o) = &t.option {
                option_delta(o, o.volatility.unwrap_or(sv.volatility)).map_err(|e| format!("trade {}: {e}", t.id))?
            } else {
                let sign = if t.long == Some(true) { 1.0 } else { -1.0 };
                match &t.tranche {
                    Some(tr) => sign * 15.0 / ((1.0 + 14.0 * tr.attachment) * (1.0 + 14.0 * tr.detachment)),
                    None => sign,
                }
            };
            let mf = match mpor_days {
                Some(m) => 1.5 * (m / BUSINESS_DAYS).sqrt(),
                None => t.maturity.clamp(MATURITY_FLOOR, 1.0).sqrt(),
            };
            let hedging_set = match t.asset_class {
                AssetClass::Ir | AssetClass::Fx => t.qualifier.clone(),
                AssetClass::Credit | AssetClass::Equity => "all".into(),
                AssetClass::Commodity => match sector(t)? {
                    "electricity" | "oil_gas" => "energy".into(),
                    s => s.into(),
                },
            };
            let hedging_set = if t.basis {
                format!("{hedging_set} (basis)")
            } else if t.volatility_trade {
                format!("{hedging_set} (volatility)")
            } else {
                hedging_set
            };
            Ok(TradeDetail {
                id: t.id.clone(),
                asset_class: t.asset_class,
                hedging_set,
                qualifier: t.qualifier.clone(),
                bucket,
                supervisory_duration: duration,
                adjusted_notional: d,
                delta,
                maturity_factor: mf,
                supervisory_factor: sv.factor,
                effective_notional: delta * d * mf,
                market_value: t.market_value,
            })
        })
        .collect()
}

/// √((Σ ρ_k A_k)² + Σ (1 − ρ_k²) A_k²): systematic and idiosyncratic parts of entity add-ons.
fn single_factor(parts: &[(f64, f64)]) -> f64 {
    let sys: f64 = parts.iter().map(|(a, r)| r * a).sum();
    let idio: f64 = parts.iter().map(|(a, r)| (1.0 - r * r) * a * a).sum();
    (sys * sys + idio).sqrt()
}

fn add_ons(trades: &[Trade], details: &[TradeDetail]) -> Result<(f64, Vec<ClassAddOn>), String> {
    let mut by_class: BTreeMap<AssetClass, BTreeMap<String, Vec<usize>>> = BTreeMap::new();
    for (i, d) in details.iter().enumerate() {
        by_class
            .entry(d.asset_class)
            .or_default()
            .entry(d.hedging_set.clone())
            .or_default()
            .push(i);
    }
    let mut total = 0.0;
    let mut classes = vec![];
    for (class, sets) in by_class {
        let mut hs = vec![];
        for (name, idx) in sets {
            let sf_d = |i: &usize| details[*i].supervisory_factor * details[*i].effective_notional;
            let set = match class {
                AssetClass::Ir => {
                    let mut b = [0.0; 3];
                    for i in &idx {
                        b[details[*i].bucket.unwrap() as usize - 1] += details[*i].effective_notional;
                    }
                    let en = (b[0] * b[0]
                        + b[1] * b[1]
                        + b[2] * b[2]
                        + 1.4 * b[0] * b[1]
                        + 1.4 * b[1] * b[2]
                        + 0.6 * b[0] * b[2])
                        .max(0.0)
                        .sqrt();
                    // One supervisory factor per hedging set (0.5%, or 0.25% for basis, 2.5% for volatility).
                    let sf = details[idx[0]].supervisory_factor;
                    HedgingSet {
                        name,
                        effective_notional: Some(en),
                        buckets: Some(b),
                        components: BTreeMap::new(),
                        add_on: sf * en,
                    }
                }
                AssetClass::Fx => {
                    let en: f64 = idx.iter().map(|i| details[*i].effective_notional).sum();
                    let a: f64 = idx.iter().map(sf_d).sum();
                    HedgingSet {
                        name,
                        effective_notional: Some(en),
                        buckets: None,
                        components: BTreeMap::new(),
                        add_on: a.abs(),
                    }
                }
                AssetClass::Credit | AssetClass::Equity | AssetClass::Commodity => {
                    // Entities (or commodity types) with their correlation.
                    let mut comp: BTreeMap<String, (f64, f64)> = BTreeMap::new();
                    for i in &idx {
                        let rho = supervisory(&trades[*i])?.correlation;
                        let e = comp.entry(details[*i].qualifier.clone()).or_insert((0.0, rho));
                        if e.1 != rho {
                            return Err(format!(
                                "{}: trades on `{}` mix single names and indices",
                                trades[*i].id, details[*i].qualifier
                            ));
                        }
                        e.0 += sf_d(i);
                    }
                    let parts: Vec<(f64, f64)> = comp.values().cloned().collect();
                    HedgingSet {
                        name,
                        effective_notional: None,
                        buckets: None,
                        components: comp.into_iter().map(|(k, (a, _))| (k, a)).collect(),
                        add_on: single_factor(&parts),
                    }
                }
            };
            hs.push(set);
        }
        let a: f64 = hs.iter().map(|h| h.add_on).sum();
        total += a;
        classes.push(ClassAddOn {
            asset_class: class,
            add_on: a,
            hedging_sets: hs,
        });
    }
    Ok((total, classes))
}

/// min(1, 5% + 95% exp((V − C) / (2 × 95% × AddOn))).
pub fn multiplier(v_minus_c: f64, add_on: f64) -> f64 {
    if add_on <= 0.0 || v_minus_c >= 0.0 {
        return 1.0;
    }
    (FLOOR + (1.0 - FLOOR) * (v_minus_c / (2.0 * (1.0 - FLOOR) * add_on)).exp()).min(1.0)
}

/// SA-CCR exposure at default of a netting set, with α (1.4 in the Basel framework).
pub fn exposure(ns: &NettingSet, alpha: f64) -> Result<Exposure, String> {
    if alpha.is_nan() || alpha <= 0.0 || !ns.collateral.is_finite() {
        return Err("alpha must be positive and the collateral finite".into());
    }
    let mut seen = std::collections::HashSet::new();
    if let Some(t) = ns.trades.iter().find(|t| !seen.insert(&t.id)) {
        return Err(format!("netting set {}: duplicate trade id `{}`", ns.id, t.id));
    }
    let v: f64 = ns.trades.iter().map(|t| t.market_value).sum();
    let vc = v - ns.collateral;
    let unmargined = |d: Vec<TradeDetail>| -> Result<_, String> {
        let (a, c) = add_ons(&ns.trades, &d)?;
        Ok((a, c, d))
    };
    let rc_unmargined = vc.max(0.0);
    let (add_u, classes_u, det_u) = unmargined(details(&ns.trades, None)?)?;
    let ead_u = alpha * (rc_unmargined + multiplier(vc, add_u) * add_u);

    let Some(m) = &ns.margin else {
        return Ok(Exposure {
            netting_set: ns.id.clone(),
            market_value: v,
            collateral: ns.collateral,
            replacement_cost: rc_unmargined,
            multiplier: multiplier(vc, add_u),
            add_on: add_u,
            pfe: multiplier(vc, add_u) * add_u,
            ead: ead_u,
            margined: false,
            mpor_days: None,
            unmargined_ead: None,
            capped: false,
            asset_classes: classes_u,
            trades: det_u,
        });
    };
    let mpor = m.mpor_days.unwrap_or(m.mpor_floor_days + m.remargin_days - 1.0);
    if mpor.is_nan() || mpor <= 0.0 || ![m.threshold, m.mta, m.nica].iter().all(|x| x.is_finite()) {
        return Err(format!("netting set {}: invalid margin terms", ns.id));
    }
    let (add, classes, det) = unmargined(details(&ns.trades, Some(mpor))?)?;
    let rc = vc.max(m.threshold + m.mta - m.nica).max(0.0);
    let mult = multiplier(vc, add);
    let ead = alpha * (rc + mult * add);
    Ok(Exposure {
        netting_set: ns.id.clone(),
        market_value: v,
        collateral: ns.collateral,
        replacement_cost: rc,
        multiplier: mult,
        add_on: add,
        pfe: mult * add,
        ead: ead.min(ead_u),
        margined: true,
        mpor_days: Some(mpor),
        unmargined_ead: Some(ead_u),
        capped: ead_u < ead,
        asset_classes: classes,
        trades: det,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn trade(id: &str, class: AssetClass, q: &str, notional: f64, long: bool, mv: f64, m: f64) -> Trade {
        Trade {
            id: id.into(),
            asset_class: class,
            qualifier: q.into(),
            rating: None,
            index: false,
            investment_grade: None,
            sector: None,
            notional,
            long: Some(long),
            market_value: mv,
            maturity: m,
            start: None,
            end: None,
            option: None,
            tranche: None,
            basis: false,
            volatility_trade: false,
        }
    }

    fn ns(id: &str, trades: Vec<Trade>) -> NettingSet {
        NettingSet {
            id: id.into(),
            trades,
            collateral: 0.0,
            margin: None,
        }
    }

    /// Example 1 of BCBS 279, Annex 4: two USD swaps and a EUR swaption bought.
    fn rates() -> Vec<Trade> {
        let mut swaption = trade("3", AssetClass::Ir, "EUR", 5000.0, true, 50.0, 11.0);
        swaption.start = Some(1.0);
        swaption.long = None;
        swaption.option = Some(OptionData {
            kind: OptionKind::Put,
            bought: true,
            price: 0.06,
            strike: 0.05,
            exercise: 1.0,
            volatility: None,
            shift: 0.0,
        });
        vec![
            trade("1", AssetClass::Ir, "USD", 10_000.0, true, 30.0, 10.0),
            trade("2", AssetClass::Ir, "USD", 10_000.0, false, -20.0, 4.0),
            swaption,
        ]
    }

    /// Example 2: two single-name CDS and an index CDS.
    fn credit() -> Vec<Trade> {
        let mut a = trade("4", AssetClass::Credit, "Firm A", 10_000.0, true, 20.0, 3.0);
        a.rating = Some("AA".into());
        let mut b = trade("5", AssetClass::Credit, "Firm B", 10_000.0, false, -40.0, 6.0);
        b.rating = Some("BBB".into());
        let mut c = trade("6", AssetClass::Credit, "CDX.IG", 10_000.0, true, 0.0, 5.0);
        c.index = true;
        c.investment_grade = Some(true);
        vec![a, b, c]
    }

    /// Example 3: oil forwards (one commodity type) and a silver forward. The example's RC counts
    /// the silver forward at +100, its table at −100: +100 gives the published EAD.
    fn commodities() -> Vec<Trade> {
        let mut v = vec![
            trade("7", AssetClass::Commodity, "crude oil", 10_000.0, true, -50.0, 0.75),
            trade("8", AssetClass::Commodity, "crude oil", 20_000.0, false, -30.0, 2.0),
            trade("9", AssetClass::Commodity, "silver", 10_000.0, true, 100.0, 5.0),
        ];
        for (t, s) in v.iter_mut().zip(["oil_gas", "oil_gas", "metals"]) {
            t.sector = Some(s.into());
        }
        v
    }

    #[test]
    fn basel_annex_4_examples() {
        // Example 1: SD 7.87 / 3.63 / 7.49, swaption delta −0.2694, add-on 347, EAD 569.
        let e1 = exposure(&ns("1", rates()), 1.4).unwrap();
        let t = &e1.trades;
        assert!((t[0].supervisory_duration.unwrap() - 7.87).abs() < 5e-3);
        assert!((t[2].supervisory_duration.unwrap() - 7.49).abs() < 5e-3);
        assert!((t[2].delta + 0.2694).abs() < 5e-5);
        let ir = &e1.asset_classes[0].hedging_sets;
        assert_eq!((ir[0].name.as_str(), ir[1].name.as_str()), ("EUR", "USD"));
        assert!((ir[1].effective_notional.unwrap() - 59_270.0).abs() < 1.0);
        assert!((ir[0].effective_notional.unwrap() - 10_083.0).abs() < 1.0);
        assert!((e1.add_on - 347.0).abs() < 0.5 && e1.replacement_cost == 60.0 && e1.multiplier == 1.0);
        assert!((e1.ead - 569.0).abs() < 0.5);

        // Example 2: entity add-ons 106 / −280 / 168, add-on 282, multiplier 0.965, EAD 381.
        let e2 = exposure(&ns("2", credit()), 1.4).unwrap();
        let c = &e2.asset_classes[0].hedging_sets[0].components;
        assert!(
            (c["Firm A"] - 106.0).abs() < 0.5 && (c["Firm B"] + 280.0).abs() < 0.5 && (c["CDX.IG"] - 168.0).abs() < 0.5
        );
        assert!((e2.add_on - 282.0).abs() < 0.5 && (e2.multiplier - 0.965).abs() < 5e-4);
        assert!((e2.ead - 381.0).abs() < 0.5);

        // Example 3: MF √(9/12), add-ons 2,041 (energy) and 1,800 (metals), EAD 5,406 with the
        // example's rounded add-on (1.4 × (20 + 3,841) = 5,405.4).
        let e3 = exposure(&ns("3", commodities()), 1.4).unwrap();
        assert!((e3.trades[0].maturity_factor - 0.75f64.sqrt()).abs() < 1e-15);
        assert!((e3.add_on - 3841.0).abs() < 0.5 && e3.replacement_cost == 20.0);
        assert!((e3.ead - 5406.0).abs() < 1.0);

        // Example 4: rates and credit in one netting set: RC 40, add-on 629, EAD 936.
        let e4 = exposure(&ns("4", [rates(), credit()].concat()), 1.4).unwrap();
        assert!(e4.replacement_cost == 40.0 && (e4.add_on - 629.0).abs() < 0.5 && e4.multiplier == 1.0);
        assert!((e4.ead - 936.0).abs() < 1.0);

        // Example 5: rates and commodities, margined weekly: MPOR 14 days, RC 0, add-on 1,401,
        // multiplier 0.958, EAD 1,879.
        let mut five = ns("5", [rates(), commodities()].concat());
        five.collateral = 200.0;
        five.margin = Some(Margin {
            threshold: 0.0,
            mta: 5.0,
            nica: 150.0,
            mpor_days: None,
            remargin_days: 5.0,
            mpor_floor_days: 10.0,
        });
        let e5 = exposure(&five, 1.4).unwrap();
        assert_eq!(e5.mpor_days, Some(14.0));
        assert!((e5.trades[0].maturity_factor - 0.3550).abs() < 5e-5);
        assert!(e5.market_value == 80.0 && e5.replacement_cost == 0.0);
        assert!((e5.add_on - 1401.0).abs() < 1.0 && (e5.multiplier - 0.958).abs() < 5e-4);
        assert!((e5.ead - 1879.0).abs() < 1.0 && !e5.capped);
    }

    #[test]
    fn margin_cap_floors_and_errors() {
        // A short-dated margined set: 1.5 √(10 / 250) = 0.3 is above √(10 / 250) = 0.2, the
        // unmargined maturity factor of a trade at the floor, so the unmargined EAD caps it.
        let mut s = ns("cap", vec![trade("fx", AssetClass::Fx, "EURUSD", 1e6, true, 0.0, 0.01)]);
        assert!((exposure(&s, 1.4).unwrap().trades[0].maturity_factor - 0.2).abs() < 1e-15);
        s.margin = Some(Margin {
            threshold: 0.0,
            mta: 0.0,
            nica: 0.0,
            mpor_days: Some(10.0),
            remargin_days: 1.0,
            mpor_floor_days: 10.0,
        });
        let e = exposure(&s, 1.4).unwrap();
        assert!(e.capped && e.ead == e.unmargined_ead.unwrap());
        assert!((e.ead - 1.4 * 0.04 * 1e6 * 0.2).abs() < 1e-6);

        // Excess collateral lowers the multiplier, never below 5%.
        assert_eq!(multiplier(10.0, 100.0), 1.0);
        assert!((multiplier(-1e9, 1.0) - 0.05).abs() < 1e-15);

        // A bought and a sold call: opposite deltas; a CDO tranche 3-7%.
        let o = OptionData {
            kind: OptionKind::Call,
            bought: true,
            price: 100.0,
            strike: 100.0,
            exercise: 1.0,
            volatility: Some(0.2),
            shift: 0.0,
        };
        let sold = OptionData {
            bought: false,
            ..o.clone()
        };
        assert!((option_delta(&o, 0.2).unwrap() + option_delta(&sold, 0.2).unwrap()).abs() < 1e-15);
        assert!((option_delta(&o, 0.2).unwrap() - norm_cdf(0.1)).abs() < 1e-15);
        let mut tr = trade("t", AssetClass::Credit, "CDX tranche", 1e6, true, 0.0, 5.0);
        tr.index = true;
        tr.investment_grade = Some(true);
        tr.tranche = Some(Tranche {
            attachment: 0.03,
            detachment: 0.07,
        });
        let d = exposure(&ns("t", vec![tr.clone()]), 1.4).unwrap().trades[0].delta;
        assert!((d - 15.0 / (1.42 * 1.98)).abs() < 1e-12);

        // Basis transactions: own hedging set at half the factor.
        let mut b = trade("b", AssetClass::Ir, "USD", 1e6, true, 0.0, 3.0);
        b.basis = true;
        let e = exposure(
            &ns("b", vec![b, trade("s", AssetClass::Ir, "USD", 1e6, false, 0.0, 3.0)]),
            1.4,
        )
        .unwrap();
        let hs = &e.asset_classes[0].hedging_sets;
        assert_eq!(
            hs.iter().map(|h| h.name.as_str()).collect::<Vec<_>>(),
            ["USD", "USD (basis)"]
        );
        assert!((hs[1].add_on - 0.5 * hs[0].add_on).abs() < 1e-9);

        let mut bad = ns("x", vec![trade("a", AssetClass::Credit, "Firm", 1.0, true, 0.0, 1.0)]);
        assert!(exposure(&bad, 1.4).unwrap_err().contains("rating"));
        bad.trades[0].rating = Some("BBB".into());
        bad.trades.push(bad.trades[0].clone());
        assert!(exposure(&bad, 1.4).unwrap_err().contains("duplicate"));
        let mut nolong = trade("n", AssetClass::Fx, "EURUSD", 1.0, true, 0.0, 1.0);
        nolong.long = None;
        assert!(exposure(&ns("n", vec![nolong]), 1.4).unwrap_err().contains("long"));
        let mut mixed = tr.clone();
        mixed.id = "m".into();
        mixed.tranche = None;
        mixed.index = false;
        mixed.rating = Some("A".into());
        assert!(exposure(&ns("m", vec![tr, mixed]), 1.4).unwrap_err().contains("mix"));
    }
}
