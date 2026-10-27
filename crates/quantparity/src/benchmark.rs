//! Compares a candidate pricer against a reference, with per-product tolerances.

use crate::curves::Market;
use crate::Status;
use crate::{value_in, Trade};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct Tolerance {
    #[serde(default)]
    pub abs: f64,
    #[serde(default)]
    pub rel: f64,
}

/// Tolerances: `{"default": {...}, "<product>": {...}}`. Allowed gap = abs + rel·|reference|.
#[derive(Debug, Clone)]
pub struct Tolerances {
    pub default: Tolerance,
    pub by_product: BTreeMap<String, Tolerance>,
}

impl Tolerances {
    pub fn from_value(v: &Value) -> Result<Self, String> {
        let mut by_product: BTreeMap<String, Tolerance> = match v {
            Value::Null => BTreeMap::new(),
            v => serde_json::from_value(v.clone()).map_err(|e| format!("invalid tolerances: {e}"))?,
        };
        let default = by_product
            .remove("default")
            .unwrap_or(Tolerance { abs: 1e-6, rel: 1e-6 });
        Ok(Self { default, by_product })
    }

    fn get(&self, product: &str) -> Tolerance {
        self.by_product.get(product).copied().unwrap_or(self.default)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TradeComparison {
    pub id: String,
    pub product: String,
    pub reference: f64,
    pub candidate: Option<f64>,
    pub abs_diff: Option<f64>,
    pub rel_diff: Option<f64>,
    pub tolerance: f64,
    pub status: Status,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ProductSummary {
    pub trades: usize,
    pub max_abs_diff: f64,
    pub status: Status,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Benchmark {
    pub compared: usize,
    pub passed: usize,
    pub warned: usize,
    pub failed: usize,
    /// Trades with no candidate price.
    pub missing: Vec<String>,
    /// Candidate prices with no matching trade.
    pub unexpected: Vec<String>,
    pub max_abs_diff: f64,
    pub by_product: BTreeMap<String, ProductSummary>,
    pub trades: Vec<TradeComparison>,
    pub status: Status,
}

/// Parses prices: `[{"id", "price"}]`, `{"valuations": [...]}` or `{"<id>": price}`.
pub fn prices_from_value(v: &Value) -> Result<BTreeMap<String, f64>, String> {
    let list = v.as_array().or_else(|| v.get("valuations").and_then(Value::as_array));
    if let Some(list) = list {
        return list
            .iter()
            .map(|x| {
                let id = x.get("id").and_then(Value::as_str).ok_or("element without `id`")?;
                let p = x
                    .get("price")
                    .and_then(Value::as_f64)
                    .ok_or_else(|| format!("{id}: missing `price`"))?;
                Ok((id.to_string(), p))
            })
            .collect();
    }
    v.as_object()
        .ok_or("unrecognized price format")?
        .iter()
        .map(|(k, p)| {
            p.as_f64()
                .map(|p| (k.clone(), p))
                .ok_or_else(|| format!("{k}: non-numeric price"))
        })
        .collect()
}

/// Parses trades: `[...]` or `{"trades": [...]}`.
pub fn trades_from_value(v: &Value) -> Result<Vec<Trade>, String> {
    let list = v.get("trades").unwrap_or(v);
    serde_json::from_value(list.clone()).map_err(|e| format!("invalid trades: {e}"))
}

/// Reference prices computed by the built-in library.
pub fn reference_prices(trades: &[Trade], market: &Market) -> Result<BTreeMap<String, f64>, String> {
    trades
        .iter()
        .map(|t| {
            value_in(t, market)
                .map(|v| (t.id.clone(), v.price))
                .map_err(|e| format!("{}: {e}", t.id))
        })
        .collect()
}

pub fn compare(
    trades: &[Trade],
    reference: &BTreeMap<String, f64>,
    candidate: &BTreeMap<String, f64>,
    tol: &Tolerances,
) -> Result<Benchmark, String> {
    let mut rows = vec![];
    let mut by_product: BTreeMap<String, ProductSummary> = BTreeMap::new();
    for t in trades {
        let product = t.instrument.product().to_string();
        let r = *reference
            .get(&t.id)
            .ok_or_else(|| format!("{}: missing reference price", t.id))?;
        let c = candidate.get(&t.id).copied();
        let tl = tol.get(&product);
        let allowed = tl.abs + tl.rel * r.abs();
        let abs_diff = c.map(|c| (c - r).abs());
        let status = match abs_diff {
            Some(d) if d.is_finite() && d <= allowed => Status::Pass,
            Some(d) if d.is_finite() && d <= 10.0 * allowed => Status::Warn,
            _ => Status::Fail,
        };
        let s = by_product.entry(product.clone()).or_insert(ProductSummary {
            trades: 0,
            max_abs_diff: 0.0,
            status: Status::Pass,
        });
        s.trades += 1;
        s.max_abs_diff = s.max_abs_diff.max(abs_diff.filter(|d| d.is_finite()).unwrap_or(0.0));
        s.status = s.status.max(status);
        rows.push(TradeComparison {
            id: t.id.clone(),
            product,
            reference: r,
            candidate: c,
            abs_diff,
            rel_diff: abs_diff.map(|d| if r != 0.0 { d / r.abs() } else { d }),
            tolerance: allowed,
            status,
        });
    }
    let count = |s: Status| rows.iter().filter(|r| r.status == s).count();
    let missing: Vec<String> = rows
        .iter()
        .filter(|r| r.candidate.is_none())
        .map(|r| r.id.clone())
        .collect();
    let unexpected: Vec<String> = candidate
        .keys()
        .filter(|k| !trades.iter().any(|t| &&t.id == k))
        .cloned()
        .collect();
    let status = rows.iter().map(|r| r.status).fold(
        if unexpected.is_empty() {
            Status::Pass
        } else {
            Status::Warn
        },
        Status::max,
    );
    Ok(Benchmark {
        compared: rows.len(),
        passed: count(Status::Pass),
        warned: count(Status::Warn),
        failed: count(Status::Fail),
        missing,
        unexpected,
        max_abs_diff: rows
            .iter()
            .filter_map(|r| r.abs_diff)
            .filter(|d| d.is_finite())
            .fold(0.0, f64::max),
        by_product,
        trades: rows,
        status,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn trades() -> Vec<Trade> {
        trades_from_value(&json!([
            {"id": "A", "product": "european_option", "option_type": "call", "spot": 42, "strike": 40, "rate": 0.1, "vol": 0.2, "maturity": 0.5},
            {"id": "B", "product": "fixed_rate_bond", "coupon_rate": 0.05, "maturity": 3, "frequency": 1, "yield_rate": 0.05}
        ]))
        .unwrap()
    }

    #[test]
    fn statuses_follow_tolerances() {
        let t = trades();
        let r = reference_prices(&t, &Market::default()).unwrap();
        let tol = Tolerances::from_value(&json!({"default": {"abs": 1e-4}})).unwrap();
        let exact = compare(&t, &r, &r, &tol).unwrap();
        assert_eq!((exact.passed, exact.status), (2, Status::Pass));

        let mut c = r.clone();
        *c.get_mut("A").unwrap() += 5e-4; // 5x the tolerance -> alert
        c.remove("B");
        c.insert("Z".into(), 1.0);
        let b = compare(&t, &r, &c, &tol).unwrap();
        assert_eq!(b.trades[0].status, Status::Warn);
        assert_eq!(b.missing, vec!["B".to_string()]);
        assert_eq!(b.unexpected, vec!["Z".to_string()]);
        assert_eq!(b.status, Status::Fail);
    }

    #[test]
    fn reads_price_formats() {
        let a = prices_from_value(&json!([{"id": "A", "price": 1.5}])).unwrap();
        let b = prices_from_value(&json!({"valuations": [{"id": "A", "price": 1.5}]})).unwrap();
        let c = prices_from_value(&json!({"A": 1.5})).unwrap();
        assert!(a == b && b == c);
    }
}
