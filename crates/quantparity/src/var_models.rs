//! One-day VaR and expected shortfall forecasts from a return series: RiskMetrics EWMA volatility
//! with normal or Student-t innovations, and filtered historical simulation (Barone-Adesi, Hull
//! and White). Every forecast for day t uses the returns up to t-1 only.

use crate::instruments::norm_pdf;
use crate::stats::{norm_inv, t_inv, t_pdf};
use serde::{Deserialize, Serialize};

/// Distribution of the standardized returns r_t / σ_t.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Innovations {
    Normal,
    /// Student-t with `nu` degrees of freedom (`nu` > 2), scaled to unit variance.
    StudentT {
        nu: f64,
    },
    /// Empirical distribution of the past standardized returns: all of them, or the latest
    /// `window`; forecasts start once `min_residuals` are available.
    Filtered {
        window: Option<usize>,
        min_residuals: usize,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VarForecasts {
    /// `ewma-normal`, `ewma-t` or `fhs`.
    pub model: String,
    pub lambda: f64,
    pub warmup: usize,
    pub innovations: Innovations,
    pub var_confidence: f64,
    pub es_confidence: f64,
    /// `var[j]` (and every other vector) is the forecast for `returns[start + j]`.
    pub start: usize,
    /// EWMA volatility forecast.
    pub sigma: Vec<f64>,
    /// VaR at `var_confidence`, as a positive loss.
    pub var: Vec<f64>,
    /// Expected shortfall at `es_confidence`, as a positive loss.
    pub es: Vec<f64>,
    /// VaR at `es_confidence`: the threshold beyond which the ES is the expected loss.
    pub es_var: Vec<f64>,
    /// Multipliers of σ for the parametric models (VaR, ES, VaR at the ES level).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub multipliers: Option<[f64; 3]>,
}

/// Quantile and ES multipliers of a unit-variance distribution: (VaR at `a`, ES at `e`, VaR at `e`).
fn multipliers(innov: Innovations, a: f64, e: f64) -> Result<[f64; 3], String> {
    match innov {
        Innovations::Normal => {
            let ze = norm_inv(e);
            Ok([norm_inv(a), norm_pdf(ze) / (1.0 - e), ze])
        }
        Innovations::StudentT { nu } => {
            if !(nu > 2.0 && nu.is_finite()) {
                return Err(format!(
                    "Student-t: nu = {nu}, more than 2 degrees of freedom are required"
                ));
            }
            let c = ((nu - 2.0) / nu).sqrt();
            let qe = t_inv(e, nu);
            let es = c * t_pdf(qe, nu) * (nu + qe * qe) / ((nu - 1.0) * (1.0 - e));
            Ok([c * t_inv(a, nu), es, c * qe])
        }
        Innovations::Filtered { .. } => unreachable!(),
    }
}

/// k-th worst (k = ⌈n(1-α)⌉) and mean of the k worst values of a sorted sample, as in
/// [`crate::risk::var_es`].
fn tail(sorted: &[f64], conf: f64) -> (f64, f64) {
    let n = sorted.len();
    let k = (((1.0 - conf) * n as f64 - 1e-9).ceil().max(1.0) as usize).min(n);
    (-sorted[k - 1], -sorted[..k].iter().sum::<f64>() / k as f64)
}

/// EWMA forecasts: σ²_t = λ σ²_{t-1} + (1-λ) r²_{t-1}, started from the mean square of the first
/// `warmup` returns; VaR_t = σ_t q_α and ES_t = σ_t ES_e of the innovations.
pub fn var_forecasts(
    returns: &[f64],
    lambda: f64,
    warmup: usize,
    var_confidence: f64,
    es_confidence: f64,
    innovations: Innovations,
) -> Result<VarForecasts, String> {
    if !(0.0..1.0).contains(&lambda) {
        return Err(format!("lambda = {lambda}: between 0 and 1"));
    }
    for c in [var_confidence, es_confidence] {
        if !(c > 0.5 && c < 1.0) {
            return Err(format!("confidence level {c}: between 0.5 and 1"));
        }
    }
    if warmup == 0 || returns.len() <= warmup {
        return Err(format!("{} returns for a warm-up of {warmup}", returns.len()));
    }
    if returns.iter().any(|x| !x.is_finite()) {
        return Err("non-finite return".into());
    }
    let mut s2 = returns[..warmup].iter().map(|x| x * x).sum::<f64>() / warmup as f64;
    let mut sigma = Vec::with_capacity(returns.len() - warmup);
    for r in &returns[warmup..] {
        sigma.push(s2.sqrt());
        s2 = lambda * s2 + (1.0 - lambda) * r * r;
    }
    if sigma.iter().any(|s| *s <= 0.0) {
        return Err("zero volatility: the warm-up returns are all zero".into());
    }
    let (model, start, var, es, es_var, mult) = match innovations {
        Innovations::Filtered { window, min_residuals } => {
            if min_residuals == 0 || window.is_some_and(|w| w < min_residuals) {
                return Err("filtered simulation: min_residuals ≥ 1 and window ≥ min_residuals".into());
            }
            let start = warmup + min_residuals;
            if returns.len() <= start {
                return Err(format!(
                    "{} returns for {start} needed before the first forecast",
                    returns.len()
                ));
            }
            // Sorted standardized residuals z_s = r_s / σ_s of the days before t.
            let z = |s: usize| returns[s] / sigma[s - warmup];
            let mut sorted: Vec<f64> = vec![];
            let (mut var, mut es, mut es_var) = (vec![], vec![], vec![]);
            for t in warmup..returns.len() {
                if t >= start {
                    let sd = sigma[t - warmup];
                    let (qv, _) = tail(&sorted, var_confidence);
                    let (qe, e) = tail(&sorted, es_confidence);
                    var.push(sd * qv);
                    es.push(sd * e);
                    es_var.push(sd * qe);
                }
                let x = z(t);
                let i = sorted.partition_point(|y| *y < x);
                sorted.insert(i, x);
                if let Some(w) = window {
                    if t + 1 >= warmup + w + 1 {
                        let old = z(t - w);
                        let j = sorted.partition_point(|y| *y < old);
                        sorted.remove(j);
                    }
                }
            }
            sigma.drain(..min_residuals);
            ("fhs", start, var, es, es_var, None)
        }
        innov => {
            let m = multipliers(innov, var_confidence, es_confidence)?;
            let scale = |k: usize| sigma.iter().map(|s| s * m[k]).collect::<Vec<f64>>();
            let name = if matches!(innov, Innovations::Normal) {
                "ewma-normal"
            } else {
                "ewma-t"
            };
            (name, warmup, scale(0), scale(1), scale(2), Some(m))
        }
    };
    Ok(VarForecasts {
        model: model.into(),
        lambda,
        warmup,
        innovations,
        var_confidence,
        es_confidence,
        start,
        sigma,
        var,
        es,
        es_var,
        multipliers: mult,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn returns(n: usize) -> Vec<f64> {
        // Deterministic heavy-tailed sample: a volatility regime change and a few large losses.
        (0..n)
            .map(|i| {
                let u = ((i * 7919 + 13) % 1000) as f64 / 1000.0 - 0.5;
                let vol = if i < n / 2 { 0.01 } else { 0.02 };
                let shock = if i % 97 == 0 { -4.0 } else { 1.0 };
                vol * u * 3.4 * shock
            })
            .collect()
    }

    #[test]
    fn normal_reproduces_riskmetrics() {
        let r = returns(600);
        let f = var_forecasts(&r, 0.94, 30, 0.99, 0.975, Innovations::Normal).unwrap();
        // Independent recursion, as in the RiskMetrics technical document.
        let mut s2: f64 = r[..30].iter().map(|x| x * x).sum::<f64>() / 30.0;
        for (j, t) in (30..600).enumerate() {
            assert!((f.var[j] - 2.326_347_874_040_840_8 * s2.sqrt()).abs() < 1e-15);
            s2 = 0.94 * s2 + 0.06 * r[t] * r[t];
        }
        let m = f.multipliers.unwrap();
        // ES 97.5% of a normal: φ(1.96)/0.025 (scipy: 2.3378027922014133).
        assert!((m[1] - 2.337_802_792_201_413).abs() < 1e-13);
        assert_eq!(f.start, 30);
        assert_eq!(f.var.len(), 570);
    }

    #[test]
    fn student_t_multipliers() {
        let f = var_forecasts(&returns(100), 0.94, 30, 0.99, 0.975, Innovations::StudentT { nu: 5.0 }).unwrap();
        let m = f.multipliers.unwrap();
        // t5 at 99%, unit variance: 3.364930 * sqrt(3/5).
        assert!((m[0] - 3.364_929_998_907_218 * 0.6f64.sqrt()).abs() < 1e-13);
        // ES and VaR at 97.5% of a unit-variance t5 (scipy, closed form and numerical integration).
        assert!((m[1] - 2.727_802_071_641_67).abs() < 1e-12);
        assert!((m[2] - 1.991_164_127_896_548).abs() < 1e-13);
        assert!(var_forecasts(&returns(100), 0.94, 30, 0.99, 0.975, Innovations::StudentT { nu: 2.0 }).is_err());
    }

    #[test]
    fn filtered_simulation_uses_past_residuals_only() {
        let r = returns(800);
        let fhs = Innovations::Filtered {
            window: None,
            min_residuals: 250,
        };
        let f = var_forecasts(&r, 0.94, 30, 0.99, 0.975, fhs).unwrap();
        assert_eq!((f.start, f.var.len(), f.sigma.len()), (280, 520, 520));
        // Brute force: forecast for day t from the sorted residuals of days 30..t.
        let ewma = var_forecasts(&r, 0.94, 30, 0.99, 0.975, Innovations::Normal).unwrap();
        for t in [280, 431, 799] {
            let mut z: Vec<f64> = (30..t).map(|s| r[s] / ewma.sigma[s - 30]).collect();
            z.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let v = crate::risk::var_es(&z, 0.99, 0.975).unwrap();
            let j = t - 280;
            assert!((f.var[j] - ewma.sigma[t - 30] * v.var).abs() < 1e-15);
            assert!((f.es[j] - ewma.sigma[t - 30] * v.es).abs() < 1e-15);
        }
        // Rolling window: the same, on the latest 300 residuals.
        let w = Innovations::Filtered {
            window: Some(300),
            min_residuals: 250,
        };
        let g = var_forecasts(&r, 0.94, 30, 0.99, 0.975, w).unwrap();
        let t = 700;
        let mut z: Vec<f64> = (t - 300..t).map(|s| r[s] / ewma.sigma[s - 30]).collect();
        z.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let (q, _) = tail(&z, 0.99);
        assert!((g.var[t - 280] - ewma.sigma[t - 30] * q).abs() < 1e-15);
    }
}
