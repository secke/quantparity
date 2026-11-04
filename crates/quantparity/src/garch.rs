//! GARCH(1,1) and GJR-GARCH(1,1) volatility, fitted by Gaussian quasi-maximum likelihood.
//!
//! With a zero mean: σ²_t = ω + (α + γ 1{r_{t-1} < 0}) r²_{t-1} + β σ²_{t-1} (γ = 0 for GARCH;
//! Glosten, Jagannathan and Runkle 1993 for the asymmetric term). The conventions are those of the
//! Python `arch` package, so that a fit can be reconciled with it:
//!
//! - the pre-sample r², σ² (and r² 1{r < 0} for GJR, halved) are the backcast: the average of the
//!   first min(75, n) squared returns with weights 0.94^i, normalized;
//! - the parameters satisfy ω > 0, α ≥ 0, α + γ ≥ 0, β ≥ 0 and α + γ/2 + β < 1.
//!
//! The likelihood is maximized by Nelder-Mead on returns scaled to unit mean square, from a grid
//! of starting points, and restarted until it no longer improves.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct GarchParams {
    pub omega: f64,
    pub alpha: f64,
    /// Asymmetric (leverage) term of GJR; 0 for GARCH.
    pub gamma: f64,
    pub beta: f64,
}

impl GarchParams {
    /// α + γ/2 + β: the variance is stationary below 1.
    pub fn persistence(&self) -> f64 {
        self.alpha + self.gamma / 2.0 + self.beta
    }

    fn feasible(&self) -> bool {
        self.omega > 0.0
            && self.alpha >= 0.0
            && self.alpha + self.gamma >= 0.0
            && self.beta >= 0.0
            && self.persistence() < 1.0
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GarchFit {
    /// `garch` or `gjr`.
    pub model: String,
    pub params: GarchParams,
    pub persistence: f64,
    /// Unconditional volatility sqrt(ω / (1 - persistence)).
    pub long_run_volatility: f64,
    /// Gaussian log-likelihood at the optimum.
    pub loglik: f64,
    pub observations: usize,
    pub backcast: f64,
}

/// Weighted average of the first min(75, n) squared returns, weights 0.94^i (the `arch` default).
pub fn backcast(returns: &[f64]) -> f64 {
    let tau = returns.len().min(75);
    let (mut s, mut w, mut total) = (0.0, 1.0, 0.0);
    for r in &returns[..tau] {
        s += w * r * r;
        total += w;
        w *= 0.94;
    }
    s / total
}

/// Conditional variances σ²_0 … σ²_{n-1} of `returns`, followed by the forecast σ²_n for the next day.
pub fn variances(returns: &[f64], p: &GarchParams, backcast: f64) -> Vec<f64> {
    let mut v = Vec::with_capacity(returns.len() + 1);
    let mut s2 = p.omega + (p.alpha + p.gamma / 2.0 + p.beta) * backcast;
    v.push(s2);
    for r in returns {
        s2 = next_variance(p, s2, *r);
        v.push(s2);
    }
    v
}

/// σ²_{t+1} from σ²_t and r_t.
pub fn next_variance(p: &GarchParams, s2: f64, r: f64) -> f64 {
    let a = if r < 0.0 { p.alpha + p.gamma } else { p.alpha };
    p.omega + a * r * r + p.beta * s2
}

/// Gaussian log-likelihood of `returns` with zero mean and the GARCH variances.
pub fn loglik(returns: &[f64], p: &GarchParams, backcast: f64) -> f64 {
    let v = variances(returns, p, backcast);
    -0.5 * returns
        .iter()
        .zip(&v)
        .map(|(r, s2)| (2.0 * std::f64::consts::PI).ln() + s2.ln() + r * r / s2)
        .sum::<f64>()
}

/// Quasi-maximum likelihood fit of a GARCH(1,1) (`asymmetric = false`) or GJR-GARCH(1,1).
pub fn garch_fit(returns: &[f64], asymmetric: bool) -> Result<GarchFit, String> {
    let n = returns.len();
    if n < 100 {
        return Err(format!("{n} returns: at least 100 are needed to fit a GARCH"));
    }
    if returns.iter().any(|x| !x.is_finite()) {
        return Err("non-finite return".into());
    }
    let m2 = returns.iter().map(|r| r * r).sum::<f64>() / n as f64;
    if m2 <= 0.0 {
        return Err("the returns are all zero".into());
    }
    // Fit on x = r / s, s² = mean square: ω' = ω / s², log-likelihood shifted by n ln s.
    let s = m2.sqrt();
    let x: Vec<f64> = returns.iter().map(|r| r / s).collect();
    let bc = backcast(&x);
    let unpack = |t: &[f64]| GarchParams {
        omega: t[0],
        alpha: t[1],
        gamma: if asymmetric { t[2] } else { 0.0 },
        beta: t[t.len() - 1],
    };
    let objective = |t: &[f64]| {
        let p = unpack(t);
        if p.feasible() {
            let l = -loglik(&x, &p, bc);
            if l.is_finite() {
                return l;
            }
        }
        f64::INFINITY
    };
    // Starting points: persistence, α and γ on a grid, ω' from the unit unconditional variance.
    let mut best: Option<(Vec<f64>, f64)> = None;
    for pers in [0.5, 0.7, 0.9, 0.98] {
        for alpha in [0.01, 0.05, 0.1, 0.2] {
            for gamma in if asymmetric { vec![0.0, 0.1, 0.2] } else { vec![0.0] } {
                let beta = pers - alpha - gamma / 2.0;
                if beta < 0.0 {
                    continue;
                }
                let mut t = vec![1.0 - pers, alpha];
                if asymmetric {
                    t.push(gamma);
                }
                t.push(beta);
                let f = objective(&t);
                if best.as_ref().map_or(true, |(_, b)| f < *b) {
                    best = Some((t, f));
                }
            }
        }
    }
    let (mut t, mut f) = best.ok_or("no feasible starting point")?;
    for _ in 0..50 {
        let (t1, f1) = nelder_mead(&objective, &t, 1e-15, 20_000);
        let improved = f - f1 > 1e-11;
        (t, f) = (t1, f1);
        if !improved {
            break;
        }
    }
    if !f.is_finite() {
        return Err("the GARCH likelihood could not be maximized".into());
    }
    let p = unpack(&t);
    let params = GarchParams {
        omega: p.omega * m2,
        ..p
    };
    let persistence = params.persistence();
    Ok(GarchFit {
        model: if asymmetric { "gjr" } else { "garch" }.into(),
        params,
        persistence,
        long_run_volatility: (params.omega / (1.0 - persistence)).sqrt(),
        loglik: -f - n as f64 * s.ln(),
        observations: n,
        backcast: bc * m2,
    })
}

/// Nelder-Mead minimization from `x0` (standard coefficients 1, 2, 0.5, 0.5); stops when the
/// values on the simplex differ by less than `ftol` relative, or its vertices by less than 1e-11.
fn nelder_mead(f: &dyn Fn(&[f64]) -> f64, x0: &[f64], ftol: f64, max_iter: usize) -> (Vec<f64>, f64) {
    let k = x0.len();
    let mut simplex: Vec<(Vec<f64>, f64)> = vec![(x0.to_vec(), f(x0))];
    for i in 0..k {
        let mut x = x0.to_vec();
        x[i] += if x[i].abs() > 1e-3 { 0.1 * x[i] } else { 0.01 };
        let fx = f(&x);
        simplex.push((x, fx));
    }
    let point = |c: &[f64], d: &[f64], a: f64| -> Vec<f64> { c.iter().zip(d).map(|(c, d)| c + a * (d - c)).collect() };
    for _ in 0..max_iter {
        simplex.sort_by(|a, b| a.1.total_cmp(&b.1));
        let spread = simplex[1..]
            .iter()
            .flat_map(|(x, _)| x.iter().zip(&simplex[0].0).map(|(a, b)| (a - b).abs()))
            .fold(0.0, f64::max);
        if simplex[k].1 - simplex[0].1 <= ftol * simplex[0].1.abs().max(1.0) || spread < 1e-11 {
            break;
        }
        let centroid: Vec<f64> = (0..k)
            .map(|j| simplex[..k].iter().map(|(x, _)| x[j]).sum::<f64>() / k as f64)
            .collect();
        let worst = simplex[k].clone();
        let xr = point(&centroid, &worst.0, -1.0);
        let fr = f(&xr);
        if fr < simplex[0].1 {
            let xe = point(&centroid, &worst.0, -2.0);
            let fe = f(&xe);
            simplex[k] = if fe < fr { (xe, fe) } else { (xr, fr) };
        } else if fr < simplex[k - 1].1 {
            simplex[k] = (xr, fr);
        } else {
            let (xc, fc) = if fr < worst.1 {
                let xc = point(&centroid, &xr, 0.5);
                let fc = f(&xc);
                (xc, fc)
            } else {
                let xc = point(&centroid, &worst.0, 0.5);
                let fc = f(&xc);
                (xc, fc)
            };
            if fc < worst.1.min(fr) {
                simplex[k] = (xc, fc);
            } else {
                let x0 = simplex[0].0.clone();
                for s in simplex.iter_mut().skip(1) {
                    s.0 = point(&x0, &s.0, 0.5);
                    s.1 = f(&s.0);
                }
            }
        }
    }
    simplex.sort_by(|a, b| a.1.total_cmp(&b.1));
    simplex.swap_remove(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Simulated GJR returns (uniform innovations scaled to unit variance, deterministic).
    fn simulate(n: usize, p: &GarchParams) -> Vec<f64> {
        let mut state: u64 = 42;
        let mut u = || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (state >> 11) as f64 / (1u64 << 53) as f64
        };
        let mut s2 = p.omega / (1.0 - p.persistence());
        (0..n)
            .map(|_| {
                // Sum of 4 uniforms: close to normal, unit variance after scaling.
                let z = ((0..4).map(|_| u()).sum::<f64>() - 2.0) * 3f64.sqrt();
                let r = s2.sqrt() * z;
                s2 = next_variance(p, s2, r);
                r
            })
            .collect()
    }

    #[test]
    fn recursion_and_backcast() {
        let r = [0.01, -0.02, 0.005, -0.001];
        let bc = backcast(&r);
        let w = [1.0, 0.94, 0.94 * 0.94, 0.94f64.powi(3)];
        let want = r.iter().zip(&w).map(|(r, w)| w * r * r).sum::<f64>() / w.iter().sum::<f64>();
        assert!((bc - want).abs() < 1e-18);
        let p = GarchParams {
            omega: 1e-6,
            alpha: 0.05,
            gamma: 0.1,
            beta: 0.9,
        };
        let v = variances(&r, &p, bc);
        assert_eq!(v.len(), 5);
        assert!((v[0] - (1e-6 + 1.0 * bc)).abs() < 1e-18);
        assert!((v[2] - (1e-6 + 0.15 * 4e-4 + 0.9 * v[1])).abs() < 1e-18);
        assert!((v[3] - (1e-6 + 0.05 * 2.5e-5 + 0.9 * v[2])).abs() < 1e-18);
    }

    #[test]
    fn fit_recovers_simulated_parameters() {
        let truth = GarchParams {
            omega: 2e-6,
            alpha: 0.03,
            gamma: 0.12,
            beta: 0.88,
        };
        let r = simulate(5000, &truth);
        let fit = garch_fit(&r, true).unwrap();
        let p = fit.params;
        assert!((p.alpha - truth.alpha).abs() < 0.03, "{p:?}");
        assert!((p.gamma - truth.gamma).abs() < 0.05, "{p:?}");
        assert!((p.beta - truth.beta).abs() < 0.04, "{p:?}");
        // The optimum beats the truth and every small perturbation of it.
        assert!(fit.loglik >= loglik(&r, &truth, fit.backcast));
        for (i, h) in [(0, 1e-8), (1, 1e-4), (2, 1e-4), (3, 1e-4)] {
            for sign in [-1.0, 1.0] {
                let mut q = p;
                match i {
                    0 => q.omega += sign * h,
                    1 => q.alpha += sign * h,
                    2 => q.gamma += sign * h,
                    _ => q.beta += sign * h,
                }
                if q.feasible() {
                    assert!(loglik(&r, &q, fit.backcast) <= fit.loglik + 1e-9, "{i} {sign}");
                }
            }
        }
        let g = garch_fit(&r, false).unwrap();
        assert_eq!((g.model.as_str(), g.params.gamma), ("garch", 0.0));
        assert!(g.loglik < fit.loglik);
        assert!(garch_fit(&r[..50], true).is_err());
    }
}
