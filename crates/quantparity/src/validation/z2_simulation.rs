//! P-values of the Acerbi-Székely Z2 statistic by simulation from the forecast distribution.
//!
//! The fixed Z2 thresholds (-0.70, -1.8) assume moderate tails; with very heavy tails a correct
//! model is rejected more often than the thresholds say. Acerbi and Székely's own recipe is to
//! simulate Z2 under the null hypothesis that the forecasts are right: on each day t, draw X_t from
//! the forecast distribution P_t and recompute Z2.
//!
//! With X_t = σ_t z_t, VaR_t = σ_t q_t and ES_t = σ_t e_t, the volatility cancels out of Z2, so only
//! the standardized law of each day is needed: the normal or unit-variance Student-t for a
//! parametric model, the past standardized returns of that day for the filtered historical
//! simulation (with their discreteness: an exception has probability m_t / n_t, the share of past
//! residuals strictly below the VaR). The random numbers come from xoshiro256** seeded by
//! SplitMix64, in blocks of 1,000 samples simulated in parallel, so a p-value is reproducible
//! from its seed whatever the number of threads.
//!
//! With n simulations the smallest p-value is 1 / (n + 1): a red zone (p < 0.01%) needs at least
//! 10,000 and is reliable from about 100,000.

use crate::stats::{norm_inv, t_inv};
use crate::var_models::{tail, volatility_path, Innovations, VarForecasts, Volatility};
use crate::Status;
use serde::{Deserialize, Serialize};

/// Law of z_t / e_t on an exception day, and the probability of an exception.
#[derive(Debug, Clone)]
enum DayLaw {
    /// Same law every day: z = scale · F⁻¹(u) for u < 1 - es_confidence.
    Parametric { nu: Option<f64>, scale: f64, es: f64 },
    /// Past standardized returns strictly below the VaR of the day, divided by its ES; n residuals.
    Empirical { tail: Vec<f64>, n: usize },
}

/// The null distribution of Z2 for a set of forecasts.
#[derive(Debug, Clone)]
pub struct Z2Null {
    start: usize,
    level: f64,
    days: Vec<DayLaw>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Z2Simulation {
    pub z2: f64,
    /// (1 + number of simulated Z2 ≤ z2) / (1 + simulations).
    pub p_value: f64,
    pub simulations: usize,
    pub seed: u64,
    pub observations: usize,
    /// 5% quantile of the simulated Z2 (the counterpart of -0.70).
    pub critical_5: f64,
    /// 0.01% quantile of the simulated Z2 (the counterpart of -1.8), with at least 10,000 simulations.
    pub critical_0_01: Option<f64>,
    /// green: p ≥ 5%, yellow: p ≥ 0.01%, red below.
    pub zone: String,
    pub status: Status,
}

const BLOCK: usize = 1_000;

struct Rng([u64; 4]);

impl Rng {
    fn new(seed: u64) -> Self {
        let mut x = seed;
        let mut next = || {
            x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = x;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        };
        Rng([next(), next(), next(), next()])
    }

    /// Uniform on (0, 1): (k + 0.5) / 2^53.
    fn uniform(&mut self) -> f64 {
        let s = &mut self.0;
        let r = s[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = s[1] << 17;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = s[3].rotate_left(45);
        ((r >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }
}

impl Z2Null {
    /// Rebuilds the standardized law of every forecast day from the model recorded in `f` and the
    /// `returns` it was computed from; fails if the forecasts do not match their model.
    pub fn new(returns: &[f64], f: &VarForecasts) -> Result<Self, String> {
        let volatility = match (f.lambda, f.volatility) {
            (Some(lambda), _) => Volatility::Ewma { lambda },
            (None, Some(v)) => v,
            (None, None) => return Err("forecasts without a volatility model".into()),
        };
        if f.warmup == 0 || returns.len() <= f.warmup || f.start + f.es.len() != returns.len() {
            return Err("the forecasts do not cover the returns".into());
        }
        let level = f.es_confidence;
        let matches = |a: f64, b: f64| (a - b).abs() <= 1e-12 * b.abs().max(1e-300);
        let days = match f.innovations {
            Innovations::Normal | Innovations::StudentT { .. } => {
                let m = f.multipliers.ok_or("parametric forecasts without multipliers")?;
                let (nu, scale) = match f.innovations {
                    Innovations::StudentT { nu } => (Some(nu), ((nu - 2.0) / nu).sqrt()),
                    _ => (None, 1.0),
                };
                if f.es.iter().zip(&f.sigma).any(|(e, s)| !matches(*e, s * m[1])) {
                    return Err("the ES forecasts do not match their model".into());
                }
                vec![DayLaw::Parametric { nu, scale, es: m[1] }]
            }
            Innovations::Filtered { window, .. } => {
                let (sigma, _) = volatility_path(returns, volatility, f.warmup)?;
                let w = f.warmup;
                let mut sorted: Vec<f64> = vec![];
                let mut days = Vec::with_capacity(f.es.len());
                for t in w..returns.len() {
                    if t >= f.start {
                        let (j, sd) = (t - f.start, sigma[t - w]);
                        let (q, e) = tail(&sorted, level);
                        if !matches(sd * e, f.es[j]) || !matches(sd * q, f.es_var[j]) {
                            return Err("the ES forecasts do not match their model".into());
                        }
                        let m = sorted.partition_point(|z| *z < -q);
                        days.push(DayLaw::Empirical {
                            tail: sorted[..m].iter().map(|z| z / e).collect(),
                            n: sorted.len(),
                        });
                    }
                    let x = returns[t] / sigma[t - w];
                    sorted.insert(sorted.partition_point(|y| *y < x), x);
                    if let Some(win) = window {
                        if t >= w + win {
                            let old = returns[t - win] / sigma[t - win - w];
                            sorted.remove(sorted.partition_point(|y| *y < old));
                        }
                    }
                }
                days
            }
        };
        Ok(Z2Null {
            start: f.start,
            level,
            days,
        })
    }

    fn law(&self, t: usize) -> &DayLaw {
        if self.days.len() == 1 {
            &self.days[0]
        } else {
            &self.days[t - self.start]
        }
    }

    /// Simulated p-value of `z2`, the Z2 observed on returns `from..to` (positions in the returns).
    pub fn test(&self, z2: f64, from: usize, to: usize, simulations: usize, seed: u64) -> Result<Z2Simulation, String> {
        let end = self.start
            + if self.days.len() == 1 {
                usize::MAX - self.start
            } else {
                self.days.len()
            };
        if from < self.start || to > end || from >= to || simulations == 0 || !z2.is_finite() {
            return Err("simulation: days outside the forecasts, or no simulation".into());
        }
        let p = 1.0 - self.level;
        let denom = (to - from) as f64 * p;
        // Blocks of 1,000 samples, each with its own stream (seed, block): the result does not
        // depend on the number of threads.
        let simulate_block = |block: usize| -> Vec<f64> {
            let mut rng = Rng::new(seed ^ (block as u64).wrapping_mul(0xD1B5_4A32_D192_ED03));
            let size = BLOCK.min(simulations - block * BLOCK);
            (0..size)
                .map(|_| {
                    let mut s = 0.0;
                    for t in from..to {
                        let u = rng.uniform();
                        match self.law(t) {
                            DayLaw::Parametric { nu, scale, es } => {
                                if u < p {
                                    let z = match nu {
                                        None => norm_inv(u),
                                        Some(nu) => t_inv(u, *nu),
                                    };
                                    s += scale * z / es;
                                }
                            }
                            DayLaw::Empirical { tail, n } => {
                                let i = (u * *n as f64) as usize;
                                if i < tail.len() {
                                    s += tail[i];
                                }
                            }
                        }
                    }
                    s / denom + 1.0
                })
                .collect()
        };
        let blocks = simulations.div_ceil(BLOCK);
        let threads = std::thread::available_parallelism().map_or(1, |n| n.get()).min(blocks);
        let next = std::sync::atomic::AtomicUsize::new(0);
        let mut done: Vec<(usize, Vec<f64>)> = std::thread::scope(|scope| {
            let workers: Vec<_> = (0..threads)
                .map(|_| {
                    scope.spawn(|| {
                        let mut mine = vec![];
                        loop {
                            let b = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            if b >= blocks {
                                break mine;
                            }
                            mine.push((b, simulate_block(b)));
                        }
                    })
                })
                .collect();
            workers.into_iter().flat_map(|w| w.join().unwrap()).collect()
        });
        done.sort_by_key(|(b, _)| *b);
        let mut sims: Vec<f64> = done.into_iter().flat_map(|(_, v)| v).collect();
        sims.sort_by(f64::total_cmp);
        let below = sims.partition_point(|x| *x <= z2);
        let p_value = (1 + below) as f64 / (1 + simulations) as f64;
        let (zone, status) = if p_value >= 0.05 {
            ("green", Status::Pass)
        } else if p_value >= 1e-4 {
            ("yellow", Status::Warn)
        } else {
            ("red", Status::Fail)
        };
        Ok(Z2Simulation {
            z2,
            p_value,
            simulations,
            seed,
            observations: to - from,
            critical_5: sims[(0.05 * simulations as f64) as usize],
            critical_0_01: (simulations >= 10_000).then(|| sims[(1e-4 * simulations as f64) as usize]),
            zone: zone.into(),
            status,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::validation::es_backtest;
    use crate::var_models::{var_forecasts, var_forecasts_with};

    fn returns(n: usize) -> Vec<f64> {
        let mut rng = Rng::new(3);
        (0..n)
            .map(|i| {
                let vol = if i % 400 < 200 { 0.01 } else { 0.02 };
                let z = norm_inv(rng.uniform());
                vol * if rng.uniform() < 0.05 { 3.0 * z } else { z }
            })
            .collect()
    }

    #[test]
    fn rng_is_uniform_and_reproducible() {
        let mut a = Rng::new(7);
        let x: Vec<f64> = (0..100_000).map(|_| a.uniform()).collect();
        let mean = x.iter().sum::<f64>() / x.len() as f64;
        assert!((mean - 0.5).abs() < 0.005 && x.iter().all(|u| *u > 0.0 && *u < 1.0));
        assert_eq!(Rng::new(7).uniform(), x[0]);
    }

    #[test]
    fn parametric_null_matches_known_thresholds() {
        // Normal forecasts on 250 days: Acerbi and Székely's 5% threshold is about -0.70.
        let r = returns(400);
        let f = var_forecasts(&r, 0.94, 30, 0.99, 0.975, Innovations::Normal).unwrap();
        let null = Z2Null::new(&r, &f).unwrap();
        let s = null.test(-0.70, 100, 350, 20_000, 1).unwrap();
        assert!((s.critical_5 + 0.70).abs() < 0.08, "{}", s.critical_5);
        assert!((s.p_value - 0.05).abs() < 0.015, "{}", s.p_value);
        assert_eq!(s, null.test(-0.70, 100, 350, 20_000, 1).unwrap());
        // The observed Z2 of the forecasts gets a p-value in (0, 1].
        let realized = &r[f.start..];
        let z2 = es_backtest(realized, &f.es_var, &f.es, 0.975).unwrap().z2;
        let all = null.test(z2, f.start, r.len(), 2_000, 9).unwrap();
        assert!(all.p_value > 0.0 && all.p_value <= 1.0 && all.observations == 370);
        // Heavier tails, lower 5% quantile than the normal one.
        let t3 = var_forecasts(&r, 0.94, 30, 0.99, 0.975, Innovations::StudentT { nu: 3.0 }).unwrap();
        let s3 = Z2Null::new(&r, &t3).unwrap().test(-0.70, 100, 350, 20_000, 1).unwrap();
        assert!(
            s3.critical_5 < s.critical_5 - 0.05,
            "{} {}",
            s3.critical_5,
            s.critical_5
        );
    }

    #[test]
    fn empirical_null_uses_the_residuals_of_each_day() {
        let r = returns(900);
        let fhs = Innovations::Filtered {
            window: Some(400),
            min_residuals: 250,
        };
        let vol = Volatility::Garch {
            asymmetric: true,
            refit: 300,
            window: None,
        };
        let f = var_forecasts_with(&r, vol, 300, 0.99, 0.975, fhs).unwrap();
        let null = Z2Null::new(&r, &f).unwrap();
        // Day 700: residuals of days 300..700, strictly below the VaR, divided by the ES.
        let t = 700;
        let (sigma, _) = volatility_path(&r, f.volatility.unwrap(), f.warmup).unwrap();
        let mut z: Vec<f64> = (t - 400..t).map(|s| r[s] / sigma[s - f.warmup]).collect();
        z.sort_by(f64::total_cmp);
        // k = ⌈400 × 2.5%⌉ = 10: the VaR is the 10th worst residual, the ES the mean of the 10 worst.
        let e = -z[..10].iter().sum::<f64>() / 10.0;
        assert!((f.es[t - f.start] - sigma[t - f.warmup] * e).abs() < 1e-15);
        let tail: Vec<f64> = z.iter().filter(|x| **x < z[9]).map(|x| x / e).collect();
        match null.law(t) {
            DayLaw::Empirical { tail: got, n } => {
                assert_eq!(*n, 400);
                assert_eq!(got.len(), tail.len());
                assert!(got.iter().zip(&tail).all(|(a, b)| (a - b).abs() < 1e-12));
            }
            _ => panic!("empirical law expected"),
        }
        let s = null.test(-0.5, f.start, r.len(), 5_000, 2).unwrap();
        assert!(s.p_value > 0.0 && s.critical_0_01.is_none());
        // Forecasts that do not come from their model are rejected.
        let mut g = f.clone();
        g.es[10] *= 1.01;
        assert!(Z2Null::new(&r, &g).is_err());
    }
}
