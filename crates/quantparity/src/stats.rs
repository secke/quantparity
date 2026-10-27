//! Basic statistical functions (no external dependency).

use std::f64::consts::PI;

/// ln Γ(x), Lanczos approximation (g = 7, n = 9).
pub fn ln_gamma(x: f64) -> f64 {
    const C: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    if x < 0.5 {
        PI.ln() - (PI * x).sin().ln() - ln_gamma(1.0 - x)
    } else {
        let x = x - 1.0;
        let t = x + 7.5;
        let a = C[0] + (1..9).map(|i| C[i] / (x + i as f64)).sum::<f64>();
        0.5 * (2.0 * PI).ln() + (x + 0.5) * t.ln() - t + a.ln()
    }
}

/// Regularized upper incomplete gamma function Q(a, x).
pub fn gamma_q(a: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 1.0;
    }
    let front = (-x + a * x.ln() - ln_gamma(a)).exp();
    if x < a + 1.0 {
        // series for P(a, x)
        let (mut ap, mut sum) = (a, 1.0 / a);
        let mut del = sum;
        for _ in 0..1000 {
            ap += 1.0;
            del *= x / ap;
            sum += del;
            if del.abs() < sum.abs() * 1e-16 {
                break;
            }
        }
        1.0 - sum * front
    } else {
        // Lentz continued fraction for Q(a, x)
        let tiny = 1e-300;
        let mut b = x + 1.0 - a;
        let mut c = 1.0 / tiny;
        let mut d = 1.0 / b;
        let mut h = d;
        for i in 1..1000 {
            let an = -(i as f64) * (i as f64 - a);
            b += 2.0;
            d = an * d + b;
            if d.abs() < tiny {
                d = tiny;
            }
            c = b + an / c;
            if c.abs() < tiny {
                c = tiny;
            }
            d = 1.0 / d;
            let del = d * c;
            h *= del;
            if (del - 1.0).abs() < 1e-16 {
                break;
            }
        }
        front * h
    }
}

/// P(χ²_k > x).
pub fn chi2_sf(x: f64, k: f64) -> f64 {
    gamma_q(k / 2.0, x / 2.0).clamp(0.0, 1.0)
}

fn ln_binom_pmf(i: u64, n: u64, p: f64) -> f64 {
    let (i, nf) = (i as f64, n as f64);
    ln_gamma(nf + 1.0) - ln_gamma(i + 1.0) - ln_gamma(nf - i + 1.0) + xlny(i, p) + xlny(nf - i, 1.0 - p)
}

/// P(X ≤ k), X ~ B(n, p).
pub fn binom_cdf(k: u64, n: u64, p: f64) -> f64 {
    if k >= n {
        return 1.0;
    }
    (0..=k)
        .map(|i| ln_binom_pmf(i, n, p).exp())
        .sum::<f64>()
        .clamp(0.0, 1.0)
}

/// P(X ≥ k), X ~ B(n, p).
pub fn binom_sf(k: u64, n: u64, p: f64) -> f64 {
    if k == 0 {
        return 1.0;
    }
    if k > n {
        return 0.0;
    }
    (k..=n)
        .map(|i| ln_binom_pmf(i, n, p).exp())
        .sum::<f64>()
        .clamp(0.0, 1.0)
}

/// x · ln(y) with the convention 0 · ln(0) = 0.
pub fn xlny(x: f64, y: f64) -> f64 {
    if x == 0.0 {
        0.0
    } else {
        x * y.ln()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn chi2_critical_values() {
        assert!(close(chi2_sf(3.841_458_820_694_124, 1.0), 0.05, 1e-9));
        assert!(close(chi2_sf(5.991_464_547_107_979, 2.0), 0.05, 1e-9));
        assert!(close(chi2_sf(18.307_038_053_275_146, 10.0), 0.05, 1e-9));
    }

    #[test]
    fn binomial_matches_basel_table() {
        // Basel Committee table (1996), 250 observations at 99%.
        assert!(close(binom_cdf(4, 250, 0.01), 0.8922, 1e-4));
        assert!(close(binom_cdf(9, 250, 0.01), 0.9997, 1e-4));
        assert!(close(binom_sf(5, 250, 0.01) + binom_cdf(4, 250, 0.01), 1.0, 1e-12));
    }

    #[test]
    fn ln_gamma_factorials() {
        assert!(close(ln_gamma(6.0), 120f64.ln(), 1e-12));
        assert!(close(ln_gamma(0.5), PI.sqrt().ln(), 1e-12));
    }
}
