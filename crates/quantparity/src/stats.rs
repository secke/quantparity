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

/// Regularized lower incomplete gamma function P(a, x) by its series, accurate for x < a + 1.
fn gamma_p_series(a: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    let front = (-x + a * x.ln() - ln_gamma(a)).exp();
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
    sum * front
}

/// Regularized upper incomplete gamma function Q(a, x).
pub fn gamma_q(a: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 1.0;
    }
    let front = (-x + a * x.ln() - ln_gamma(a)).exp();
    if x < a + 1.0 {
        1.0 - gamma_p_series(a, x)
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

/// Standard normal cumulative distribution function (via the incomplete gamma function).
fn phi(x: f64) -> f64 {
    let q = 0.5 * gamma_q(0.5, x * x / 2.0);
    if x >= 0.0 {
        1.0 - q
    } else {
        q
    }
}

/// Quantile of the standard normal distribution: Acklam's rational approximation refined by
/// one Halley step on the lower tail, for full double precision.
pub fn norm_inv(p: f64) -> f64 {
    if !(0.0..=1.0).contains(&p) || p.is_nan() {
        return f64::NAN;
    }
    if p == 0.0 {
        return f64::NEG_INFINITY;
    }
    if p == 1.0 {
        return f64::INFINITY;
    }
    if p > 0.5 {
        return -norm_inv(1.0 - p);
    }
    const A: [f64; 6] = [
        -3.969_683_028_665_376e1,
        2.209_460_984_245_205e2,
        -2.759_285_104_469_687e2,
        1.383_577_518_672_69e2,
        -3.066_479_806_614_716e1,
        2.506_628_277_459_239,
    ];
    const B: [f64; 5] = [
        -5.447_609_879_822_406e1,
        1.615_858_368_580_409e2,
        -1.556_989_798_598_866e2,
        6.680_131_188_771_972e1,
        -1.328_068_155_288_572e1,
    ];
    const C: [f64; 6] = [
        -7.784_894_002_430_293e-3,
        -3.223_964_580_411_365e-1,
        -2.400_758_277_161_838,
        -2.549_732_539_343_734,
        4.374_664_141_464_968,
        2.938_163_982_698_783,
    ];
    const D: [f64; 4] = [
        7.784_695_709_041_462e-3,
        3.224_671_290_700_398e-1,
        2.445_134_137_142_996,
        3.754_408_661_907_416,
    ];
    let x = if p < 0.02425 {
        let q = (-2.0 * p.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else {
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    };
    // Halley refinement on e = phi(x) - p. Near the centre phi(x) - 1/2 = -P(1/2, x²/2) / 2
    // (x ≤ 0) and p - 1/2 is exact, which avoids the cancellation of phi(x) - p.
    let e = if x * x / 2.0 < 1.5 {
        -0.5 * gamma_p_series(0.5, x * x / 2.0) - (p - 0.5)
    } else {
        phi(x) - p
    };
    let u = e * (2.0 * PI).sqrt() * (x * x / 2.0).exp();
    x - u / (1.0 + x * u / 2.0)
}

/// Continued fraction of the regularized incomplete beta function (modified Lentz).
fn beta_cf(a: f64, b: f64, x: f64) -> f64 {
    let tiny = 1e-300;
    let (qab, qap, qam) = (a + b, a + 1.0, a - 1.0);
    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < tiny {
        d = tiny;
    }
    d = 1.0 / d;
    let mut h = d;
    for m in 1..10_000 {
        let m = m as f64;
        let m2 = 2.0 * m;
        let aa = m * (b - m) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if d.abs() < tiny {
            d = tiny;
        }
        c = 1.0 + aa / c;
        if c.abs() < tiny {
            c = tiny;
        }
        d = 1.0 / d;
        h *= d * c;
        let aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if d.abs() < tiny {
            d = tiny;
        }
        c = 1.0 + aa / c;
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
    h
}

/// Regularized incomplete beta function I_x(a, b).
pub fn beta_inc(a: f64, b: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    let ln_front = ln_gamma(a + b) - ln_gamma(a) - ln_gamma(b) + a * x.ln() + b * (1.0 - x).ln();
    if x < (a + 1.0) / (a + b + 2.0) {
        ln_front.exp() * beta_cf(a, b, x) / a
    } else {
        1.0 - ln_front.exp() * beta_cf(b, a, 1.0 - x) / b
    }
}

/// Density of Student's t distribution with `nu` degrees of freedom.
pub fn t_pdf(x: f64, nu: f64) -> f64 {
    (ln_gamma((nu + 1.0) / 2.0)
        - ln_gamma(nu / 2.0)
        - 0.5 * (nu * PI).ln()
        - (nu + 1.0) / 2.0 * (1.0 + x * x / nu).ln())
    .exp()
}

/// P(T ≤ x) for Student's t with `nu` degrees of freedom (`nu` > 0, not necessarily an integer).
pub fn t_cdf(x: f64, nu: f64) -> f64 {
    // Tail probability P(T > |x|), computed directly so that small tails keep their precision.
    let tail = 0.5 * beta_inc(nu / 2.0, 0.5, nu / (nu + x * x));
    if x >= 0.0 {
        1.0 - tail
    } else {
        tail
    }
}

/// Quantile of Student's t distribution: Newton iterations on the tail probability, started from
/// the Cornish-Fisher expansion and safeguarded by bisection.
pub fn t_inv(p: f64, nu: f64) -> f64 {
    if !(0.0..=1.0).contains(&p) || p.is_nan() || nu.is_nan() || nu <= 0.0 {
        return f64::NAN;
    }
    if p == 0.5 {
        return 0.0;
    }
    if p > 0.5 {
        return -t_inv(1.0 - p, nu);
    }
    if p == 0.0 {
        return f64::NEG_INFINITY;
    }
    // Solve P(T > x) = p for x > 0, then return -x.
    let z = -norm_inv(p);
    let g1 = (z.powi(3) + z) / 4.0;
    let g2 = (5.0 * z.powi(5) + 16.0 * z.powi(3) + 3.0 * z) / 96.0;
    let mut x = (z + g1 / nu + g2 / (nu * nu)).max(1e-3);
    let tail = |x: f64| 0.5 * beta_inc(nu / 2.0, 0.5, nu / (nu + x * x));
    // Bracket [lo, hi] with tail(lo) > p > tail(hi).
    let (mut lo, mut hi) = (0.0, x.max(1.0));
    while tail(hi) > p {
        lo = hi;
        hi *= 2.0;
        if hi > 1e300 {
            return f64::NEG_INFINITY;
        }
    }
    if !(lo..=hi).contains(&x) {
        x = 0.5 * (lo + hi);
    }
    for _ in 0..200 {
        let f = tail(x) - p;
        if f > 0.0 {
            lo = x;
        } else {
            hi = x;
        }
        let step = f / t_pdf(x, nu); // d tail / dx = -pdf
        let mut next = x + step;
        if !(next > lo && next < hi && next.is_finite()) {
            next = 0.5 * (lo + hi);
        }
        if (next - x).abs() <= 1e-15 * x.abs() {
            x = next;
            break;
        }
        x = next;
    }
    -x
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
    fn normal_and_student_quantiles() {
        // Reference values: Abramowitz & Stegun table 26.5 and standard t tables.
        assert!(close(norm_inv(0.99), 2.326_347_874_040_840_8, 1e-15));
        assert!(close(norm_inv(0.975), 1.959_963_984_540_054, 2e-15));
        assert!(close(norm_inv(0.500_000_1) / 2.506_628_273_311_648e-7, 1.0, 1e-14));
        assert!(close(norm_inv(1e-10), -6.361_340_902_404_056, 1e-12));
        assert!(close(t_inv(0.99, 5.0), 3.364_929_998_907_218, 1e-13));
        assert!(close(t_inv(0.025, 10.0), -2.228_138_851_986_274, 1e-13));
        assert!(close(t_inv(0.999, 3.0), 10.214_531_852_407_38, 1e-11));
        assert!(close(t_inv(0.99, 1e6), norm_inv(0.99), 1e-5));
        for nu in [2.5, 4.0, 7.3, 30.0] {
            for p in [1e-6, 0.01, 0.2, 0.7, 0.975] {
                assert!(close(t_cdf(t_inv(p, nu), nu), p, 1e-14 * p.max(1e-2) / 1e-2));
            }
        }
        // Cauchy (nu = 1): F(x) = 1/2 + atan(x)/pi.
        assert!(close(t_cdf(2.0, 1.0), 0.5 + 2f64.atan() / PI, 1e-14));
    }

    #[test]
    fn ln_gamma_factorials() {
        assert!(close(ln_gamma(6.0), 120f64.ln(), 1e-12));
        assert!(close(ln_gamma(0.5), PI.sqrt().ln(), 1e-12));
    }
}
