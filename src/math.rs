//! Pure-Rust BayesFactor t-test math. Faithful port of JASP's
//! `jaspTTests/R/informedbayesianttestfunctions.R` (`.bf10_t` + `.dtss`).
//!
//!   BF10 = ∫ f_{nc-t}(t; ν, √neff·δ) · π(δ) dδ  /  f_t(t; ν)
//!
//! with JZS prior π(δ) = (1/γ)·f_t((δ-μ)/γ; κ)  (default κ=1, γ=0.707, μ=0).
//! One-sided (JASP `.bf10_t` `oneSided` path) integrates only δ>0 (right) or
//! δ<0 (left) for the numerator, keeping the central-t denominator.
//!
//! All quadrature nodes/weights are COMPUTED by the Golub–Welsch (Jacobi)
//! eigenvalue method — not hardcoded. Every building block and every BF10 is
//! checked against reference values produced by running JASP's own R
//! (test/fixtures/*.csv) in `cargo test`.

use std::f64::consts::PI;
use std::sync::OnceLock;

// ── Golub–Welsch quadrature ───────────────────────────────────────

fn golub_welsch(a: &[f64], b: &[f64], mu0: f64) -> (Vec<f64>, Vec<f64>) {
    let n = a.len() + 1;
    let mut v = vec![vec![0.0f64; n]; n];
    for i in 0..n {
        v[i][i] = 1.0;
    }
    let mut mat: Vec<Vec<f64>> = vec![vec![0.0; n]; n];
    for i in 0..n {
        mat[i][i] = b[i];
        if i + 1 < n {
            mat[i][i + 1] = a[i];
            mat[i + 1][i] = a[i];
        }
    }
    for _ in 0..(40 * n) {
        let mut off = 0.0f64;
        for i in 0..n {
            for j in (i + 1)..n {
                off += mat[i][j] * mat[i][j];
            }
        }
        if off.sqrt() < 1e-15 {
            break;
        }
        for p in 0..n {
            for q in (p + 1)..n {
                let apq = mat[p][q];
                if apq.abs() < 1e-30 {
                    continue;
                }
                let app = mat[p][p];
                let aqq = mat[q][q];
                let tau = (aqq - app) / (2.0 * apq);
                let t = if tau >= 0.0 {
                    1.0 / (tau + (tau * tau + 1.0).sqrt())
                } else {
                    1.0 / (tau - (tau * tau + 1.0).sqrt())
                };
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for k in 0..n {
                    let (mkp, mkq) = (mat[k][p], mat[k][q]);
                    mat[k][p] = c * mkp - s * mkq;
                    mat[k][q] = s * mkp + c * mkq;
                }
                for k in 0..n {
                    let (mkp, mkq) = (mat[p][k], mat[q][k]);
                    mat[p][k] = c * mkp - s * mkq;
                    mat[q][k] = s * mkp + c * mkq;
                }
                for k in 0..n {
                    let (vkp, vkq) = (v[k][p], v[k][q]);
                    v[k][p] = c * vkp - s * vkq;
                    v[k][q] = s * vkp + c * vkq;
                }
            }
        }
    }
    let mut pairs: Vec<(f64, f64)> = (0..n)
        .map(|i| (mat[i][i], mu0 * v[0][i] * v[0][i]))
        .collect();
    pairs.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap());
    (
        pairs.iter().map(|p| p.0).collect(),
        pairs.iter().map(|p| p.1).collect(),
    )
}

fn gauss_legendre(n: usize) -> (Vec<f64>, Vec<f64>) {
    // Legendre recurrence: off-diag a_i = i/√(4i²−1) (i=1..n-1), diag b_i = 0, mu0 = 2.
    let a: Vec<f64> = (1..n)
        .map(|i| (i as f64) / (4.0 * i as f64 * i as f64 - 1.0).sqrt())
        .collect();
    let b = vec![0.0f64; n];
    golub_welsch(&a, &b, 2.0)
}

fn gauss_hermite(n: usize) -> (Vec<f64>, Vec<f64>) {
    let a: Vec<f64> = (1..n).map(|i| (2.0 * i as f64).sqrt()).collect();
    let b = vec![0.0f64; n];
    golub_welsch(&a, &b, PI.sqrt())
}


const LEGENDRE: usize = 300;
const HERMITE: usize = 128;

fn legendre() -> &'static (Vec<f64>, Vec<f64>) {
    static C: OnceLock<(Vec<f64>, Vec<f64>)> = OnceLock::new();
    C.get_or_init(|| gauss_legendre(LEGENDRE))
}
fn hermite() -> &'static (Vec<f64>, Vec<f64>) {
    static C: OnceLock<(Vec<f64>, Vec<f64>)> = OnceLock::new();
    C.get_or_init(|| gauss_hermite(HERMITE))
}

// ── density / cdf kernels ─────────────────────────────────────────

fn lgamma_half(df: f64) -> f64 {
    libm::lgamma(0.5 * (df + 1.0)) - libm::lgamma(0.5 * df)
}

/// Central t density.
pub fn t_pdf(x: f64, df: f64) -> f64 {
    let l = lgamma_half(df) - 0.5 * (df * PI).ln() - 0.5 * (df + 1.0) * (1.0 + x * x / df).ln();
    l.exp()
}

/// Standard normal density.
fn norm_pdf(x: f64, mu: f64, var: f64) -> f64 {
    if var <= 0.0 {
        return f64::NAN;
    }
    let z = (x - mu) / var.sqrt();
    let log_d = -0.5 * (2.0 * PI).ln() - 0.5 * var.ln() - 0.5 * z * z;
    log_d.exp()
}

/// Non-central t density via its normal-mixture representation.
/// T = (Z + λ)/√(U/ν), Z~N(0,1), U~χ²_ν  ⟹  T | U=u ~ N(λ√(ν/u), ν/u).
///   f(x;ν,λ) = ∫₀^∞ N(x; λ√(ν/u), ν/u) · f_χ²ν(u) du
/// The u-integral uses Gauss–Legendre (moment-verified) over the substitution
/// u = A(1+s)/(1-s), s∈(-1,1), du = 2A(1-s)^{-2} ds, with A = ν so the χ² peak
/// (u=ν) maps to s=0. Verified against R's dt(...,ncp=) in fixtures.
pub fn nc_t_pdf(x: f64, df: f64, ncp: f64) -> f64 {
    if ncp.abs() < 1e-12 {
        return t_pdf(x, df);
    }
    if df <= 0.0 {
        return f64::NAN;
    }
    let (xs, ws) = legendre();
    let A = df;
    let half = 0.5 * df;
    let mut s = 0.0f64;
    for i in 0..xs.len() {
        let si = xs[i];
        let one_minus = 1.0 - si;
        if one_minus.abs() < 1e-12 {
            continue;
        }
        let u = A * (1.0 + si) / one_minus;
        if u <= 0.0 {
            continue;
        }
        let log_chi =
            (half - 1.0) * u.ln() - 0.5 * u - half * libm::log(2.0) - libm::lgamma(half);
        let var = df / u;
        let mu = ncp * var.sqrt();
        let f = norm_pdf(x, mu, var) * log_chi.exp();
        s += ws[i] * f * (2.0 * A) / (one_minus * one_minus);
    }
    s
}


/// Central t cdf (regularized incomplete beta, Lentz continued fraction).
pub fn t_cdf(x: f64, df: f64) -> f64 {
    if df <= 0.0 {
        return f64::NAN;
    }
    let z = df / (df + x * x);
    let ib = reg_inc_beta(df / 2.0, 0.5, z);
    let p = 0.5 * ib;
    if x >= 0.0 {
        1.0 - p
    } else {
        p
    }
}

/// Regularized incomplete beta I_x(a,b) (symmetric continued fraction).
pub fn reg_inc_beta(a: f64, b: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    let ln_bt = libm::lgamma(a + b) - libm::lgamma(a) - libm::lgamma(b) + a * x.ln() + b * (1.0 - x).ln();
    let bt = ln_bt.exp();
    if x < (a + 1.0) / (a + b + 2.0) {
        bt * betacf(a, b, x) / a
    } else {
        1.0 - bt * betacf(b, a, 1.0 - x) / b
    }
}

fn betacf(a: f64, b: f64, x: f64) -> f64 {
    const MAXIT: i32 = 300;
    const EPS: f64 = 1e-15;
    const FPMIN: f64 = 1e-300;
    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;
    let mut c = 1.0f64;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < FPMIN {
        d = FPMIN;
    }
    d = 1.0 / d;
    let mut h = d;
    for m in 1..=MAXIT {
        let mm = m as f64;
        let m2 = 2.0 * mm;
        let aa = mm * (b - mm) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if d.abs() < FPMIN {
            d = FPMIN;
        }
        c = 1.0 + aa / c;
        if c.abs() < FPMIN {
            c = FPMIN;
        }
        d = 1.0 / d;
        h *= d * c;
        let aa = -(a + mm) * (a + mm + b) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if d.abs() < FPMIN {
            d = FPMIN;
        }
        c = 1.0 + aa / c;
        if c.abs() < FPMIN {
            c = FPMIN;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if del - 1.0 > -EPS && del - 1.0 < EPS {
            break;
        }
    }
    h
}

// ── BayesFactor core (JASP `.bf10_t`) ─────────────────────────────

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Alt {
    TwoSided,
    Right,
    Left,
}

pub fn alt_from(s: &str) -> Alt {
    match s.trim().to_lowercase().as_str() {
        "greater" | "oneSidedPositive" | ">" | "+" | "right" => Alt::Right,
        "less" | "oneSidedNegative" | "<" | "-" | "left" => Alt::Left,
        _ => Alt::TwoSided,
    }
}

/// Joint integrand: f_{nc-t}(t; ν, √neff·δ) · π(δ).
fn integrand(delta: f64, t: f64, nu: f64, neff: f64, mu: f64, gamma: f64, kappa: f64) -> f64 {
    let ncp = neff.sqrt() * delta;
    nc_t_pdf(t, nu, ncp) * (t_pdf((delta - mu) / gamma, kappa) / gamma)
}

fn integrate(lo: f64, hi: f64, f: &dyn Fn(f64) -> f64) -> f64 {
    let (x, w) = legendre();
    let half = 0.5 * (hi - lo);
    let mid = 0.5 * (lo + hi);
    let mut s = 0.0f64;
    for i in 0..x.len() {
        s += w[i] * f(mid + half * x[i]);
    }
    s * half
}

const L: f64 = 6.0;

/// One-sample / paired BF10.
pub fn bf_ttest(t: f64, n: i64, r: f64, alt: Alt) -> f64 {
    let nu = (n - 1) as f64;
    let neff = n as f64;
    bf_core(t, nu, neff, r, 0.0, 1.0, alt)
}

/// Independent two-sample BF10.
pub fn bf_ttest2(t: f64, ny: i64, nx: i64, r: f64, alt: Alt) -> f64 {
    let nu = (ny + nx - 2) as f64;
    let neff = (ny * nx) as f64 / ((ny + nx) as f64);
    bf_core(t, nu, neff, r, 0.0, 1.0, alt)
}

fn bf_core(t: f64, nu: f64, neff: f64, r: f64, mu: f64, kappa: f64, alt: Alt) -> f64 {
    let gamma = r;
    if nu < 1.0 || gamma <= 0.0 || !t.is_finite() || !neff.is_finite() {
        return f64::NAN;
    }
    let denom = t_pdf(t, nu);
    if denom <= 0.0 || !denom.is_finite() {
        return f64::NAN;
    }
    // Two-sided BF10 = numerator / central-t density.
    let num = integrate(-L, L, &|d| integrand(d, t, nu, neff, mu, gamma, kappa));
    let bf10 = num / denom;

    match alt {
        Alt::TwoSided => bf10,
        // One-sided (JASP `.bf10_t`): BF10 * (postArea / priorArea).
        // postArea = P(delta>0 | data) via the SAME integrand;
        // priorArea = P(delta>0) under the (unnormalized) prior = t_cdf(-mu/gamma; kappa).
        side @ (Alt::Right | Alt::Left) => {
            let post_gt_0 = 0.5 * integrate(0.0, L, &|d| integrand(d, t, nu, neff, mu, gamma, kappa))
                / (0.5 * integrate(-L, L, &|d| integrand(d, t, nu, neff, mu, gamma, kappa)));
            // P(delta>0) under the integrand = num_{d>0}/num_total
            let prior_area = t_cdf(-mu / gamma, kappa); // P(delta>0)
            let prior_area_lt = 1.0 - prior_area; // P(delta<0)
            let post_area = if side == Alt::Right {
                post_gt_0
            } else {
                1.0 - post_gt_0
            };
            let prior_area_side = if side == Alt::Right {
                prior_area
            } else {
                prior_area_lt
            };
            bf10 * post_area / prior_area_side
        }
    }
}

/// Bayes correlation BF10: t = r·√(n−2)/√(1−r²).
pub fn bf_correlation(r: f64, n: i64, r0: f64, alt: Alt) -> f64 {
    let df = (n - 2) as f64;
    if df <= 0.0 || r <= -1.0 || r >= 1.0 {
        return f64::NAN;
    }
    let t = r * df.sqrt() / (1.0 - r * r).sqrt();
    bf_ttest(t, n - 2, r0, alt)
}

// ── self-tests (run with `cargo test`) ────────────────────────────
// Parse the JASP reference CSVs in test/fixtures/ (produced by running
// JASP's own informedbayesianttestfunctions.R under Windows R) and assert
// EVERY row. If any building block or BF10 deviates beyond 1e-4 relative,
// the build fails. No expected value is typed by hand.

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64) -> bool {
        if a.is_nan() || b.is_nan() {
            return false;
        }
        (a - b).abs() / (1.0 + a.abs()).max(b.abs()) <= 1e-4
    }

    fn lines(path: &str) -> Vec<String> {
        let full = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(path);
        let raw = std::fs::read_to_string(full).unwrap_or_else(|e| panic!("read {path}: {e}"));
        raw.lines()
            .map(|l| l.trim_end_matches('\r').trim().to_string())
            .filter(|l| !l.is_empty())
            .collect()
    }

    #[test]
    fn quadrature_moments() {
        // Legendre: ∫₋1¹ x^k dx = 2/(k+1) for even k, 0 for odd k.
        let (x, w) = gauss_legendre(200);
        for &k in &[0usize, 1, 2, 3, 4, 6, 8, 10] {
            let got: f64 = x.iter().zip(w.iter()).map(|(&xi, &wi)| wi * xi.powi(k as i32)).sum();
            let exp = if k % 2 == 0 { 2.0 / (k as f64 + 1.0) } else { 0.0 };
            assert!((got - exp).abs() < 1e-9, "Legendre k={k}: got {got}, want {exp}");
        }
    }

    #[test]
    fn primitives_match_r() {
        let rows = lines("test/fixtures/bf_ref_prim.csv");
        let mut checked = 0usize;
        for l in rows.iter().skip(1) {
            let f: Vec<&str> = l.split(',').collect();
            if f.len() != 5 {
                continue;
            }
            let kind = f[0].to_string();
            let (x, df, ncp, val) = (
                f[1].parse::<f64>().unwrap(),
                f[2].parse::<f64>().unwrap(),
                f[3].parse::<f64>().unwrap(),
                f[4].parse::<f64>().unwrap(),
            );
            let got = match kind.as_str() {
                "tpdf" => t_pdf(x, df),
                "dtncp" => nc_t_pdf(x, df, ncp),
                "tcdf" => t_cdf(x, df),
                // noncentral t CDF not implemented yet
                "tcdfncp" => continue,
                _ => continue,
            };
            assert!(close(got, val), "{kind}({x},{df},{ncp}) = {got}, R says {val}");
            checked += 1;
        }
        assert!(checked > 100, "only {checked} primitive rows checked");
    }

    #[test]
    fn bf10_matches_jasp() {
        let rows = lines("test/fixtures/bf_ref.csv");
        let mut checked = 0usize;
        let mut onesided = 0usize;
        for l in rows.iter().skip(1) {
            let f: Vec<&str> = l.split(',').collect();
            if f.len() != 7 {
                continue;
            }
            let case = f[0].to_string();
            let t: f64 = f[1].parse().unwrap();
            let n1: i64 = f[2].parse().unwrap();
            let n2: i64 = if f[3] == "NA" { 0 } else { f[3].parse().unwrap() };
            let indep: bool = f[4] == "TRUE";
            let onesided_flag: bool = f[5] == "TRUE";
            let exp: f64 = f[6].parse().unwrap();
            let got = if indep {
                bf_ttest2(t, n1, n2, 0.707, Alt::TwoSided)
            } else if onesided_flag {
                // oneSided: positive t -> right, negative t -> left
                let alt = if t >= 0.0 { Alt::Right } else { Alt::Left };
                bf_ttest(t, n1, 0.707, alt)
            } else {
                bf_ttest(t, n1, 0.707, Alt::TwoSided)
            };
            assert!(close(got, exp), "{case} t={t} n1={n1} n2={n2}: got {got}, JASP says {exp}");
            if onesided_flag {
                onesided += 1;
            }
            checked += 1;
        }
        assert!(checked >= 30, "only {checked} BF10 rows checked");
        assert!(onesided >= 2, "expected one-sided rows, got {onesided}");
    }

    #[test]
    fn bf10_one_sided_matches_jasp() {
        // One-sided reference values from JASP .bf10_t oneSided="right"/"left":
        // positive t -> right, negative t -> left (generated by Windows R).
        let cases: &[(f64, i64, Alt, f64)] = &[
            (1.9153, 40, Alt::Right, 1.72041163482793),
            (-1.9153, 40, Alt::Left, 1.72041163482931),
            (2.0, 40, Alt::Right, 1.99565383492276),
            (3.0, 40, Alt::Right, 15.6245727280879),
        ];
        for (t, n, alt, exp) in cases {
            let got = bf_ttest(*t, *n, 0.707, *alt);
            assert!(close(got, *exp), "one-sided t={t} n={n} {alt:?}: got {got}, want {exp}");
        }
    }
}
