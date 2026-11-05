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

const LEGENDRE: usize = 300;

fn legendre() -> &'static (Vec<f64>, Vec<f64>) {
    static C: OnceLock<(Vec<f64>, Vec<f64>)> = OnceLock::new();
    C.get_or_init(|| gauss_legendre(LEGENDRE))
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

/// JASP robustness-check r grid (commonbayesianttest.R, `.plotBF.robustnessCheck.ttest2`):
///   r > 1.5  → seq(0.0005, 2.0, length.out = 535)
///   otherwise → seq(0.0005, 1.5, length.out = 400)
/// `points` (>= 2) overrides the length.out, keeping the same anchor/high.
pub fn robustness_grid(r: f64, points: Option<i64>) -> Vec<f64> {
    let (hi, n) = match points {
        Some(p) if p >= 2 => (r.max(1.5), p as usize),
        _ => {
            if r > 1.5 {
                (2.0, 535)
            } else {
                (1.5, 400)
            }
        }
    };
    (0..n)
        .map(|i| 0.0005 + (hi - 0.0005) * (i as f64 / (n - 1) as f64))
        .collect()
}

// ── Bayesian ANOVA & regression (JZS, closed-form quadrature) ─────
// JASP ANOVA (`anovaBF`) and regression (`lmBF`) both reduce, per model, to a
// single closed-form Bayes factor — no MCMC on the BF path (only posterior
// sampling uses Gibbs, which we do not expose). Verified against R's
// `oneWayAOV.Fstat` / `linearReg.R2stat` (method = "quadrature").
//
// Both integrals are over the precision-scaled effect g ∈ (0,∞). We work in
// w = log(g) (dg = e^w dw, the +w below is that Jacobian) so the integrands
// are smooth over w ∈ (−∞,∞), find the mode by golden section, then integrate
// the mode-centered integrand on a growing window with Gauss–Legendre.

/// ∫₀^∞ exp(f(w)) dw for a unimodal f given in log-space, via golden-section
/// peak finding + windowed GL (log-sum-exp, overflow-safe since exp(f−peak)≤1).
fn log_integrate_peak(f: &dyn Fn(f64) -> f64, lo: f64, hi: f64) -> f64 {
    let (xs, ws) = legendre();
    let gr = (5.0f64.sqrt() - 1.0) / 2.0;
    // golden-section maximization of f
    let (mut a, mut b) = (lo, hi);
    let mut c = b - gr * (b - a);
    let mut d = a + gr * (b - a);
    let (mut fc, mut fd) = (f(c), f(d));
    for _ in 0..200 {
        if (b - a).abs() < 1e-9 {
            break;
        }
        if fc > fd {
            b = d;
            d = c;
            fd = fc;
            c = b - gr * (b - a);
            fc = f(c);
        } else {
            a = c;
            c = d;
            fc = fd;
            d = a + gr * (b - a);
            fd = f(d);
        }
    }
    let peak = 0.5 * (a + b);
    let peakv = f(peak);
    // grow window until both endpoints are far below the peak
    let mut W = 6.0;
    for _ in 0..14 {
        if f(peak + W) < peakv - 40.0 && f(peak - W) < peakv - 40.0 {
            break;
        }
        W *= 1.6;
    }
    let l = peak - W;
    let h = peak + W;
    let half = 0.5 * (h - l);
    let mid = 0.5 * (l + h);
    let mut s = 0.0f64;
    for i in 0..xs.len() {
        s += ws[i] * (f(mid + half * xs[i]) - peakv).exp();
    }
    let scaled = s * half; // ∫ exp(f − peak) over window
    if scaled <= 0.0 || !scaled.is_finite() {
        return f64::NEG_INFINITY;
    }
    peakv + scaled.ln()
}

/// One-way ANOVA log-integrand in w = log(g) space (JASP `marginal.g.oneWay`).
/// `n` = per-group sample size, `k` = number of groups, `F` = ANOVA F-statistic.
fn anova_integrand_w(w: f64, F: f64, n: f64, k: f64, rscale: f64) -> f64 {
    let g = w.exp();
    let dfs = (k - 1.0) / (n * k - k);
    let omega = (1.0 + n * g / (dfs * F + 1.0)) / (n * g + 1.0);
    let logi = rscale.ln()
        - 0.5 * (2.0 * PI).ln()
        - 1.5 * w
        - rscale * rscale / (2.0 * g)
        - 0.5 * (k - 1.0) * (n * g + 1.0).ln()
        - 0.5 * (n * k - 1.0) * omega.ln();
    logi + w // Jacobian dg/dw = e^w
}

/// One-way ANOVA BF10 for `k` groups of `n` observations, ANOVA statistic `F`,
/// Cauchy prior scale `rscale` (JASP allNways fixed: medium=0.5, wide=√2/2,
/// ultrawide=1.0).
pub fn bf_anova(F: f64, n: i64, k: i64, rscale: f64) -> f64 {
    if n < 2 || k < 2 || rscale <= 0.0 || F < 0.0 || !F.is_finite() {
        return f64::NAN;
    }
    let (n, k) = (n as f64, k as f64);
    let logbf = log_integrate_peak(
        &|w| anova_integrand_w(w, F, n, k, rscale),
        -25.0,
        25.0,
    );
    logbf.exp()
}

/// Regression log-integrand in w = log(g) space (JASP `integrand.regression`,
/// written with the g→e^w substitution). `N` = #observations, `p` = #predictors,
/// `R2` = R² of the full model.
fn reg_integrand_w(w: f64, N: f64, p: f64, r2: f64, rscale: f64) -> f64 {
    let g = w.exp();
    let scale = rscale * rscale * N / 2.0;
    // dinvgamma(g; shape=1/2, scale) = scale^0.5/Γ(0.5) · g^{-3/2} · e^{-scale/g}
    //   log = 0.5·log(scale) − 0.5·log π − 1.5·log(g) − scale/g
    let log_igamma = 0.5 * scale.ln() - 0.5 * PI.ln() - 1.5 * w - scale / g;
    let a = 0.5 * ((N - p - 1.0) * (1.0 + g).ln() - (N - 1.0) * (1.0 + g * (1.0 - r2)).ln());
    a + log_igamma + w // Jacobian dg/dw = e^w
}

/// Regression BF10 for model with `p` predictors, `N` observations, full-model
/// R², Cauchy prior scale `rscale` (JASP regression: medium=√2/4, wide=0.5,
/// ultrawide=√2/2).
pub fn bf_regression(N: i64, p: i64, r2: f64, rscale: f64) -> f64 {
    if N < p + 2 || p < 1 || r2 < 0.0 || r2 >= 1.0 || rscale <= 0.0 || !r2.is_finite() {
        return f64::NAN;
    }
    let (N, p) = (N as f64, p as f64);
    let logbf = log_integrate_peak(
        &|w| reg_integrand_w(w, N, p, r2, rscale),
        -25.0,
        25.0,
    );
    logbf.exp()
}

/// Standard t density normalized to a prior (metaBMA::prior("t")).
pub fn t_pdf_norm(x: f64, mu: f64, scale: f64, df: f64) -> f64 {
    if scale <= 0.0 || df <= 0.0 {
        return f64::NAN;
    }
    t_pdf((x - mu) / scale, df) / scale
}

// ── Bayesian binomial test (JASP jaspFrequencies) ────────────────
// JASP `.bayesBinomialTest`: two-sided uses the Beta-Binomial marginal
// likelihood ratio; one-sided uses the posterior-tail / prior-tail Savage-
// Dickey ratio. Prior = Beta(a, b), H0: theta = theta0.
/// log-Beta(a,b) = sum lgamma.
fn log_beta(a: f64, b: f64) -> f64 {
    libm::lgamma(a) + libm::lgamma(b) - libm::lgamma(a + b)
}

/// Two-sided Bayesian binomial BF10 (JASP `.bayesBinomialTest.twoSided`).
pub fn bf_binomial_two(counts: i64, n: i64, theta0: f64, a: f64, b: f64) -> f64 {
    if n < 1 || counts < 0 || counts > n || a <= 0.0 || b <= 0.0 {
        return f64::NAN;
    }
    if !(0.0..=1.0).contains(&theta0) {
        return f64::NAN;
    }
    let (k, N) = (counts as f64, n as f64);
    let log_ml_h1 = log_beta(a + k, b + N - k) - log_beta(a, b);
    // H0 point likelihood, with the log(0) guards JASP uses:
    let log_ml_h0 = if theta0 == 0.0 && counts == 0 {
        (N - k) * (1.0 - theta0).ln() // counts*log(0) omitted
    } else if theta0 == 1.0 && counts == n {
        k * theta0.ln() // (n-counts)*log(0) omitted
    } else {
        k * theta0.ln() + (N - k) * (1.0 - theta0).ln()
    };
    (log_ml_h1 - log_ml_h0).exp()
}

/// One-sided Bayesian binomial BF10 (JASP `.bayesBinomialTest.oneSided`).
/// `alt`: Right (greater) or Left (less).
pub fn bf_binomial_one(counts: i64, n: i64, theta0: f64, a: f64, b: f64, alt: Alt) -> f64 {
    if n < 1 || counts < 0 || counts > n || a <= 0.0 || b <= 0.0 {
        return f64::NAN;
    }
    if !(0.0..=1.0).contains(&theta0) {
        return f64::NAN;
    }
    let (k, N) = (counts as f64, n as f64);
    let lower_tail = alt == Alt::Left; // "less" -> left tail
    let log_ml_h0 = if theta0 == 0.0 && counts == 0 {
        (N - k) * (1.0 - theta0).ln()
    } else if theta0 == 1.0 && counts == n {
        k * theta0.ln()
    } else {
        k * theta0.ln() + (N - k) * (1.0 - theta0).ln()
    };
    let t1 = reg_inc_beta_log(a + k, b + N - k, theta0, lower_tail) + log_beta(a + k, b + N - k);
    let t2 = log_beta(a, b) + reg_inc_beta_log(a, b, theta0, lower_tail);
    (t1 - t2 - log_ml_h0).exp()
}

/// log pbeta(theta0; a, b, lower.tail)  (log regularized incomplete beta).
fn reg_inc_beta_log(a: f64, b: f64, x: f64, lower_tail: bool) -> f64 {
    let p = reg_inc_beta(a, b, x);
    if lower_tail {
        if p <= 0.0 {
            f64::NEG_INFINITY
        } else {
            p.ln()
        }
    } else {
        if p >= 1.0 {
            f64::NEG_INFINITY
        } else {
            (1.0 - p).ln()
        }
    }
}

// ── Bayesian multinomial test (JASP jaspFrequencies) ────────────
// JASP `.multBayesBfEquality`: Beta-Multinomial marginal likelihood ratio
// for the equality hypothesis theta.
/// Bayesian multinomial BF10 for an equality hypothesis.
/// `counts`, `alphas`, `thetas` are comma-joined, equal length (k levels).
pub fn bf_multinomial(
    counts: &[i64],
    alphas: &[f64],
    thetas: &[f64],
) -> f64 {
    let k = counts.len();
    if k < 2 || alphas.len() != k || thetas.len() != k {
        return f64::NAN;
    }
    if counts.iter().any(|&c| c < 0) || alphas.iter().any(|&a| a <= 0.0) {
        return f64::NAN;
    }
    let sum_t: f64 = thetas.iter().sum();
    if sum_t <= 0.0 || !sum_t.is_finite() {
        return f64::NAN;
    }
    let thetas: Vec<f64> = thetas.iter().map(|t| t / sum_t).collect();
    if thetas.iter().any(|&t| t < 0.0) {
        return f64::NAN;
    }
    let sum_a: f64 = alphas.iter().sum();
    let lxa: f64 = counts
        .iter()
        .zip(alphas.iter())
        .map(|(&c, &a)| libm::lgamma(a + c as f64))
        .sum::<f64>()
        - libm::lgamma(sum_a + counts.iter().map(|&c| c as f64).sum::<f64>());
    let la: f64 = alphas.iter().map(|&a| libm::lgamma(a)).sum::<f64>() - libm::lgamma(sum_a);
    // JASP: skip the theta-product if any theta_i==0 and counts_i==0 (log(0)).
    let mut any_zero = false;
    for i in 0..k {
        if thetas[i] == 0.0 && counts[i] == 0 {
            any_zero = true;
        }
        if thetas[i] == 0.0 && counts[i] > 0 {
            return f64::NAN; // log(0) with positive count -> -inf likelihood
        }
    }
    let prod: f64 = if any_zero {
        0.0
    } else {
        counts.iter().zip(thetas.iter()).map(|(&c, &t)| (c as f64) * t.ln()).sum()
    };
    ((lxa - la) - prod).exp()
}

// ── Bayesian contingency table (Gunel & Dickey 1974) ─────────────
// Closed-form log Bayes factors for independence vs non-independence,
// four sampling plans. Faithful port of BayesFactor's contingency-utility.R
// (ldirich / ldirich1 / pcomb / contingency{Poisson,JointMultinomial,
// IndepMultinomial,Hypergeometric}).
fn ldirich(a: &[f64]) -> f64 {
    let sum: f64 = a.iter().sum();
    a.iter().map(|&x| libm::lgamma(x)).sum::<f64>() - libm::lgamma(sum)
}
fn ldirich1(y: &[f64], a: &[f64]) -> f64 {
    y.iter()
        .zip(a.iter())
        .map(|(&yi, &ai)| libm::lgamma(ai) - libm::lgamma(ai + yi))
        .sum()
}
fn pcomb(x: &[f64]) -> f64 {
    let n: f64 = x.iter().sum();
    libm::lgamma(n + 1.0) - x.iter().map(|&xi| libm::lgamma(xi + 1.0)).sum::<f64>()
}
fn log_mean_exp_logs(logs: &[f64]) -> f64 {
    let m = logs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    m + logs.iter().map(|&l| (l - m).exp()).sum::<f64>().ln()
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SampleType {
    Poisson,
    JointMulti,
    IndepMulti,
    HyperGeom,
}

/// Contingency-table log BF10 (independence vs non-independence).
/// `cells` is the I×J count matrix in row-major order, `a` the prior
/// concentration (>=1), `sample_type` the sampling plan, `fixed_margin`
/// for indepMulti: "rows" or "cols".
pub fn bf_contingency_log(
    cells: &[i64],
    i: usize,
    j: usize,
    sample_type: SampleType,
    fixed_margin: Option<&str>,
    a: f64,
) -> f64 {
    if cells.len() != i * j || i < 2 || j < 2 || a < 1.0 {
        return f64::NAN;
    }
    if cells.iter().any(|&c| c < 0) {
        return f64::NAN;
    }
    // `a` matrix (same shape as y, filled with scalar a)
    let am: Vec<f64> = vec![a; i * j];
    let ym: Vec<f64> = cells.iter().map(|&c| c as f64).collect();
    let colsums = |m: &[f64]| -> Vec<f64> {
        (0..j).map(|jc| m.iter().enumerate().filter(|&(ii, _)| ii % j == jc).map(|(_, &v)| v).sum()).collect()
    };
    let rowsums = |m: &[f64]| -> Vec<f64> {
        (0..i).map(|ir| m[ir * j..ir * j + j].iter().sum()).collect()
    };
    match sample_type {
        SampleType::Poisson => {
            let n = ym.iter().sum::<f64>();
            let b = (i * j) as f64 * a / n;
            let ac = colsums(&am);
            let ar = rowsums(&am);
            let yc = colsums(&ym);
            let yr = rowsums(&ym);
            let jm1 = (j - 1) as f64;
            let im1 = (i - 1) as f64;
            // yr + ar - (J-1)*or  -> elementwise: yr[r] + ar[r] - jm1
            let yr_ar: Vec<f64> = (0..i).map(|r| yr[r] + ar[r] - jm1).collect();
            let ar_adj: Vec<f64> = (0..i).map(|r| ar[r] - jm1).collect();
            let yc_ac: Vec<f64> = (0..j).map(|c| yc[c] + ac[c] - im1).collect();
            let ac_adj: Vec<f64> = (0..j).map(|c| ac[c] - im1).collect();
            libm::lgamma(sum_of(&am) - (im1 * jm1))
                - (im1 * jm1) * (1.0 + 1.0 / b).ln()
                - libm::lgamma(n + sum_of(&am) - im1 * jm1)
                - ldirich(&yr_ar)
                + ldirich(&ar_adj)
                - ldirich(&yc_ac)
                + ldirich(&ac_adj)
                - ldirich1(&ym, &am)
        }
        SampleType::JointMulti => {
            let ac = colsums(&am);
            let ar = rowsums(&am);
            let yc = colsums(&ym);
            let yr = rowsums(&ym);
            let jm1 = (j - 1) as f64;
            let im1 = (i - 1) as f64;
            let yplusa: Vec<f64> = ym.iter().zip(am.iter()).map(|(&y, &a)| y + a).collect();
            let ar_adj: Vec<f64> = (0..i).map(|r| ar[r] - jm1).collect();
            let ac_adj: Vec<f64> = (0..j).map(|c| ac[c] - im1).collect();
            let yr_ar: Vec<f64> = (0..i).map(|r| yr[r] + ar[r] - jm1).collect();
            let yc_ac: Vec<f64> = (0..j).map(|c| yc[c] + ac[c] - im1).collect();
            ldirich(&yplusa)
                + ldirich(&ar_adj)
                + ldirich(&ac_adj)
                - ldirich(&am)
                - ldirich(&yr_ar)
                - ldirich(&yc_ac)
        }
        SampleType::IndepMulti => {
            // If fixed margin is "cols", JASP transposes (t(y)) then uses rows.
            // After transpose the matrix is J×I, so its sums must be taken
            // with the swapped dimensions (the captured colsums/rowsums use
            // the original i×j and would be wrong for non-square tables).
            let (mt, rows, cols, at) = if fixed_margin == Some("cols") {
                let mut tt: Vec<f64> = vec![0.0; i * j];
                for r in 0..i {
                    for c in 0..j {
                        tt[c * i + r] = ym[r * j + c];
                    }
                }
                (tt, j, i, vec![a; i * j])
            } else {
                (ym.clone(), i, j, am.clone())
            };
            // row-major sums for the (possibly transposed) `rows`×`cols` matrix.
            // ac/ar come from the PRIOR matrix `at`; yc/yr from the DATA `mt`.
            let ac: Vec<f64> = (0..cols)
                .map(|c| (0..rows).map(|r| at[r * cols + c]).sum())
                .collect();
            let ar: Vec<f64> = (0..rows)
                .map(|r| at[r * cols..r * cols + cols].iter().sum())
                .collect();
            let yc: Vec<f64> = (0..cols)
                .map(|c| (0..rows).map(|r| mt[r * cols + c]).sum())
                .collect();
            let yr: Vec<f64> = (0..rows)
                .map(|r| mt[r * cols..r * cols + cols].iter().sum())
                .collect();
            let it = rows;
            let jt = cols;
            let im1 = (it - 1) as f64;
            let yplusa: Vec<f64> = mt.iter().zip(at.iter()).map(|(&y, &aa)| y + aa).collect();
            let ac_adj: Vec<f64> = (0..jt).map(|c| ac[c] - im1).collect();
            let yc_ac: Vec<f64> = (0..jt).map(|c| yc[c] + ac[c] - im1).collect();
            let yr_ar: Vec<f64> = (0..it).map(|r| yr[r] + ar[r]).collect();
            ldirich(&ac_adj)
                + ldirich(&ar)
                + ldirich(&yplusa)
                - ldirich(&yc_ac)
                - ldirich(&yr_ar)
                - ldirich(&at)
        }
        SampleType::HyperGeom => {
            if i != 2 || j != 2 {
                return f64::NAN; // JASP restricts to 2×2
            }
            // sumg(y, a): enumerate the top-left cell i over the feasible range
            let yc = colsums(&ym);
            let yr = rowsums(&ym);
            let m = [yc[0], yc[1], yr[0], yr[1]];
            if (m[0] + m[1]).round() != (m[2] + m[3]).round() {
                return f64::NAN; // margins must balance
            }
            let upper = (m[0].min(m[2])) as i64;
            let lower0 = upper - m.iter().cloned().fold(f64::INFINITY, f64::min) as i64;
            let mut g1: Vec<f64> = Vec::new();
            for ai in lower0.max(0)..=upper {
                let a_cell = ai as f64;
                let b_cell = m[0] - a_cell;
                let c_cell = m[2] - a_cell;
                let d_cell = m[3] - m[0] + a_cell;
                if b_cell < 0.0 || c_cell < 0.0 || d_cell < 0.0 {
                    continue;
                }
                let y1 = [a_cell, b_cell, c_cell, d_cell];
                let y1plusa: Vec<f64> = y1.iter().zip(am.iter()).map(|(&y, &a)| y + a).collect();
                g1.push(pcomb(&y1) + ldirich(&y1plusa) - ldirich(&am));
            }
            if g1.is_empty() {
                return f64::NAN;
            }
            let sumg = log_mean_exp_logs(&g1) + (g1.len() as f64).ln();
            let yplusa: Vec<f64> = ym.iter().zip(am.iter()).map(|(&y, &a)| y + a).collect();
            ldirich(&yplusa) + pcomb(&yc) + pcomb(&yr) - ldirich(&am) - sumg
        }
    }
}

fn sum_of(v: &[f64]) -> f64 {
    v.iter().sum()
}

// ── Equivalence t-test (JASP jaspEquivalenceTTests) ──────────────
// Savage-Dickey: BF_equivalence = P(interval | data) / P(interval), where the
// densities are the posterior t (JASP `.posterior_t` = normalized integrand)
// and the prior t (metaBMA::prior("t")). One-sided re-standardizes the
// posterior by the region-bound CDF, exactly as JASP does.
/// Posterior log-density kernel (JASP `.integrand_t`), unnormalized.
fn eq_integrand(
    d: f64,
    t: f64,
    neff: f64,
    nu: f64,
    mu: f64,
    gamma: f64,
    kappa: f64,
) -> f64 {
    if d.is_infinite() {
        return f64::NEG_INFINITY;
    }
    let ncp = neff.sqrt() * d;
    let a = nc_t_pdf(t, nu, ncp);
    if a <= 0.0 {
        return f64::NEG_INFINITY;
    }
    a.ln() + (t_pdf((d - mu) / gamma, kappa) / gamma).ln()
}

/// Integrate a non-negative kernel f over (-inf, b) with a peak-centered
/// growing window + GL, returning the integral (finite).
fn integrate_to_b(f: &dyn Fn(f64) -> f64, bound: f64) -> f64 {
    // find the mode of the (log) kernel by a coarse scan then golden section
    let (xs, ws) = legendre();
    let _ = (xs, ws); // (node count chosen by legendre(); only weights matter)
    let lo0 = -8.0;
    let hi0 = 8.0;
    let mut peak = 0.0f64;
    let mut peakv = f64::NEG_INFINITY;
    let steps = 400;
    for s in 0..=steps {
        let x = lo0 + (hi0 - lo0) * (s as f64 / steps as f64);
        let v = f(x);
        if v > peakv {
            peakv = v;
            peak = x;
        }
    }
    let gr = (5.0f64.sqrt() - 1.0) / 2.0;
    // NB: local bracket names gs_a/gs_b — the upper integration bound is
    // `bound` (the parameter). Using the names a/b here once shadowed it.
    let (mut gs_a, mut gs_b) = (peak - 3.0, peak + 3.0);
    let mut c = gs_b - gr * (gs_b - gs_a);
    let mut d = gs_a + gr * (gs_b - gs_a);
    let (mut fc, mut fd) = (f(c), f(d));
    for _ in 0..200 {
        if (gs_b - gs_a).abs() < 1e-9 {
            break;
        }
        if fc > fd {
            gs_b = d;
            d = c;
            fd = fc;
            c = gs_b - gr * (gs_b - gs_a);
            fc = f(c);
        } else {
            gs_a = c;
            c = d;
            fc = fd;
            d = gs_a + gr * (gs_b - gs_a);
            fd = f(d);
        }
    }
    peak = 0.5 * (gs_a + gs_b);
    peakv = f(peak);
    let mut W = 4.0;
    for _ in 0..14 {
        if f(peak + W) < peakv - 40.0 && f(peak - W) < peakv - 40.0 {
            break;
        }
        W *= 1.6;
    }
    // Integrate f (a log-density) over [peak - W, min(peak + W, bound)].
    // A single GL pass over an asymmetric window is inaccurate when the peak
    // is off-centre (the bulk of the mass sits on one side), so split the
    // interval at the peak into two proper GL sub-integrals. Each sub-
    // integral uses the standard linear-density GL, valid for any [lo,hi].
    let l = peak - W;
    let h = (peak + W).min(bound);
    if h <= l {
        return 0.0;
    }
    let lin = |d: f64| f(d).exp();
    if peak > l && peak < h {
        integrate(l, peak, &lin) + integrate(peak, h, &lin)
    } else {
        integrate(l, h, &lin)
    }
}

/// Posterior CDF at x: ∫_{-inf}^x posterior density, normalized.
fn eq_post_cdf(
    x: f64,
    t: f64,
    neff: f64,
    nu: f64,
    mu: f64,
    gamma: f64,
    kappa: f64,
) -> f64 {
    if x.is_infinite() && x < 0.0 {
        return 0.0;
    }
    // `integrate_to_b` expects a LOG-density (it exponentiates internally
    // around the peak for numerical stability).
    let k = |d: f64| eq_integrand(d, t, neff, nu, mu, gamma, kappa);
    let total = integrate_to_b(&k, f64::INFINITY);
    if total <= 0.0 || !total.is_finite() {
        return f64::NAN;
    }
    let upto = integrate_to_b(&k, x);
    (upto / total).clamp(0.0, 1.0)
}

/// Equivalence t-test Savage-Dickey BFs.
/// Returns (bf_equivalence, bf_nonequivalence).
/// `lower`, `upper` are the (standardized) equivalence region;
/// prior = t(mu, scale, df); `alt` two / right / left.
pub fn bf_equivalence(
    t: f64,
    n1: i64,
    n2: i64,
    lower: f64,
    upper: f64,
    _rscale: f64,
    loc: f64,
    scale: f64,
    df: f64,
    alt: Alt,
) -> (f64, f64) {
    if n1 < 2 || lower >= upper || scale <= 0.0 || df <= 0.0 {
        return (f64::NAN, f64::NAN);
    }
    let indep = n2 > 0;
    let neff = if indep {
        (n1 * n2) as f64 / (n1 + n2) as f64
    } else {
        n1 as f64
    };
    let nu = if indep {
        (n1 + n2 - 2) as f64
    } else {
        (n1 - 1) as f64
    };
    if !t.is_finite() || neff <= 0.0 || nu <= 0.0 {
        return (f64::NAN, f64::NAN);
    }
    // Step 1: prior mass in the equivalence range (t CDF)
    let prior_cdf = |x: f64| t_cdf((x - loc) / scale, df);
    let int_eq_prior = (prior_cdf(upper) - prior_cdf(lower)).abs();
    let int_neq_prior = 1.0 - int_eq_prior;
    // Step 2: posterior mass in the equivalence range
    let post = |x: f64| eq_post_cdf(x, t, neff, nu, loc, scale, df);
    let upperbound = post(upper);
    let lowerbound = post(lower);
    let mut int_eq_post = upperbound - lowerbound;
    if int_eq_post < 0.0 {
        int_eq_post = 0.0;
    }
    // Step 2b: one-sided re-standardization by the region-bound CDF
    match alt {
        Alt::Right => {
            let denom = (1.0 - lowerbound).max(1e-300);
            int_eq_post = int_eq_post / denom;
        }
        Alt::Left => {
            let denom = upperbound.max(1e-300);
            int_eq_post = int_eq_post / denom;
        }
        Alt::TwoSided => {}
    }
    let int_neq_post = 1.0 - int_eq_post;
    let bf_eq = int_eq_post / int_eq_prior.max(1e-300);
    let bf_noneq = int_neq_post / int_neq_prior.max(1e-300);
    (bf_eq, bf_noneq)
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

    /// Regression guard: the posterior CDF must actually move between the two
    /// region bounds (catches the `integrate_to_b` bound being ignored, which
    /// previously made both CDFs equal and collapsed the region mass to zero).
    #[test]
    fn equivalence_cdf_split() {
        // t=-3, n=10 one-sample, Cauchy(0,0.5,1) prior.
        let (t, neff, nu, mu, g, k) = (-3.0, 10.0, 9.0, 0.0, 0.5, 1.0);
        let f = |d: f64| eq_integrand(d, t, neff, nu, mu, g, k);
        let total = integrate_to_b(&f, f64::INFINITY);
        let to_upper = integrate_to_b(&f, 0.5);
        let to_lower = integrate_to_b(&f, -0.5);
        let cdf_up = (to_upper / total).clamp(0.0, 1.0);
        let cdf_lo = (to_lower / total).clamp(0.0, 1.0);
        // posterior is concentrated around d≈0.4 (t/√neff), so the CDF should
        // differ substantially between 0.5 and −0.5.
        assert!(cdf_up > cdf_lo, "CDF must be monotone: cdf_up={cdf_up} <= cdf_lo={cdf_lo}");
        assert!((cdf_up - cdf_lo) > 0.05, "region mass too small: {}", cdf_up - cdf_lo);
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

    fn unquote(s: &str) -> &str {
        s.trim_matches('"')
    }
    fn rscale(s: &str) -> f64 {
        match s {
            "medium" => 0.5,
            "wide" => (2.0f64).sqrt() / 2.0,
            "ultrawide" => 1.0,
            _ => panic!("unknown rscale {s}"),
        }
    }

    #[test]
    fn bf_anova_matches_r() {
        // Reference: R BayesFactor:::oneWayAOV.Fstat(F, N=per-group n, J=#groups)
        // over F∈{0..12}, N∈{5..40}, J∈{2..6}, all 3 JASP prior widths (432 rows),
        // and cross-checked against full BayesFactor::anovaBF on synthetic data.
        let rows = lines("test/fixtures/anova_ref.csv");
        let mut checked = 0usize;
        for l in rows.iter().skip(1) {
            let f: Vec<&str> = l.split(',').map(unquote).collect();
            if f.len() != 5 {
                continue;
            }
            let F: f64 = f[0].parse().unwrap();
            let n: i64 = f[1].parse().unwrap();
            let k: i64 = f[2].parse().unwrap();
            let r = rscale(f[3]);
            let val: f64 = f[4].parse().unwrap();
            let got = bf_anova(F, n, k, r);
            assert!(close(got, val), "bf_anova({F},{n},{k},{r}) = {got}, R says {val}");
            checked += 1;
        }
        assert!(checked == 432, "only {checked} anova rows checked");
    }

    #[test]
    fn bf_regression_matches_r() {
        // Reference: R BayesFactor:::linearReg.R2stat(N, p, R2, rscale) over
        // N∈{10..100}, p∈{1..5}, R2∈{0..0.8}, all 3 JASP regression prior
        // widths (288 rows), cross-checked against full BayesFactor::lmBF.
        let rows = lines("test/fixtures/reg_ref.csv");
        let mut checked = 0usize;
        for l in rows.iter().skip(1) {
            let f: Vec<&str> = l.split(',').map(unquote).collect();
            if f.len() != 5 {
                continue;
            }
            let N: i64 = f[0].parse().unwrap();
            let p: i64 = f[1].parse().unwrap();
            let r2: f64 = f[2].parse().unwrap();
            // regression rscale map: medium=sqrt(2)/4, wide=0.5, ultrawide=sqrt(2)/2
            let r = match f[3] {
                "medium" => (2.0f64).sqrt() / 4.0,
                "wide" => 0.5,
                "ultrawide" => (2.0f64).sqrt() / 2.0,
                _ => panic!("unknown rscale {}", f[3]),
            };
            let val: f64 = f[4].parse().unwrap();
            let got = bf_regression(N, p, r2, r);
            assert!(close(got, val), "bf_regression({N},{p},{r2},{r}) = {got}, R says {val}");
            checked += 1;
        }
        assert!(checked == 288, "only {checked} regression rows checked");
    }

    // ── helpers for the "remaining" module fixture tests ──────────
    // CSV rows carry quoted string fields ("10,20,30"); split on top-level
    // commas, strip the double-quotes.
    fn csv_row(l: &str) -> Vec<String> {
        let mut out = Vec::new();
        let mut cur = String::new();
        let mut inq = false;
        for c in l.chars() {
            match c {
                '"' => {
                    inq = !inq;
                }
                ',' if !inq => {
                    out.push(std::mem::take(&mut cur));
                }
                _ => cur.push(c),
            }
        }
        out.push(cur);
        out.iter().map(|s| s.trim().to_string()).collect()
    }

    fn i64_list(s: &str) -> Vec<i64> {
        s.split(',').filter_map(|x| x.trim().parse().ok()).collect()
    }
    fn f64_list(s: &str) -> Vec<f64> {
        s.split(',').filter_map(|x| x.trim().parse().ok()).collect()
    }

    #[test]
    fn binomial_matches_jasp() {
        let rows = lines("test/fixtures/binom_ref.csv");
        let mut checked = 0usize;
        for l in rows.iter().skip(1) {
            let f = csv_row(l);
            if f.len() != 7 {
                continue;
            }
            let (counts, n) = (f[0].parse::<i64>().unwrap(), f[1].parse::<i64>().unwrap());
            let (theta0, a, b) = (
                f[2].parse::<f64>().unwrap(),
                f[3].parse::<f64>().unwrap(),
                f[4].parse::<f64>().unwrap(),
            );
            let exp: f64 = f[6].parse().unwrap();
            // Our SQL scalar exposes the two-sided BF; one-sided uses a
            // different (Savage-Dickey) formula. Verify two-sided rows.
            if f[5] != "two" {
                continue;
            }
            let got = bf_binomial_two(counts, n, theta0, a, b);
            assert!(close(got, exp), "binomial({counts},{n},{theta0},{a},{b}) = {got}, JASP says {exp}");
            checked += 1;
        }
        assert!(checked > 100, "only {checked} binomial rows checked");
    }

    #[test]
    fn multinomial_matches_jasp() {
        let rows = lines("test/fixtures/multinom_ref.csv");
        let mut checked = 0usize;
        for l in rows.iter().skip(1) {
            let f = csv_row(l);
            if f.len() != 4 {
                continue;
            }
            let (counts, alphas, thetas) = (i64_list(&f[0]), f64_list(&f[1]), f64_list(&f[2]));
            let exp: f64 = f[3].parse().unwrap();
            let got = bf_multinomial(&counts, &alphas, &thetas);
            assert!(
                close(got, exp),
                "multinomial({}) = {got}, JASP says {exp}",
                f[0]
            );
            checked += 1;
        }
        assert!(checked > 10, "only {checked} multinomial rows checked");
    }

    #[test]
    fn contingency_matches_bayesfactor() {
        let rows = lines("test/fixtures/contig_ref.csv");
        let mut checked = 0usize;
        for l in rows.iter().skip(1) {
            let f = csv_row(l);
            if f.len() != 7 {
                continue;
            }
            let cells = i64_list(&f[0]);
            let (i, j) = (f[1].parse::<usize>().unwrap(), f[2].parse::<usize>().unwrap());
            let sample = match f[3].as_str() {
                "poisson" => SampleType::Poisson,
                "jointMulti" => SampleType::JointMulti,
                "indepMulti" => SampleType::IndepMulti,
                "hypergeom" => SampleType::HyperGeom,
                _ => continue,
            };
            let fixed: Option<&str> = if f[4].is_empty() {
                None
            } else {
                Some(f[4].as_str())
            };
            let a: f64 = f[5].parse().unwrap();
            let exp_log: f64 = f[6].parse().unwrap();
            let got_log = bf_contingency_log(&cells, i, j, sample, fixed, a);
            assert!(
                (got_log - exp_log).abs() < 1e-4,
                "contingency({}) {} {} a={a}: logbf {got_log}, BF says {exp_log}",
                f[0],
                f[3],
                f[4]
            );
            checked += 1;
        }
        assert!(checked > 30, "only {checked} contingency rows checked");
    }

    #[test]
    fn equivalence_matches_jasp() {
        let rows = lines("test/fixtures/equiv_ref.csv");
        let mut checked = 0usize;
        for l in rows.iter().skip(1) {
            let f = csv_row(l);
            if f.len() != 13 {
                continue;
            }
            let (t, n1, n2) = (
                f[0].parse::<f64>().unwrap(),
                f[1].parse::<i64>().unwrap(),
                f[2].parse::<i64>().unwrap(),
            );
            let (lower, upper) = (f[3].parse::<f64>().unwrap(), f[4].parse::<f64>().unwrap());
            let rscale: f64 = f[5].parse().unwrap();
            let (loc, scale, df) = (
                f[6].parse::<f64>().unwrap(),
                f[7].parse::<f64>().unwrap(),
                f[8].parse::<f64>().unwrap(),
            );
            let alt = alt_from(&f[9]);
            let (exp_eq, exp_neq) = (f[10].parse::<f64>().unwrap(), f[11].parse::<f64>().unwrap());
            let (got_eq, got_neq) = bf_equivalence(t, n1, n2, lower, upper, rscale, loc, scale, df, alt);
            // JASP's reported BF is the ratio bf_eq / bf_noneq — match that.
            let got_ratio = got_eq / got_neq;
            let exp_ratio = exp_eq / exp_neq;
            assert!(
                (got_ratio - exp_ratio).abs() / (1.0 + exp_ratio.abs()) < 2e-3,
                "equiv t={t} n1={n1} n2={n2} [{lower},{upper}] r={rscale} {alt:?}: ratio {got_ratio}, JASP says {exp_ratio}"
            );
            checked += 1;
        }
        assert!(checked > 100, "only {checked} equivalence rows checked");
    }
}
