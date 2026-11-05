//! duckdb_bayes — BayesFactor t-test BF10 as DuckDB scalar functions.
//!
//! Functions (see README.md):
//!   bf_ttest(t, n, r, alt)       → BF10, one-sample / paired (DOUBLE)
//!   bf_ttest2(t, ny, nx, r, alt) → BF10, independent two-sample (DOUBLE)
//!
//!   t    observed t-statistic
//!   n    one-sample size      (two-sample: ny, nx group sizes)
//!   r    JZS Cauchy prior scale (JASP default 0.707)
//!   alt  'two' | 'right' | 'left'
//!
//! All math lives in `src/math.rs` — a faithful port of JASP's
//! `jaspTTests/R/informedbayesianttestfunctions.R` (`.bf10_t` + `.dtss`),
//! asserted by `cargo test` against JASP R-generated fixtures. This file is
//! only the C-API glue: read input vectors, call the math, write results.

pub mod math;

use quack_rs::{
    entry_point::init_extension_v2,
    prelude::*,
    table::TableFunctionBuilder,
    vector::{VectorReader, VectorWriter},
};
use libduckdb_sys::{
    duckdb_data_chunk, duckdb_data_chunk_get_size, duckdb_data_chunk_get_vector,
    duckdb_data_chunk_set_size, duckdb_extension_access, duckdb_extension_info,
    duckdb_function_info, duckdb_vector_get_data, duckdb_vector_get_validity,
    duckdb_vector, duckdb_validity_set_row_invalid, idx_t,
};

use math::{alt_from, bf_anova, bf_regression, bf_ttest, bf_ttest2, robustness_grid};

// ── scalar function callbacks ─────────────────────────────────────
// Signature: (info, input_chunk, output_vector). One call per data chunk.

unsafe extern "C" fn bf_ttest_fn(
    _info: duckdb_function_info,
    input: duckdb_data_chunk,
    output: duckdb_vector,
) {
    let n = usize::try_from(unsafe { duckdb_data_chunk_get_size(input) }).unwrap_or(0);
    let rt = unsafe { VectorReader::new(input, 0) }; // t    DOUBLE
    let rn = unsafe { VectorReader::new(input, 1) }; // n    BIGINT
    let rr = unsafe { VectorReader::new(input, 2) }; // r    DOUBLE
    let ra = unsafe { VectorReader::new(input, 3) }; // alt  VARCHAR
    let mut out = unsafe { VectorWriter::new(output) };
    for i in 0..n {
        if !unsafe { rt.is_valid(i) } || !unsafe { rn.is_valid(i) } || !unsafe { rr.is_valid(i) } || !unsafe { ra.is_valid(i) }
        {
            unsafe { out.set_null(i) };
            continue;
        }
        let t = unsafe { rt.read_f64(i) };
        let nobs = unsafe { rn.read_i64(i) };
        let r = unsafe { rr.read_f64(i) };
        let alt = alt_from(unsafe { ra.read_str(i) });
        let bf = bf_ttest(t, nobs, r, alt);
        if bf.is_finite() && bf > 0.0 {
            unsafe { out.write_f64(i, bf) };
        } else {
            unsafe { out.set_null(i) };
        }
    }
}

unsafe extern "C" fn bf_ttest2_fn(
    _info: duckdb_function_info,
    input: duckdb_data_chunk,
    output: duckdb_vector,
) {
    let n = usize::try_from(unsafe { duckdb_data_chunk_get_size(input) }).unwrap_or(0);
    let rt = unsafe { VectorReader::new(input, 0) }; // t    DOUBLE
    let rny = unsafe { VectorReader::new(input, 1) }; // ny   BIGINT
    let rnx = unsafe { VectorReader::new(input, 2) }; // nx   BIGINT
    let rr = unsafe { VectorReader::new(input, 3) }; // r    DOUBLE
    let ra = unsafe { VectorReader::new(input, 4) }; // alt  VARCHAR
    let mut out = unsafe { VectorWriter::new(output) };
    for i in 0..n {
        if !unsafe { rt.is_valid(i) } || !unsafe { rny.is_valid(i) } || !unsafe { rnx.is_valid(i) }
            || !unsafe { rr.is_valid(i) } || !unsafe { ra.is_valid(i) }
        {
            unsafe { out.set_null(i) };
            continue;
        }
        let t = unsafe { rt.read_f64(i) };
        let ny = unsafe { rny.read_i64(i) };
        let nx = unsafe { rnx.read_i64(i) };
        let r = unsafe { rr.read_f64(i) };
        let alt = alt_from(unsafe { ra.read_str(i) });
        let bf = bf_ttest2(t, ny, nx, r, alt);
        if bf.is_finite() && bf > 0.0 {
            unsafe { out.write_f64(i, bf) }
        } else {
            unsafe { out.set_null(i) }
        }
    }
}

/// JASP prior-width words → Cauchy scale. ANOVA (allNways fixed) and regression
/// use different maps, so each is a small match. Numeric strings pass through.
fn anova_rscale(s: &str) -> f64 {
    match s.trim().to_lowercase().as_str() {
        "medium" => 0.5,
        "wide" => (2.0f64).sqrt() / 2.0,
        "ultrawide" => 1.0,
        other => other.parse().unwrap_or(f64::NAN),
    }
}
fn reg_rscale(s: &str) -> f64 {
    match s.trim().to_lowercase().as_str() {
        "medium" => (2.0f64).sqrt() / 4.0,
        "wide" => 0.5,
        "ultrawide" => (2.0f64).sqrt() / 2.0,
        other => other.parse().unwrap_or(f64::NAN),
    }
}

unsafe extern "C" fn bf_anova_fn(
    _info: duckdb_function_info,
    input: duckdb_data_chunk,
    output: duckdb_vector,
) {
    let n = usize::try_from(unsafe { duckdb_data_chunk_get_size(input) }).unwrap_or(0);
    let rf = unsafe { VectorReader::new(input, 0) }; // F     DOUBLE
    let rn = unsafe { VectorReader::new(input, 1) }; // n     BIGINT
    let rk = unsafe { VectorReader::new(input, 2) }; // k     BIGINT
    let rp = unsafe { VectorReader::new(input, 3) }; // prior VARCHAR
    let mut out = unsafe { VectorWriter::new(output) };
    for i in 0..n {
        if !unsafe { rf.is_valid(i) } || !unsafe { rn.is_valid(i) }
            || !unsafe { rk.is_valid(i) } || !unsafe { rp.is_valid(i) }
        {
            unsafe { out.set_null(i) };
            continue;
        }
        let F = unsafe { rf.read_f64(i) };
        let nobs = unsafe { rn.read_i64(i) };
        let k = unsafe { rk.read_i64(i) };
        let r = anova_rscale(unsafe { rp.read_str(i) });
        let bf = bf_anova(F, nobs, k, r);
        if bf.is_finite() && bf > 0.0 {
            unsafe { out.write_f64(i, bf) }
        } else {
            unsafe { out.set_null(i) }
        }
    }
}

unsafe extern "C" fn bf_regression_fn(
    _info: duckdb_function_info,
    input: duckdb_data_chunk,
    output: duckdb_vector,
) {
    let n = usize::try_from(unsafe { duckdb_data_chunk_get_size(input) }).unwrap_or(0);
    let rn = unsafe { VectorReader::new(input, 0) }; // N     BIGINT
    let rp = unsafe { VectorReader::new(input, 1) }; // p     BIGINT
    let rr = unsafe { VectorReader::new(input, 2) }; // R2    DOUBLE
    let rq = unsafe { VectorReader::new(input, 3) }; // prior VARCHAR
    let mut out = unsafe { VectorWriter::new(output) };
    for i in 0..n {
        if !unsafe { rn.is_valid(i) } || !unsafe { rp.is_valid(i) }
            || !unsafe { rr.is_valid(i) } || !unsafe { rq.is_valid(i) }
        {
            unsafe { out.set_null(i) };
            continue;
        }
        let N = unsafe { rn.read_i64(i) };
        let p = unsafe { rp.read_i64(i) };
        let r2 = unsafe { rr.read_f64(i) };
        let r = reg_rscale(unsafe { rq.read_str(i) });
        let bf = bf_regression(N, p, r2, r);
        if bf.is_finite() && bf > 0.0 {
            unsafe { out.write_f64(i, bf) }
        } else {
            unsafe { out.set_null(i) }
        }
    }
}

// ── Bayesian binomial / multinomial / contingency / equivalence ──
// jaspFrequencies + jaspEquivalenceTTests. Math in math.rs, ported from JASP
// R + the BayesFactor 0.9.12 closed forms (Gunel-Dickey 1974 for contingency).

unsafe extern "C" fn bf_binomial_fn(
    _info: duckdb_function_info,
    input: duckdb_data_chunk,
    output: duckdb_vector,
) {
    let n = usize::try_from(unsafe { duckdb_data_chunk_get_size(input) }).unwrap_or(0);
    let rk = unsafe { VectorReader::new(input, 0) }; // k  successes BIGINT
    let rn = unsafe { VectorReader::new(input, 1) }; // n  trials     BIGINT
    let rt = unsafe { VectorReader::new(input, 2) }; // theta0         DOUBLE
    let ra = unsafe { VectorReader::new(input, 3) }; // priorA         DOUBLE
    let rb = unsafe { VectorReader::new(input, 4) }; // priorB         DOUBLE
    let mut out = unsafe { VectorWriter::new(output) };
    for i in 0..n {
        if !unsafe { rk.is_valid(i) } || !unsafe { rn.is_valid(i) } || !unsafe { rt.is_valid(i) }
            || !unsafe { ra.is_valid(i) } || !unsafe { rb.is_valid(i) }
        {
            unsafe { out.set_null(i) };
            continue;
        }
        let k = unsafe { rk.read_i64(i) };
        let nobs = unsafe { rn.read_i64(i) };
        let theta0 = unsafe { rt.read_f64(i) };
        let a = unsafe { ra.read_f64(i) };
        let b = unsafe { rb.read_f64(i) };
        let bf = math::bf_binomial_two(k, nobs, theta0, a, b);
        if bf.is_finite() && bf > 0.0 {
            unsafe { out.write_f64(i, bf) }
        } else {
            unsafe { out.set_null(i) }
        }
    }
}

unsafe extern "C" fn bf_multinomial_fn(
    _info: duckdb_function_info,
    input: duckdb_data_chunk,
    output: duckdb_vector,
) {
    let n = usize::try_from(unsafe { duckdb_data_chunk_get_size(input) }).unwrap_or(0);
    let rc = unsafe { VectorReader::new(input, 0) }; // counts  VARCHAR (comma-joined)
    let rp = unsafe { VectorReader::new(input, 1) }; // priors  VARCHAR
    let rt = unsafe { VectorReader::new(input, 2) }; // thetas  VARCHAR
    let mut out = unsafe { VectorWriter::new(output) };
    for i in 0..n {
        if !unsafe { rc.is_valid(i) } || !unsafe { rp.is_valid(i) } || !unsafe { rt.is_valid(i) } {
            unsafe { out.set_null(i) };
            continue;
        }
        let counts: Vec<i64> = parse_i64_list(unsafe { rc.read_str(i) });
        let priors: Vec<f64> = parse_f64_list(unsafe { rp.read_str(i) });
        let thetas: Vec<f64> = parse_f64_list(unsafe { rt.read_str(i) });
        let bf = math::bf_multinomial(&counts, &priors, &thetas);
        if bf.is_finite() && bf > 0.0 {
            unsafe { out.write_f64(i, bf) }
        } else {
            unsafe { out.set_null(i) }
        }
    }
}

unsafe extern "C" fn bf_contingency_fn(
    _info: duckdb_function_info,
    input: duckdb_data_chunk,
    output: duckdb_vector,
) {
    let n = usize::try_from(unsafe { duckdb_data_chunk_get_size(input) }).unwrap_or(0);
    let rc = unsafe { VectorReader::new(input, 0) }; // cells  VARCHAR (comma-joined)
    let rr = unsafe { VectorReader::new(input, 1) }; // rows   BIGINT
    let rk = unsafe { VectorReader::new(input, 2) }; // cols   BIGINT
    let rm = unsafe { VectorReader::new(input, 3) }; // sample VARCHAR
    let rl = unsafe { VectorReader::new(input, 4) }; // fixed  VARCHAR
    let ra = unsafe { VectorReader::new(input, 5) }; // a      DOUBLE
    let mut out = unsafe { VectorWriter::new(output) };
    for i in 0..n {
        if !unsafe { rc.is_valid(i) } || !unsafe { rr.is_valid(i) } || !unsafe { rk.is_valid(i) }
            || !unsafe { rm.is_valid(i) } || !unsafe { rl.is_valid(i) } || !unsafe { ra.is_valid(i) }
        {
            unsafe { out.set_null(i) };
            continue;
        }
        let cells: Vec<i64> = parse_i64_list(unsafe { rc.read_str(i) });
        let nrows = unsafe { rr.read_i64(i) };
        let ncols = unsafe { rk.read_i64(i) };
        let sample = sample_type_from(unsafe { rm.read_str(i) });
        let fixed_s = unsafe { rl.read_str(i) }.trim().to_lowercase();
        let fixed: Option<&str> = if fixed_s.is_empty() { None } else { Some(fixed_s.as_str()) };
        let a = unsafe { ra.read_f64(i) };
        let logbf = math::bf_contingency_log(&cells, nrows as usize, ncols as usize, sample, fixed, a);
        let bf = logbf.exp();
        if bf.is_finite() && bf > 0.0 {
            unsafe { out.write_f64(i, bf) }
        } else {
            unsafe { out.set_null(i) }
        }
    }
}

unsafe extern "C" fn bf_equivalence_fn(
    _info: duckdb_function_info,
    input: duckdb_data_chunk,
    output: duckdb_vector,
) {
    let n = usize::try_from(unsafe { duckdb_data_chunk_get_size(input) }).unwrap_or(0);
    let rt = unsafe { VectorReader::new(input, 0) }; // t     DOUBLE
    let rn1 = unsafe { VectorReader::new(input, 1) }; // n1    BIGINT
    let rn2 = unsafe { VectorReader::new(input, 2) }; // n2    BIGINT (0 = one-sample)
    let rl = unsafe { VectorReader::new(input, 3) }; // lower DOUBLE (std. units)
    let ru = unsafe { VectorReader::new(input, 4) }; // upper DOUBLE
    let rr = unsafe { VectorReader::new(input, 5) }; // rscale DOUBLE (JZS Cauchy scale)
    let ra = unsafe { VectorReader::new(input, 6) }; // alt   VARCHAR
    let mut out = unsafe { VectorWriter::new(output) };
    for i in 0..n {
        if !(0..7).all(|c| unsafe { VectorReader::new(input, c).is_valid(i) }) {
            unsafe { out.set_null(i) };
            continue;
        }
        let t = unsafe { rt.read_f64(i) };
        let n1 = unsafe { rn1.read_i64(i) };
        let n2 = unsafe { rn2.read_i64(i) };
        let lower = unsafe { rl.read_f64(i) };
        let upper = unsafe { ru.read_f64(i) };
        let rscale = unsafe { rr.read_f64(i) };
        let alt = alt_from(unsafe { ra.read_str(i) });
        // JASP default prior: Cauchy(location 0, scale rscale, df 1).
        let (be, _bn) = math::bf_equivalence(t, n1, n2, lower, upper, rscale, 0.0, rscale, 1.0, alt);
        if be.is_finite() {
            // return equivalence BF (BF for H_eq); non-equivalence as a second
            // column would need a table fn — keep the primary equivalence BF.
            unsafe { out.write_f64(i, be) }
        } else {
            unsafe { out.set_null(i) }
        }
    }
}

/// Parse "1,2,3" → [1,2,3] (tolerant of spaces/empties).
fn parse_i64_list(s: &str) -> Vec<i64> {
    s.split(',').filter_map(|x| x.trim().parse::<i64>().ok()).collect()
}
fn parse_f64_list(s: &str) -> Vec<f64> {
    s.split(',').filter_map(|x| x.trim().parse::<f64>().ok()).collect()
}

/// JASP/BayesFactor sampling-plan word → math::SampleType.
fn sample_type_from(s: &str) -> math::SampleType {
    match s.trim().to_lowercase().as_str() {
        "poisson" => math::SampleType::Poisson,
        "jointmulti" => math::SampleType::JointMulti,
        "indepmulti" | "independent" | "indep" => math::SampleType::IndepMulti,
        "hypergeom" | "hypergeometric" => math::SampleType::HyperGeom,
        _ => math::SampleType::Poisson,
    }
}

// ── 3. bf_robustness_curve (table function) ───────────────────────
// JASP "BF robustness check": BF10 as a function of the Cauchy prior scale r,
// over JASP's default grid (see math::robustness_grid).
//
//   bf_robustness_curve(t, n, r [, alt])                 one-/paired-sample
//   bf_robustness_curve(t, ny, nx, r [, alt])            independent two-sample
// returns columns: r DOUBLE, bf10 DOUBLE

struct RobustState {
    rows: Vec<(f64, f64)>,
    offset: usize,
}

const ROB_BATCH: usize = 128;

fn build_robustness_curve() -> Result<TableFunctionBuilder, ExtensionError> {
    TableFunctionBuilder::new("bf_robustness_curve")
        .param(TypeId::Double) // t
        .param(TypeId::Double) // n or ny
        .param(TypeId::Double) // (nx or) r
        .named_param("alt", TypeId::Varchar)
        .named_param("points", TypeId::BigInt)
        .with_state::<RobustState, _>(|bind| {
            let t = unsafe { bind.get_parameter_value(0) }.as_f64_or(f64::NAN);
            let a = unsafe { bind.get_parameter_value(1) }.as_f64_or(f64::NAN);
            let b = unsafe { bind.get_parameter_value(2) }.as_f64_or(f64::NAN);
            let alt_s = unsafe { bind.get_named_parameter_value("alt") }.as_str_or("two");
            let points = unsafe { bind.get_named_parameter_value("points") }.as_i64_or(-1);
            let alt = alt_from(alt_s.as_str());
            if !t.is_finite() || !a.is_finite() || !b.is_finite() {
                return Err(ExtensionError::new(
                    "bf_robustness_curve: t, sample sizes and r must be finite",
                ));
            }
            // one-sample / paired: (t, n, r). Two-sample uses bf_robustness_curve2.
            let n = a as i64;
            let r = b;
            if n < 2 || r <= 0.0 {
                return Err(ExtensionError::new(
                    "bf_robustness_curve: n must be >= 2 and r > 0",
                ));
            }
            let grid = robustness_grid(r, if points >= 2 { Some(points) } else { None });
            let rows: Vec<(f64, f64)> = grid
                .iter()
                .map(|&ri| {
                    let bf = bf_ttest(t, n, ri, alt);
                    (ri, if bf.is_finite() && bf > 0.0 { bf } else { f64::NAN })
                })
                .collect();
            bind.add_result_column("r", TypeId::Double);
            bind.add_result_column("bf10", TypeId::Double);
            Ok(RobustState { rows, offset: 0 })
        })
        .scan(|state, chunk| {
            if state.offset >= state.rows.len() {
                unsafe { duckdb_data_chunk_set_size(chunk.as_raw(), 0) };
                return Ok(());
            }
            let n = ROB_BATCH.min(state.rows.len() - state.offset);
            let raw = unsafe { chunk.as_raw() };
            // col 0: r (DOUBLE)
            {
                let vec = unsafe { duckdb_data_chunk_get_vector(raw, 0) };
                let ptr = unsafe { duckdb_vector_get_data(vec) }.cast::<f64>();
                for i in 0..n {
                    unsafe { *ptr.add(i) = state.rows[state.offset + i].0 };
                }
            }
            // col 1: bf10 (DOUBLE, NaN -> NULL)
            {
                let vec = unsafe { duckdb_data_chunk_get_vector(raw, 1) };
                let ptr = unsafe { duckdb_vector_get_data(vec) }.cast::<f64>();
                let has_null = (0..n).any(|i| !state.rows[state.offset + i].1.is_finite());
                let validity = unsafe { duckdb_vector_get_validity(vec) };
                for i in 0..n {
                    let v = state.rows[state.offset + i].1;
                    unsafe { *ptr.add(i) = v };
                    if has_null && !v.is_finite() {
                        unsafe { duckdb_validity_set_row_invalid(validity, i as idx_t) };
                    }
                }
            }
            unsafe { duckdb_data_chunk_set_size(raw, n as idx_t) };
            state.offset += n;
            Ok(())
        })
        .build()
}

// Two-sample independent variant: bf_robustness_curve2(t, ny, nx, r [, alt]).
fn build_robustness_curve2() -> Result<TableFunctionBuilder, ExtensionError> {
    TableFunctionBuilder::new("bf_robustness_curve2")
        .param(TypeId::Double) // t
        .param(TypeId::BigInt) // ny
        .param(TypeId::BigInt) // nx
        .param(TypeId::Double) // r
        .named_param("alt", TypeId::Varchar)
        .named_param("points", TypeId::BigInt)
        .with_state::<RobustState, _>(|bind| {
            let t = unsafe { bind.get_parameter_value(0) }.as_f64_or(f64::NAN);
            let ny = unsafe { bind.get_parameter_value(1) }.as_i64_or(0);
            let nx = unsafe { bind.get_parameter_value(2) }.as_i64_or(0);
            let r = unsafe { bind.get_parameter_value(3) }.as_f64_or(f64::NAN);
            let alt_s = unsafe { bind.get_named_parameter_value("alt") }.as_str_or("two");
            let points = unsafe { bind.get_named_parameter_value("points") }.as_i64_or(-1);
            let alt = alt_from(alt_s.as_str());
            if !t.is_finite() || !r.is_finite() || ny < 2 || nx < 2 || r <= 0.0 {
                return Err(ExtensionError::new(
                    "bf_robustness_curve2: t and r must be finite, ny/nx >= 2, r > 0",
                ));
            }
            let grid = robustness_grid(r, if points >= 2 { Some(points) } else { None });
            let rows: Vec<(f64, f64)> = grid
                .iter()
                .map(|&ri| {
                    let bf = bf_ttest2(t, ny, nx, ri, alt);
                    (ri, if bf.is_finite() && bf > 0.0 { bf } else { f64::NAN })
                })
                .collect();
            bind.add_result_column("r", TypeId::Double);
            bind.add_result_column("bf10", TypeId::Double);
            Ok(RobustState { rows, offset: 0 })
        })
        .scan(|state, chunk| {
            if state.offset >= state.rows.len() {
                unsafe { duckdb_data_chunk_set_size(chunk.as_raw(), 0) };
                return Ok(());
            }
            let n = ROB_BATCH.min(state.rows.len() - state.offset);
            let raw = unsafe { chunk.as_raw() };
            {
                let vec = unsafe { duckdb_data_chunk_get_vector(raw, 0) };
                let ptr = unsafe { duckdb_vector_get_data(vec) }.cast::<f64>();
                for i in 0..n {
                    unsafe { *ptr.add(i) = state.rows[state.offset + i].0 };
                }
            }
            {
                let vec = unsafe { duckdb_data_chunk_get_vector(raw, 1) };
                let ptr = unsafe { duckdb_vector_get_data(vec) }.cast::<f64>();
                let has_null = (0..n).any(|i| !state.rows[state.offset + i].1.is_finite());
                let validity = unsafe { duckdb_vector_get_validity(vec) };
                for i in 0..n {
                    let v = state.rows[state.offset + i].1;
                    unsafe { *ptr.add(i) = v };
                    if has_null && !v.is_finite() {
                        unsafe { duckdb_validity_set_row_invalid(validity, i as idx_t) };
                    }
                }
            }
            unsafe { duckdb_data_chunk_set_size(raw, n as idx_t) };
            state.offset += n;
            Ok(())
        })
        .build()
}

// ── registration ──────────────────────────────────────────────────

fn register_all(reg: &impl Registrar) -> Result<(), ExtensionError> {
    unsafe {
        reg.register_scalar(
            ScalarFunctionBuilder::new("bf_ttest")
                .param(TypeId::Double)
                .param(TypeId::BigInt)
                .param(TypeId::Double)
                .param(TypeId::Varchar)
                .returns(TypeId::Double)
                .function(bf_ttest_fn),
        )?;
        reg.register_scalar(
            ScalarFunctionBuilder::new("bf_ttest2")
                .param(TypeId::Double)
                .param(TypeId::BigInt)
                .param(TypeId::BigInt)
                .param(TypeId::Double)
                .param(TypeId::Varchar)
                .returns(TypeId::Double)
                .function(bf_ttest2_fn),
        )?;
        reg.register_scalar(
            ScalarFunctionBuilder::new("bf_anova")
                .param(TypeId::Double) // F
                .param(TypeId::BigInt) // n (per group)
                .param(TypeId::BigInt) // k (groups)
                .param(TypeId::Varchar) // prior
                .returns(TypeId::Double)
                .function(bf_anova_fn),
        )?;
        reg.register_scalar(
            ScalarFunctionBuilder::new("bf_regression")
                .param(TypeId::BigInt) // N
                .param(TypeId::BigInt) // p (predictors)
                .param(TypeId::Double) // R2
                .param(TypeId::Varchar) // prior
                .returns(TypeId::Double)
                .function(bf_regression_fn),
        )?;
        reg.register_scalar(
            ScalarFunctionBuilder::new("bf_binomial")
                .param(TypeId::BigInt) // k  successes
                .param(TypeId::BigInt) // n  trials
                .param(TypeId::Double) // theta0
                .param(TypeId::Double) // priorA
                .param(TypeId::Double) // priorB
                .returns(TypeId::Double)
                .function(bf_binomial_fn),
        )?;
        reg.register_scalar(
            ScalarFunctionBuilder::new("bf_multinomial")
                .param(TypeId::Varchar) // counts
                .param(TypeId::Varchar) // priors
                .param(TypeId::Varchar) // thetas
                .returns(TypeId::Double)
                .function(bf_multinomial_fn),
        )?;
        reg.register_scalar(
            ScalarFunctionBuilder::new("bf_contingency")
                .param(TypeId::Varchar) // cells
                .param(TypeId::BigInt) // rows
                .param(TypeId::BigInt) // cols
                .param(TypeId::Varchar) // sample
                .param(TypeId::Varchar) // fixed
                .param(TypeId::Double) // a
                .returns(TypeId::Double)
                .function(bf_contingency_fn),
        )?;
        reg.register_scalar(
            ScalarFunctionBuilder::new("bf_equivalence")
                .param(TypeId::Double) // t
                .param(TypeId::BigInt) // n1
                .param(TypeId::BigInt) // n2 (0 = one-sample)
                .param(TypeId::Double) // lower
                .param(TypeId::Double) // upper
                .param(TypeId::Double) // rscale
                .param(TypeId::Varchar) // alt
                .returns(TypeId::Double)
                .function(bf_equivalence_fn),
        )?;
        reg.register_table(build_robustness_curve()?)?;
        reg.register_table(build_robustness_curve2()?)?;
    }
    Ok(())
}

// Entry point — DuckDB expects the symbol `{EXTENSION_NAME}_init_c_api`
// (EXTENSION_NAME = bayes_ext per Makefile / Cargo [lib]).
#[no_mangle]
pub unsafe extern "C" fn bayes_ext_init_c_api(
    info: duckdb_extension_info,
    access: *const duckdb_extension_access,
) -> bool {
    unsafe {
        init_extension_v2(info, access, quack_rs::DUCKDB_API_VERSION, |con| register_all(con))
    }
}
