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
