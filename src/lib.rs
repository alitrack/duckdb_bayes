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
    vector::{VectorReader, VectorWriter},
};
use libduckdb_sys::{
    duckdb_data_chunk, duckdb_data_chunk_get_size, duckdb_extension_access,
    duckdb_extension_info, duckdb_function_info, duckdb_vector, idx_t,
};

use math::{bf_ttest, bf_ttest2, alt_from, Alt};

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
            unsafe { out.write_f64(i, bf) };
        } else {
            unsafe { out.set_null(i) };
        }
    }
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
