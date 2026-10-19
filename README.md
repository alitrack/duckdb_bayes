# DuckDB Bayes Extension

DuckDB scalar functions for **BayesFactor t-tests** with JASP-compatible
defaults (JZS Cauchy prior, scale 0.707). Closes the biggest gap in the
DuckDB statistics ecosystem: no existing DuckDB extension provides BayesFactor
analyses.

## Functions

| function | args | returns |
|---|---|---|
| `bf_ttest(t, n, r, alt)` | one-/paired-sample | BF10 (DOUBLE) |
| `bf_ttest2(t, ny, nx, r, alt)` | independent two-sample | BF10 (DOUBLE) |

- `t` = observed t-statistic
- `n` / `ny, nx` = sample size(s) (one-sample `n`; two-sample group sizes)
- `r` = Cauchy prior scale (JASP default **0.707**)
- `alt` = `'two'` (default) | `'right'` | `'left'`

## Example

```sql
LOAD 'bayes_ext.duckdb_extension';
SELECT
  bf_ttest(1.9153, 20, 0.707, 'two')    AS bf10_two,
  bf_ttest2(1.9153, 10, 10, 0.707, 'two') AS bf10_two_indep,
  bf_ttest(1.9153, 20, 0.707, 'right')  AS bf10_right;
```

## Correctness (no simulated values)

The math in `src/math.rs` is a **faithful port of JASP's own**
`jaspTTests/R/informedbayesianttestfunctions.R` (`.bf10_t` + `.dtss`).

Every building block and every BF10 is asserted by `cargo test` against
reference values in `test/fixtures/*.csv` **produced by running that JASP R
source under Windows R** (BayesFactor 0.9.12.4.7). If any number deviates
beyond 1e-4 relative, the build fails — so no expected value is typed by hand
and no result is fabricated.

To regenerate the reference values (Windows):
```
"C:\Program Files\R\R-4.5.2\bin\Rscript.exe" gen_bf_ref.R
```

## Build (WSL, DuckDB 1.2.0 C-API host)

```
make release          # or: cargo build --release
make test
```
