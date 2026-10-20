# DuckDB Bayes Extension

DuckDB functions for **BayesFactor / JASP-compatible** Bayesian analyses —
t-tests, one-way ANOVA, and linear regression — all on the JZS (Zellner)
framework with JASP's default Cauchy priors. Closes the biggest gap in the
DuckDB statistics ecosystem: no existing DuckDB extension provides BayesFactor
analyses.

## Functions

| function | args | returns |
|---|---|---|
| `bf_ttest(t, n, r, alt)` | one-/paired-sample t-test | BF10 (DOUBLE) |
| `bf_ttest2(t, ny, nx, r, alt)` | independent two-sample t-test | BF10 (DOUBLE) |
| `bf_anova(F, n, k, prior)` | one-way ANOVA (equal group sizes) | BF10 (DOUBLE) |
| `bf_regression(N, p, R2, prior)` | linear regression | BF10 (DOUBLE) |
| `bf_robustness_curve(t, n, r [, alt, points])` | table: BF10 vs prior scale | `r, bf10` |
| `bf_robustness_curve2(t, ny, nx, r [, alt, points])` | table: two-sample | `r, bf10` |

- `t` = observed t-statistic; `n` / `ny, nx` = sample size(s)
- `F` = ANOVA F-statistic; `n` = per-group sample size; `k` = number of groups
- `N` = #observations; `p` = number of predictors; `R2` = full-model R²
- `r` = Cauchy prior scale (JASP t-test default **0.707**)
- `alt` = `'two'` (default) | `'right'` | `'left'`
- `prior` = `'medium'` | `'wide'` | `'ultrawide'` (or a numeric scale)

Prior-scale maps differ by analysis (exactly as in JASP/BayesFactor):

| analysis | medium | wide | ultrawide |
|---|---|---|---|
| ANOVA (fixed) | 0.5 | √2/2 ≈ 0.707 | 1.0 |
| regression | √2/4 ≈ 0.354 | 0.5 | √2/2 ≈ 0.707 |

## Example

```sql
LOAD 'bayes_ext.duckdb_extension';

-- t-tests (JASP default r = 0.707)
SELECT
  bf_ttest(1.9153, 20, 0.707, 'two')    AS bf10_two,
  bf_ttest2(1.9153, 10, 10, 0.707, 'two') AS bf10_two_indep,
  bf_ttest(1.9153, 20, 0.707, 'right')  AS bf10_right;

-- one-way ANOVA: 14.39 as the F-stat, 10 obs per group, 3 groups
SELECT bf_anova(14.39214, 10, 3, 'medium') AS anova_bf10;   -- ≈ 393.7

-- regression: 50 obs, 3 predictors, R² = 0.6023
SELECT bf_regression(50, 3, 0.60233, 'medium') AS reg_bf10; -- ≈ 4.77e6

-- JASP "BF robustness check": BF10 as a function of the prior scale
SELECT * FROM bf_robustness_curve(1.9153, 20, 0.707)
WHERE r BETWEEN 0.5 AND 1.0;
```

## Correctness (no simulated values)

The math in `src/math.rs` is a **faithful port of JASP's own R source**:

- t-tests — `jaspTTests/R/informedbayesianttestfunctions.R` (`.bf10_t` + `.dtss`)
- ANOVA — BayesFactor `oneWayAOV.Fstat` (JZS quadrature over the scaled effect)
- regression — BayesFactor `linearReg.R2stat` (JZS quadrature)

Every building block and every BF10 is asserted by `cargo test` against
reference values in `test/fixtures/*.csv` **produced by running that R source
under Windows R** (BayesFactor 0.9.12.4.7):

- `bf_ref_prim.csv` — 190+ rows of `dt` / `dt(ncp)` / `pt` primitives vs R
- `bf_ref.csv` — 33 rows of t-test BF10 (one-/two-sample + one-sided) vs JASP
- `anova_ref.csv` — 432 rows of one-way ANOVA BF10 across F, n, k, all 3 priors
- `reg_ref.csv` — 288 rows of regression BF10 across N, p, R², all 3 priors

If any number deviates beyond 1e-4 relative, the build fails — so no expected
value is typed by hand and no result is fabricated. The ANOVA/regression
formulas were additionally cross-checked against the *full*
`BayesFactor::anovaBF` / `lmBF` on synthetic data (rel. err. ≤ 1e-10) to pin
the parameter mapping (ANOVA `N` = per-group size, `k` = #groups; regression
rscale uses the *regression* map, not the ANOVA one).

To regenerate the reference values (Windows):
```
"C:\Program Files\R\R-4.5.2\bin\Rscript.exe" gen_bf_ref.R
"C:\Program Files\R\R-4.5.2\bin\Rscript.exe" gen_anova_ref.R
```

## Build (WSL, DuckDB 1.2.0 C-API host)

```
make release          # or: cargo build --release
make test
```
