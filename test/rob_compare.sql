-- Compare DuckDB bf_robustness_curve / _curve2 against R ttest.tstat reference.
-- rob_ref.csv columns: tag,t,n1,n2,r_target,r,bf10
LOAD 'target/release/bayes_ext.duckdb_extension';
CREATE TABLE ref AS SELECT * FROM read_csv_auto('D:/wsl2/bayes_poc/rob_ref.csv');

-- one-sample case (n1=20, n2=0): compare curve on same grid
WITH got AS (
  SELECT r, bf10 FROM bf_robustness_curve(1.9153, 20, 0.70710678)
)
SELECT max(abs(got.bf10/ref.bf10) - 1) AS max_rel_err_one,
       count(*) AS n
FROM got JOIN ref ON abs(got.r - ref.r) < 1e-9
WHERE ref.tag = 'one_r0.707';

-- two-sample (ny=nx=30)
WITH got2 AS (
  SELECT r, bf10 FROM bf_robustness_curve2(2.5, 30, 30, 1.0)
)
SELECT max(abs(got2.bf10/ref.bf10) - 1) AS max_rel_err_two, count(*) AS n
FROM got2 JOIN ref ON abs(got2.r - ref.r) < 1e-9
WHERE ref.tag = 'two_r1';

-- one-sample r=2 grid (535 pts)
WITH got3 AS (
  SELECT r, bf10 FROM bf_robustness_curve(1.2, 50, 2.0)
)
SELECT max(abs(got3.bf10/ref.bf10) - 1) AS max_rel_err_one2, count(*) AS n
FROM got3 JOIN ref ON abs(got3.r - ref.r) < 1e-9
WHERE ref.tag = 'one_r2';
