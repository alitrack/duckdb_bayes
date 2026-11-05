LOAD '/mnt/d/wsl2/duckdb_bayes/build/release/bayes_ext.duckdb_extension';
SELECT
  bf_binomial(11, 20, 0.5, 1, 1) AS binom_11_20,
  bf_multinomial('12,8', '1,1', '0.5,0.5') AS multinom,
  bf_contingency('50,10,12,28', 2, 2, 'poisson', '', 1.0) AS contig_pois,
  bf_contingency('50,10,12,28', 2, 2, 'indepMulti', 'cols', 1.0) AS contig_indep,
  bf_equivalence(-3.0, 10, 0, -0.5, 0.5, 0.5, 'two') AS equiv_t_neg3,
  bf_equivalence(-2.0, 20, 20, -1.0, 1.0, 0.707, 'two') AS equiv_t_neg2,
  bf_equivalence(1.0, 20, 0, 0.0, 0.5, 0.707, 'greater') AS equiv_t_1_gt;
