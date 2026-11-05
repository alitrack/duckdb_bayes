#!/usr/bin/env Rscript
# Faithful port of JASP jaspEquivalenceTTests:::.equivalence_bf_t +
# .equivalence_cdf_t + jaspTTests:::.posterior_t  into base R, used as the
# oracle for duckdb_bavies equivalence t-test. JASP's default prior is
# Cauchy(location=0, scale=rscale, df=1); we also exercise informative t priors.
#   .posterior_t is IDENTICAL to BayesFactor/JASP .integrand_t normalized:
#       dt(t, nu, ncp=sqrt(neff)*delta) * (1/gamma)*dt((delta-mu)/gamma, kappa)
#   metaBMA::prior("t", c(loc,scale,nu)) == dt((x-loc)/scale, df=nu)/scale
# Run from this dir:  Rscript gen_equiv.R  .

outdir <- commandArgs(trailingOnly = TRUE)[1]

# ── .posterior_t (jaspTTests, verbatim) ──
integrand_t <- function(delta, t, n, nu, mu, gamma, kappa) {
  suppressWarnings(
    dt(x = t, df = nu, ncp = sqrt(n) * delta) *
      (1 / gamma) * dt((delta - mu) / gamma, df = kappa)
  )
}
posterior_t <- function(delta, t, n1, n2 = NULL, independentSamples = FALSE,
                        prior.location, prior.scale, prior.df) {
  neff <- ifelse(independentSamples, n1 * n2 / (n1 + n2), n1)
  nu   <- ifelse(independentSamples, n1 + n2 - 2, n1 - 1)
  numerator <- integrand_t(delta, t, neff, nu, prior.location, prior.scale, prior.df)
  denom <- integrate(function(d) integrand_t(d, t, neff, nu,
                   prior.location, prior.scale, prior.df),
                   -Inf, Inf, rel.tol = 1e-10)$value
  out <- numerator / denom
  out[is.na(out)] <- 0
  out
}
# posterior CDF (JASP .equivalence_cdf_t: integrate posterior_t -Inf..x)
posterior_cdf <- function(x, t, n1, n2, independentSamples,
                          prior.location, prior.scale, prior.df) {
  integrate(function(d) posterior_t(d, t, n1, n2, independentSamples,
            prior.location, prior.scale, prior.df),
            -Inf, x, rel.tol = 1e-10)$value
}
# t-prior CDF  (metaBMA::prior("t", c(loc,scale,nu)))
prior_t_cdf <- function(x, prior.location, prior.scale, prior.df) {
  pt((x - prior.location) / prior.scale, df = prior.df)
}

# ── .equivalence_bf_t (jaspEquivalenceTTests, verbatim) ──
equivalence_bf_t <- function(t, n1, n2, independentSamples,
                             prior.location, prior.scale, prior.df,
                             lower, upper, alternative) {
  # Step 1: prior mass in the equivalence range
  intEqPrior <- prior_t_cdf(upper, prior.location, prior.scale, prior.df) -
                prior_t_cdf(lower, prior.location, prior.scale, prior.df)
  intNeqPrior <- 1 - intEqPrior

  # Step 2: posterior mass in the equivalence range
  # JASP always evaluates both region-bound CDFs (pt(-Inf)=0 for -Inf bound).
  upperbound <- posterior_cdf(upper, t, n1, n2, independentSamples,
                              prior.location, prior.scale, prior.df)
  lowerbound <- posterior_cdf(lower, t, n1, n2, independentSamples,
                              prior.location, prior.scale, prior.df)
  intEqPost <- upperbound - lowerbound
  if (intEqPost < 0) intEqPost <- 0
  # Step 2b: re-standardize posteriors for one-sided tests (JASP uses the
  # posterior CDF at the REGION BOUNDS, not at 0)
  if (alternative == "greater") {
    intEqPost <- intEqPost / (1 - lowerbound)
  } else if (alternative == "less") {
    intEqPost <- intEqPost / upperbound
  }
  intNeqPost <- 1 - intEqPost

  list(bf_equivalence    = intEqPost / intEqPrior,
       bf_nonequivalence = intNeqPost / intNeqPrior)
}

rows <- data.frame()
for (t in c(-3, -2, -1, -0.5, 0, 0.5, 1, 2, 3))
  for (np in list(c(10L, 0L), c(20L, 20L))) {
    n1 <- np[1]; n2 <- np[2]
    indep <- n2 > 0
    for (rscale in c(0.5, 0.707, 1.0))
      for (iv in list(c(-0.5, 0.5), c(-0.2, 0.2), c(-1, 1))) {
        # JASP default prior: Cauchy location 0, scale rscale, df 1
        r <- equivalence_bf_t(t, n1, n2, indep,
                              prior.location = 0, prior.scale = rscale, prior.df = 1,
                              lower = iv[1], upper = iv[2], alternative = "two")
        rows <- rbind(rows, data.frame(
          t = t, n1 = n1, n2 = n2, lower = iv[1], upper = iv[2], rscale = rscale,
          loc = 0, scale = rscale, df = 1, alt = "two",
          bf_eq = r$bf_equivalence, bf_noneq = r$bf_nonequivalence,
          bf_ratio = r$bf_equivalence / r$bf_nonequivalence))
      }
    # one-sided (alternative=greater) and an informative t prior (df=4, loc=0.2)
    for (iv in list(c(0, 0.5))) {
      rg <- equivalence_bf_t(t, n1, n2, indep,
                prior.location = 0, prior.scale = 0.707, prior.df = 1,
                lower = iv[1], upper = iv[2], alternative = "greater")
      rows <- rbind(rows, data.frame(
        t = t, n1 = n1, n2 = n2, lower = iv[1], upper = iv[2], rscale = 0.707,
        loc = 0, scale = 0.707, df = 1, alt = "greater",
        bf_eq = rg$bf_equivalence, bf_noneq = rg$bf_nonequivalence,
        bf_ratio = rg$bf_equivalence / rg$bf_nonequivalence))
      ri <- equivalence_bf_t(t, n1, n2, indep,
                prior.location = 0.2, prior.scale = 0.5, prior.df = 4,
                lower = iv[1], upper = iv[2], alternative = "two")
      rows <- rbind(rows, data.frame(
        t = t, n1 = n1, n2 = n2, lower = iv[1], upper = iv[2], rscale = 0.5,
        loc = 0.2, scale = 0.5, df = 4, alt = "two",
        bf_eq = ri$bf_equivalence, bf_noneq = ri$bf_nonequivalence,
        bf_ratio = ri$bf_equivalence / ri$bf_nonequivalence))
    }
  }
write.csv(rows, file.path(outdir, "equiv_ref.csv"), row.names = FALSE)
cat("equiv (JASP-faithful):", nrow(rows), "rows\n")
