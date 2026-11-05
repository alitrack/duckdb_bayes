#!/usr/bin/env Rscript
# Oracle fixture generator for duckdb_bayes "remaining" JASP modules.
# Uses BayesFactor (installed) as the authoritative oracle for the
# equivalence t-test (interval null BFs), and ports JASP's own closed forms
# verbatim for the frequencies modules (binomial / multinomial / contingency).
# Run from this directory:  Rscript gen_remaining.R  .

suppressWarnings(suppressMessages(library(BayesFactor)))
outdir <- commandArgs(trailingOnly = TRUE)[1]
dir.create(outdir, recursive = TRUE, showWarnings = FALSE)

# ── 1) Binomial BF10  (JASP jaspFrequencies .bayesBinomialTest, verbatim) ──
gen_binom <- function() {
  two <- function(counts, n, theta0, a, b) {
    if (theta0 == 0 && counts == 0)
      lb <- lbeta(counts + a, n - counts + b) - lbeta(a, b) - (n - counts) * log(1 - theta0)
    else if (theta0 == 1 && counts == n)
      lb <- lbeta(counts + a, n - counts + b) - lbeta(a, b) - counts * log(theta0)
    else
      lb <- lbeta(counts + a, n - counts + b) - lbeta(a, b) -
        counts * log(theta0) - (n - counts) * log(1 - theta0)
    exp(lb)
  }
  one <- function(counts, n, theta0, a, b, lowerTail) {
    if (theta0 == 0 && counts == 0) l0 <- (n - counts) * log(1 - theta0)
    else if (theta0 == 1 && counts == n) l0 <- counts * log(theta0)
    else l0 <- counts * log(theta0) + (n - counts) * log(1 - theta0)
    t1 <- pbeta(theta0, a + counts, b + n - counts, lower.tail = lowerTail, log.p = TRUE) +
      lbeta(a + counts, b + n - counts)
    t2 <- lbeta(a, b) + pbeta(theta0, a, b, lower.tail = lowerTail, log.p = TRUE)
    exp((t1 - t2) - l0)
  }
  rows <- data.frame()
  for (n in c(8L, 15L, 30L, 60L)) {
    for (theta0 in c(0.2, 0.5, 0.8)) {
      for (ab in list(c(1, 1), c(2, 3), c(0.5, 0.5), c(2, 2))) {
        a <- ab[1]; b <- ab[2]
        for (alt in c("two", "greater", "less")) {
          for (counts in 0:n) {
            bf <- if (alt == "two") two(counts, n, theta0, a, b)
                  else one(counts, n, theta0, a, b, lowerTail = (alt == "less"))
            rows <- rbind(rows, data.frame(counts = counts, n = n, theta0 = theta0,
                                           a = a, b = b, alt = alt, bf10 = as.numeric(bf)))
          }
        }
      }
    }
  }
  rows
}

# ── 2) Multinomial BF10  (JASP .multBayesBfEquality, verbatim) ──
gen_multinom <- function() {
  mult_bf <- function(alphas, counts, thetas) {
    if (sum(thetas) != 1) thetas <- thetas / sum(thetas)
    lxa <- sum(lgamma(alphas + counts)) - lgamma(sum(alphas + counts))
    la  <- sum(lgamma(alphas)) - lgamma(sum(alphas))
    lb <- if (any(rowSums(cbind(thetas, counts)) == 0)) (lxa - la)
          else (lxa - la) - sum(counts * log(thetas))
    exp(lb)
  }
  rows <- data.frame()
  countsets <- list(c(10, 20, 30), c(5, 5, 40), c(1, 9, 20, 70),
                    c(0, 15, 35), c(12, 13, 14))
  alphasets <- list(c(1, 1, 1), c(2, 3, 5), c(1, 1, 1, 1), c(0.5, 0.5, 0.5))
  theta_sets <- list(c(1/3, 1/3, 1/3), c(0.5, 0.25, 0.25),
                     c(0.25, 0.25, 0.25, 0.25), c(0.1, 0.4, 0.5), c(0.5, 0.25, 0.25))
  for (counts in countsets) {
    k <- length(counts)
    for (alphas in alphasets) if (length(alphas) == k)
      for (thetas in theta_sets) if (length(thetas) == k) {
        bf <- mult_bf(alphas, counts, thetas)
        rows <- rbind(rows, data.frame(
          counts = paste(counts, collapse = ","),
          alphas = paste(alphas, collapse = ","),
          thetas = paste(thetas, collapse = ","), bf10 = as.numeric(bf)))
      }
  }
  rows
}

# ── 3) Contingency-table BF  (Gunel & Dickey 1974, BayesFactor source) ──
gen_contig <- function() {
  src <- if (file.exists("contingency-utility.R")) "contingency-utility.R"
         else stop("contingency-utility.R not next to script")
  e <- new.env()
  sys.source(src, envir = e)
  e$logMeanExpLogs <- function(logs) { mx <- max(logs); mx + log(sum(exp(logs - mx))) }
  # hypergeom's sumg is defined inside contingencyHypergeometric in the file.
  ct <- function(y, sampleType, fixedMargin, a) {
    y <- as.matrix(y)
    if (sampleType == "indepMulti" && fixedMargin == "cols") y <- t(y)
    switch(sampleType,
      poisson = e$contingencyPoisson(y, a),
      jointMulti = e$contingencyJointMultinomial(y, a),
      indepMulti = e$contingencyIndepMultinomial(y, a),
      hypergeom = e$contingencyHypergeometric(y, a))
  }
  # Counts are stored row-major to match the Rust/DuckDB cell layout:
  # counts[c] = row r, col c  (cells[r*J+c]). The R oracle builds the same
  # table with byrow=TRUE so it computes contingencyBF on exactly what Rust
  # will reconstruct from the counts string.
  rows <- data.frame()
  # (values, I, J)  — values are row-major
  spec <- list(
    list(c(50, 10, 12, 28), 2, 2),
    list(c(10, 20, 30, 40), 2, 2),
    list(c(8, 3, 1, 5), 2, 2),
    list(c(20, 30, 15, 35, 25, 10), 3, 2),
    list(c(12, 8, 14, 6, 10, 9), 3, 2))
  for (sp in spec) {
    v <- sp[[1]]; I <- sp[[2]]; J <- sp[[3]]
    tbl <- matrix(v, nrow = I, ncol = J, byrow = TRUE)
    counts_str <- paste(as.integer(v), collapse = ",")
    for (a in c(1, 2, 5))
      for (st in c("poisson", "jointMulti", "indepMulti", "hypergeom")) {
        if (st == "hypergeom" && !identical(dim(tbl), c(2L, 2L))) next
        fms <- if (st == "indepMulti") c("rows", "cols") else ""
        for (f in fms) {
          lbf <- tryCatch(ct(tbl, st, f, a), error = function(e) NA_real_)
          rows <- rbind(rows, data.frame(
            counts = counts_str, I = I, J = J, sample_type = st,
            fixed_margin = f, a = a, logbf = as.numeric(lbf)))
        }
      }
  }
  rows
}

# NOTE: the equivalence t-test fixture (equiv_ref.csv) is produced separately
# by gen_equiv.R, which ports JASP's own .equivalence_bf_t/.posterior_t (the
# algorithm this crate implements) into base R. BayesFactor::ttest.tstat uses
# a different (BayesFactor-internal) formulation, so it is NOT the oracle here.

binom <- gen_binom();    write.csv(binom,    file.path(outdir, "binom_ref.csv"),    row.names = FALSE)
mult  <- gen_multinom(); write.csv(mult,     file.path(outdir, "multinom_ref.csv"), row.names = FALSE)
cont  <- gen_contig();   write.csv(cont,     file.path(outdir, "contig_ref.csv"),   row.names = FALSE)

cat(sprintf("binom   : %d rows\n", nrow(binom)))
cat(sprintf("multinom: %d rows\n", nrow(mult)))
cat(sprintf("contig  : %d rows\n", nrow(cont)))
cat("DONE. files in", outdir, "\n")
