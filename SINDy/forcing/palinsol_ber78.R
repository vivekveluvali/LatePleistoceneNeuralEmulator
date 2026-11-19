# Orbital forcing components and 65N insolation from palinsol.
#
# For each orbital solution in SOLUTIONS, writes palinsol_<solution>_raw.csv with,
# from START_KA to 0 ka in 0.1 kyr steps (past -> present):
#   age_kyr            kyr before 1950
#   Pi_P_raw           climatic precession  e*sin(varpi)   (varpi as in palinsol, ~282 deg today)
#   Pi_C_raw           coprecession         e*cos(varpi)
#   E_raw              obliquity (radians)
#   Q65N_solstice_raw  daily-mean insolation at 65N, true solar longitude 90 deg  (W/m^2)
#   Q65N_lam120_raw    daily-mean insolation at 65N, true solar longitude 120 deg (W/m^2), SM90 "July"
# Values are NOT standardised; do that in Python over the window you choose.
# Insolation uses palinsol's default solar constant (S0 = 1365 W/m^2).
#
# Solutions (see ?ber78 in palinsol):
#   ber78  Berger (1978); palinsol documents it as accurate for +/- 1 Myr around present.
#          This is the solution that reproduces the SM90 forcing.
#   ber90  Berger and Loutre (1991); accurate for +/- 3 Myr. Use this beyond 1 Ma.
#
# Run from a terminal:  Rscript palinsol_ber78.R
# One-time setup:       Rscript -e 'install.packages("palinsol", repos = "https://cloud.r-project.org")'
#
# Cite: Berger (1978) and/or Berger and Loutre (1991) for the solution used, and
# palinsol (Crucifix, 2023, doi:10.5281/zenodo.14893715) for the implementation.

library(palinsol)

START_KA  <- 1500                           # how far back to go (kyr before 1950)
SOLUTIONS <- c("ber78", "ber90")            # one output file per solution

yr    <- seq(-START_KA * 1e3, 0, by = 100)  # years relative to 1950, 0.1 kyr steps
lat65 <- 65 * pi / 180

for (sol in SOLUTIONS) {
  solfun <- get(sol)                        # palinsol::ber78 or palinsol::ber90
  orb <- t(sapply(yr, solfun))              # one row per time: eps, ecc, varpi, epsp

  PP  <- orb[, "ecc"] * sin(orb[, "varpi"]) # climatic precession
  PC  <- orb[, "ecc"] * cos(orb[, "varpi"]) # coprecession
  OB  <- orb[, "eps"]                       # obliquity (radians)

  Q90  <- apply(orb, 1, function(o) Insol(o, long = pi / 2,         lat = lat65))
  Q120 <- apply(orb, 1, function(o) Insol(o, long = 120 * pi / 180, lat = lat65))

  out <- data.frame(age_kyr           = -yr / 1000,
                    Pi_P_raw          = PP,
                    Pi_C_raw          = PC,
                    E_raw             = OB,
                    Q65N_solstice_raw = Q90,
                    Q65N_lam120_raw   = Q120)
  out <- out[order(-out$age_kyr), ]         # past -> present
  fname <- paste0("palinsol_", sol, "_raw.csv")
  write.csv(out, fname, row.names = FALSE)

  # Sanity check on the last row (0 ka). Expected roughly:
  #   Pi_P_raw ~ -0.0164, E_raw ~ 0.4092, Q65N_lam120_raw ~ 428.7
  cat("\n", fname, ":", nrow(out), "rows\n")
  print(tail(out, 1))
}

# How much do the two solutions differ? Expect small differences before 1 Ma, larger after.
if (all(c("ber78", "ber90") %in% SOLUTIONS)) {
  a <- read.csv("palinsol_ber78_raw.csv")
  b <- read.csv("palinsol_ber90_raw.csv")
  for (win in list(c(0, 1000), c(1000, START_KA))) {
    w <- a$age_kyr >= win[1] & a$age_kyr <= win[2]
    cat(sprintf("\nber78 vs ber90, %g-%g ka: corr(Q65N_lam120) = %.4f, corr(Pi_P) = %.4f, corr(E) = %.4f",
                win[1], win[2],
                cor(a$Q65N_lam120_raw[w], b$Q65N_lam120_raw[w]),
                cor(a$Pi_P_raw[w], b$Pi_P_raw[w]),
                cor(a$E_raw[w], b$E_raw[w])))
  }
  cat("\n")
}

cat("\npalinsol version:", as.character(packageVersion("palinsol")), "\n")