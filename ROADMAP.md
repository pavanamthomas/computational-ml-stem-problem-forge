# Roadmap

Current as of August 2026.

## In scope now

- Twelve complete problem laboratories under `problems/`, covering grouped
  leakage, temporal leakage, nested-CV optimism, imbalance metrics,
  calibration versus discrimination, log-sum-exp, Monte Carlo coverage,
  QP KKT residuals, leaked AUC, candidate-protocol audit, clustered
  bootstrap, and ill-conditioned normal equations.
- A path-based registry that does not require a hand-maintained index.
- CLI: `list`, `validate`, `run`, `audit`.
- CI: `python -m pytest`, `python -m problemforge validate all`, and
  `python scripts/run_all.py`.

## Failures that are part of the design

- Naive KFold accuracy on an entity-intercept DGP.
- Inner-CV best score as a generalisation estimate.
- Majority-class accuracy with recall zero at 2% prevalence.
- iid bootstrap variance under clustering.
- `(X'X)^{-1}X'y` on a Hilbert-like design (finite garbage inverse, or
  LAPACK `LinAlgError`, depending on the BLAS build).

Details: `docs/failures_and_corrections.md`.

## Open (issues)

1. The candidate audit reads declared protocol blocks. Free-form Python
   solutions without a protocol YAML are not parsed.
2. Nested CV optimism is demonstrated on accuracy. Log-loss nested CV is
   not in the corpus.
3. The QP laboratory is inequality-only. Equalities would change the KKT map.

## Explicitly not in scope

- Treating a simulated DGP as an observational finding.
- A leaderboard that ranks models by a single scalar across heterogeneous
  estimands.
- Invented empirical benchmarks or backdated results.

Close an issue only with a test or a limitation sentence.
