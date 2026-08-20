# computational-ml-stem-problem-forge

[![CI](https://github.com/pavanamthomas/computational-ml-stem-problem-forge/actions/workflows/ci.yml/badge.svg)](https://github.com/pavanamthomas/computational-ml-stem-problem-forge/actions)

A laboratory for original computationally intensive ML, statistical, and STEM
problems: unambiguous formalization, a Python reference, an independent
verifier, detection of subtly wrong but numerically plausible answers,
reproducible ground truth, and audit of candidate solutions.

This is working notes for quantitative review — including review of
computational arguments in AI-evaluator settings — not a product, not a
question bank, and not a claim of empirical discovery.

Author: Dr. Pavanam Thomas ([GitHub](https://github.com/pavanamthomas), thomaspavanam@gmail.com).

**Problem → formalization → assumptions → computation → validation → interpretation → limitations.**

## Recruiter 90-Second Audit

Inspect, in this order:

1. [`FLAGSHIP_CASE_STUDY.md`](FLAGSHIP_CASE_STUDY.md) — entity-intercept leakage from formulation through an incorrect candidate.
2. [`problems/ml/entity_group_leakage/`](problems/ml/entity_group_leakage/) and [`problems/ml/nested_cv_optimism/`](problems/ml/nested_cv_optimism/) — EXPERT laboratories with three ground-truth units each.
3. [`problems/adversarial/`](problems/adversarial/) — a leaked AUC that looks like 0.90, and an audit that names the earliest protocol failure.
4. [`docs/ground_truth_protocol.md`](docs/ground_truth_protocol.md) and [`docs/adversarial_failure_taxonomy.md`](docs/adversarial_failure_taxonomy.md).
5. [`src/problemforge/`](src/problemforge/) — registry, schema, runner, audit. Built for 100+ problems; twelve are complete.
6. [`tests/`](tests/) — seed regeneration, schema, independence of verifier files, deliberate-failure detection, tolerance boundaries.

```bash
python -m pip install -e .
python -m pytest
python scripts/run_all.py
python -m problemforge list
```

Python 3.11 or newer.

## Technical Decisions I Can Defend

- **Three ground-truth units, not one number.** GT1 is the primary object. GT2 is an invariant that is a different computation (split occupancy, majority-classifier identities, translation of log-sum-exp, Gram condition number, Monte Carlo SE). GT3 is a different code path. Copying the reference module is not verification.
- **Tolerance is a named policy.** Coverage is judged by Monte Carlo SE, not `phat == 0.95`. KKT is a residual, not solver status. Boundaries are tested just inside and just outside.
- **Leakage has sentinels that do not reuse the primary metric.** Entity overlap counts. Source timestamps. Feature permutation while labels stay fixed.
- **Audit reports the earliest substantive failure.** A random split on grouped data is a split failure even if the quoted score is also inner CV.
- **The registry is path-based.** Adding `problems/<domain>/<id>/problem.yaml` is enough. No hand-maintained index.

## Deliberate Failure Cases

These are required to keep failing in the documented way:

- Naive KFold on an entity-intercept DGP (`entity_group_leakage`).
- Inner-CV best score as generalisation (`nested_cv_optimism`).
- Majority accuracy with recall 0 at ~2% prevalence (`imbalance_metrics_threshold`).
- A leaked score with AUC in 0.85–0.95 (`plausible_wrong_auc`).
- iid bootstrap under clustering (`bootstrap_dependence_trap`).
- `(X'X)^{-1}X'y` on a Hilbert-like design (`ill_conditioned_normal_equations`).
- Naive `log(sum(exp(x)))` overflow (`logsumexp_stability`).

Locked by per-problem tests and `docs/failures_and_corrections.md`.

## Independent Validation

Every problem has `independent_verifier.py` imported from a qualified path so
that twelve files named `reference_solution.py` do not collide. Repo-level
tests assert that the two source files are not identical and that a
deliberately broken reference fails invariants. SciPy log-sum-exp, Mann–Whitney
AUC, analytic cluster-robust variance, NNLS KKT multipliers, and SVD least
squares appear as GT3 paths.

## Reproduce Everything

```bash
python -m pip install -e .
python -m pytest
python scripts/run_all.py
python -m problemforge validate all
python -m problemforge run entity_group_leakage
python -m problemforge audit ai_nested_cv_audit
```

Default seed `2026`. Figures and `outputs/tables/run_summary.csv` are
regenerable. Provenance: [`docs/data_policy.md`](docs/data_policy.md),
[`docs/reproducibility.md`](docs/reproducibility.md).

There is no observational dataset. Every draw is generated in code.

## Limitations and Non-Claims

- The twelve DGPs are stylised. They check procedures. They are not models of
  a labour market, a clinic, or a trading book.
- Candidate audit reads declared protocol YAML. It does not parse arbitrary
  Python.
- Nested-CV optimism is shown for accuracy on one grid. It is not a universal
  constant.
- No result here is a causal finding. Predictive metrics are not identifying
  assumptions.
- Passing CI means the laboratory still runs. It is not a warranty for an
  applied study.

## Interview Questions This Repository Naturally Raises

- If production only scores known entities, is GroupKFold the right split, and
  what estimand did you just change?
- Why is `GridSearchCV.best_score_` not a test accuracy? What quantity is it?
- How would you test leakage without access to the feature-construction source?
- When should coverage of a 95% interval be allowed to be 0.946?
- Why is cond(X'X) the wrong one-number summary of whether β is usable, and
  what residual would you quote instead?
- How do you recover KKT multipliers if the solver does not return them?
- If a candidate both shuffled grouped rows and quoted inner CV, which failure
  do you report first, and why is that a specification rather than taste?
- What would break if GT2 were implemented as `gt2 = gt1`?
- How does an iid bootstrap fail for a mean under a random intercept, and what
  linear-functional sandwich replaces it?
- Discrimination and calibration disagreed on the same draws. Which loss would
  force a choice, and what does the repository refuse to choose for you?

## Corpus (12 problems)

| id | difficulty | object |
| --- | --- | --- |
| `ml/entity_group_leakage` | EXPERT | naive vs GroupKFold on intercepts |
| `ml/temporal_feature_leakage` | EXPERT | rolling window includes t+1 |
| `ml/nested_cv_optimism` | EXPERT | inner best vs nested outer |
| `ml/imbalance_metrics_threshold` | EXPERT | accuracy vs recall/PR-AUC at 2% |
| `ml/calibration_vs_discrimination` | EXPERT | AUC vs ECE |
| `numerical/logsumexp_stability` | L4 | max-subtraction vs overflow |
| `stats/monte_carlo_ci_coverage` | L3 | coverage judged by MC SE |
| `stem/qp_kkt_verification` | EXPERT | KKT residual, not status |
| `adversarial/plausible_wrong_auc` | ADVERSARIAL | leaked 0.9 AUC |
| `adversarial/ai_nested_cv_audit` | ADVERSARIAL | earliest protocol failure |
| `stats/bootstrap_dependence_trap` | L4 | iid vs cluster bootstrap |
| `numerical/ill_conditioned_normal_equations` | ADVERSARIAL | Gram inverse vs QR |

## Repository structure

```text
computational-ml-stem-problem-forge/
├── FLAGSHIP_CASE_STUDY.md
├── src/problemforge/
├── problems/<domain>/<id>/
├── docs/
├── scripts/run_all.py
├── tests/
└── .github/workflows/ci.yml
```

## Related repositories

- [statistical-reasoning-validation](https://github.com/pavanamthomas/statistical-reasoning-validation) — probability identities and inferential mistakes.
- [econometrics-causal-inference-lab](https://github.com/pavanamthomas/econometrics-causal-inference-lab) — estimands when the question is causal.
- [ai-response-evaluation-benchmarks](https://github.com/pavanamthomas/ai-response-evaluation-benchmarks) — structured review of quantitative arguments.

## Citation

See [`CITATION.cff`](CITATION.cff). Licence: MIT, Copyright 2026 Dr. Pavanam Thomas.
