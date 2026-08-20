# Failures and corrections

The laboratory keeps inferential and numerical mistakes visible. A successful
test here often means the **wrong procedure still misbehaves** under a known
DGP.

| What was tried | How it failed | Diagnostic | Correction | Locked by | What remains unknown |
| --- | --- | --- | --- | --- | --- |
| Naive KFold on entity-intercept rows with entity dummies | CV accuracy far above GroupKFold | Entities in train ∩ test | GroupKFold; name the deployment unit | `problems/ml/entity_group_leakage` | Closed-set vs new-entity deployment |
| Inner `GridSearchCV.best_score_` as test accuracy | Optimistic relative to nested outer CV | Inner − nested > 0 | Nested CV; C is not a score | `problems/ml/nested_cv_optimism` | Other grids and scores |
| Accuracy at 2% prevalence | High accuracy, poor recall / PR-AUC | Majority identity | Report recall, PR-AUC, costs | `problems/ml/imbalance_metrics_threshold` | Application cost ratios |
| Higher AUC as ‘better model’ | ECE moves the other way | Reliability bins | Report both functionals | `problems/ml/calibration_vs_discrimination` | Loss that would select A or B |
| `log(sum(exp(x)))` on large x | `+inf` | Non-finite naive value | Max-subtraction log-sum-exp | `problems/numerical/logsumexp_stability` | float32 GPU kernels |
| iid bootstrap of clustered rows | Variance too small | Cluster/iid ratio | Cluster bootstrap or CRVE | `problems/stats/bootstrap_dependence_trap` | Serial dependence |
| `(X'X)^{-1}X'y` on Hilbert-like X | Finite β, worse residual | cond(X'X), residual gap | QR / SVD / lstsq | `problems/numerical/ill_conditioned_normal_equations` | Ridge as a different estimand |
| Solver `success=True` as optimality | Status is not KKT | Stationarity + slackness | Assemble residuals from x | `problems/stem/qp_kkt_verification` | Nonconvex QPs |
| Leaked label in a score | AUC 0.85–0.95 | Permute X; residual corr with y | Use only permitted columns | `problems/adversarial/plausible_wrong_auc` | Other contamination channels |
| Candidate YAML with mixed protocol faults | ‘Wrong number’ diagnosis | Stage order | Earliest substantive failure | `problems/adversarial/ai_nested_cv_audit` | Free-form Python |

Process: `docs/lab_process.md`.
