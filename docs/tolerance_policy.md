# Tolerance policy

Disagreement is judged by a named kind, not by 'close enough'.

## Kinds

| kind | Rule | Typical use |
| --- | --- | --- |
| `absolute` | `|a-b| <= abs` | CV gaps, AUC gaps |
| `relative` | `|a-b| / max(|b|, ε) <= rel` | Variance ratios, residual gaps |
| `monte_carlo_se` | `|a-b| <= se_mult * SE` | Coverage estimates |
| `kkt_residual` | residual `<= abs` | QP stationarity |
| `invariant` | uses `abs` as a default bound | identities |
| `mixed` | uses `abs` as a default bound | heterogeneous reports |

Boundary convention, locked by tests: `error <= bound` passes (just inside or
exact). `error > bound` fails (just outside). The implementation adds a
`1e-15` absolute pad for float representation of the inequality, not a hidden
widening of scientific tolerance.

## Monte Carlo coverage

Coverage is an expectation. With R replicates of a Bernoulli(p) indicator,
SE = sqrt(p(1-p)/R). Requiring `phat == 0.95` is a specification error. This
laboratory uses R = 2000 for the mean-interval problem so the check is
meaningful and still CI-friendly.

## What tolerance does not do

It does not convert a leaked AUC of 0.90 and a legitimate AUC of 0.60 into
'agreement'. Those are different estimands. It does not excuse a GT2
invariant failure.
