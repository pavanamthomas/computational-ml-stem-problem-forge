# Ground-truth protocol

Every problem ships three ground-truth units. They are different objects.

## GT1 — primary answer

A numerical object computed by the reference solver: a gap, a coverage
estimate, an objective value, an AUC. Regeneration with the same seed must
reproduce GT1. GT1 is not a hard-coded literal.

## GT2 — invariant

A mathematical or statistical property that is **not** the same computation
as GT1. Examples in this corpus:

- Entities in train ∩ test under naive KFold, and zero under GroupKFold.
- Majority-classifier recall = 0 and accuracy = 1 − prevalence.
- `logsumexp(x) = c + logsumexp(x − c)`.
- `cond(X'X)` huge on a Hilbert-like design.
- Monte Carlo SE bound for coverage, not `phat == 0.95`.

If GT2 can be obtained by copying GT1 into another key, it is not GT2.

## GT3 — independent check

A different implementation or derivation. Different splitter, different
reduction, analytic sandwich variance, SciPy versus a loop, NNLS KKT map,
Mann-Whitney versus `sklearn.metrics`. Agreement is judged by the problem's
tolerance policy.

## What ground truth is not

- Solver status flags.
- A single floating-point equality without a named policy.
- An observational 'known answer' imported from a paper table.

## Deliberate broken solvers

Reference modules expose `deliberately_broken_solve` where the wrong
procedure remains numerically plausible. Invariants must fail on that
output. That is part of the protocol, not a joke test.
