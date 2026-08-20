# Adversarial failure taxonomy

Subtly wrong answers that remain numerically plausible are the object of the
`ADVERSARIAL` problems. They are not 'syntax errors'.

## Contamination

- Test labels mixed into a score (plausible AUC 0.85–0.95).
- Future timestamps in a rolling feature.
- Entity identity available on both sides of a row-wise split.

Sentinels: permute permitted features; count source timestamps; count
entities in train ∩ test.

## Protocol

- Inner CV quoted as generalisation.
- Scaling or encoding fit on all rows.
- Random KFold despite grouping.

Diagnosis is the **earliest** stage in

1. split integrity
2. preprocessing isolation
3. estimation protocol
4. leakage sentinel
5. numerical claim

A candidate that splits randomly and also quotes inner CV is a split failure
first.

## Numerical theatre

- Naive log-sum-exp that overflows, replaced by a clip that changes the value.
- `(X'X)^{-1}X'y` with finite, ordinary-looking coefficients.
- Solver `success=True` without KKT residuals.

## What this taxonomy is not

A complete logic of ML mistakes. It is the set of traps this corpus is built
to keep visible.
