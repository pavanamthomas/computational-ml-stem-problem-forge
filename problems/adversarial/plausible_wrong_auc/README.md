# A leaked score yields a convincing AUC

Estimand: ROC-AUC of a logistic model that uses only X, versus a contaminated
score that uses y. DGP: weak one-coordinate signal. What is not identified:
whether a high AUC in an observational report is leaked.

## What problem is being solved?

Build a wrong implementation whose AUC looks like a strong result, and detect
the leak without requiring `score == y`.

## What assumptions are required?

Permitted features are the columns of X. y is not permitted at evaluation.

## Why was this method chosen?

A noisy mixture with y produces 0.85–0.95 AUC. Feature permutation is a
sentinel that does not need the source code.

## What alternative method could have been used?

Target encoding leakage, or future-feature leakage. Different channels.

## What can go wrong?

Reporting the leaked AUC. Shuffling y as a 'null'.

## How is correctness independently checked?

Permute X. Residualise the leaked score on X and correlate with y.
Mann-Whitney AUC.

## What can legitimately be concluded?

On this DGP a convincing AUC is obtainable by contamination, and the
legitimate-feature AUC is much lower.

## What cannot be concluded?

That every 0.90 AUC is leaked.
