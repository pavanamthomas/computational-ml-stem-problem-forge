# Naive (X'X)^{-1}X'y looks plausible and is worse

Estimand: residual-norm gap between the naive Gram inverse and QR on a
Hilbert-like design. What is not identified: a causal slope.

## What problem is being solved?

Show that a finite, ordinary-looking least-squares vector can be a worse
residual minimiser than QR/SVD.

## What assumptions are required?

Hilbert-like columns. float64.

## Why was this method chosen?

It is the classic squared-condition-number trap. Residuals are the check.

## What alternative method could have been used?

SVD (GT3), or ridge (a different estimand).

## What can go wrong?

Trusting finite β. Forming X'X.

## How is correctness independently checked?

SciPy SVD solve. `np.linalg.lstsq`. cond(X'X).

## What can legitimately be concluded?

On this design the naive inverse loses residual accuracy.

## What cannot be concluded?

That every use of a Gram matrix is wrong.
