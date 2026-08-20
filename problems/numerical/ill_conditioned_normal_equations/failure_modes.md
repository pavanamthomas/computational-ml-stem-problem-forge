# Plausible coefficients, worse residuals

`β = (X'X)^{-1} X'y` on a Hilbert-like design often returns finite numbers
of ordinary magnitude. The residual ||Xβ - y|| and the normal-equation
residual ||X'Xβ - X'y|| reveal the damage.

## What looks right

A coefficient vector with entries like 0.8, -1.2, 3.4. No NaNs.

## Mechanism

cond(X'X) ≈ cond(X)². Forming the Gram matrix squares the condition number.
QR / SVD act on X.

## Incorrect candidate

Print β_naive, note that it is finite, and stop.

## What remains unknown

A regularised estimator's MSE on a different DGP. That is not this problem.
