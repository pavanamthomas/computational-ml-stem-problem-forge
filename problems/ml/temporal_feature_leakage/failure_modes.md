# Temporal feature leakage

The permitted information set at time t is `{y[s] : s < t}`. A rolling mean
that includes y[t] or y[t+1] is not an element of that set.

## What looks right

Out-of-time R² on the leaked feature is often large. The test segment is later
in calendar time, so the number is easy to misread as forecast skill.

## Mechanism

`Series.rolling(w).mean().shift(-1)` aligns a window that ends at t+1 with
index t. Predicting y[t] from that window uses the future. Because the DGP is
AR(1), the leaked feature is also strongly correlated with y[t+1], which is a
sentinel that does not require inspecting pandas offsets.

## Incorrect candidate

`rolling(..., center=True)`, or standardising with the full series mean before
the chronological cut. Shuffling rows before the split destroys time and
hides the timestamp sentinel.

## What remains unknown

A production feature store's actual as-of join. This DGP does not observe
one.
