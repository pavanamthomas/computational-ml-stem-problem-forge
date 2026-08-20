# Rolling-window features that include the future

Estimand: out-of-time R² gap between a leaked rolling mean and a lagged
rolling mean for predicting y[t]. DGP: Gaussian AR(1), φ = 0.72. What is not
identified: forecast skill of a production feature store.

## What problem is being solved?

Separate a legal lagged statistic from a rolling window that includes time t
or t+1, using both an R² gap and a timestamp sentinel.

## What assumptions are required?

Causal AR(1). Permitted information at t is strictly {y[s] : s < t}. One
chronological cut.

## Why was this method chosen?

A one-feature linear model attributes the R² gap to the feature construction.

## What alternative method could have been used?

Walk-forward CV, or an explicit Kalman filter. Different estimands.

## What can go wrong?

Centered rolling windows, `shift(-1)`, shuffling, scaling on the whole series.

## How is correctness independently checked?

A NumPy loop rebuilds the lagged mean. Source timestamps are counted from
loop bounds, not from pandas.

## What can legitimately be concluded?

On this DGP the leaked OOS R² is not forecast skill. The timestamp sentinel
is independent of R².

## What cannot be concluded?

That every rolling mean is illegal. The information set has to be named.
