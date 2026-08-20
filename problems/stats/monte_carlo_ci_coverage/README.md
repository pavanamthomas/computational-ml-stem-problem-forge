# 95% mean-interval coverage is a Monte Carlo statement

Estimand: P(interval contains 0) under iid N(0,1), n=30, Student-t interval.
What is not identified: posterior probability for one interval, or coverage
under a non-Normal parent.

## What problem is being solved?

Estimate coverage and judge it by Monte Carlo SE rather than exact equality
to 0.95.

## What assumptions are required?

iid Normal, two-sided t interval, R=2000.

## Why was this method chosen?

It is the textbook sampling model of the t-interval. That is why it is the
check, not a licence for all data.

## What alternative method could have been used?

z-interval, bootstrap percentile. Different objects.

## What can go wrong?

`assert phat == 0.95`. Tiny R. Bayesian reading of one interval.

## How is correctness independently checked?

A replicate loop with `t.ppf`. The SE bound.

## What can legitimately be concluded?

Under this DGP, simulated coverage is consistent with 95% within MC error.

## What cannot be concluded?

Coverage in an observational study.
