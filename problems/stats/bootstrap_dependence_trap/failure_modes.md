# iid bootstrap of clustered rows

The iid bootstrap mimics iid sampling. It is not a model for a shared
cluster intercept.

## What looks right

A bootstrap SE from 400 row-resamples, smaller than the cluster bootstrap SE,
quoted as 'the' uncertainty.

## Mechanism

Var(mean) is driven by G cluster means, not by n = G×m rows. Resampling rows
treats within-cluster copies as new information.

## Incorrect candidate

Report the iid SE. Or use a time-series block bootstrap of arbitrary length
and call it cluster-robust.

## What remains unknown

Serial dependence that is not nested in known clusters.
