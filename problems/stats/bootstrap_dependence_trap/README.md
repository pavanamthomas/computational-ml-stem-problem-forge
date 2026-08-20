# iid bootstrap understates variance under clustering

Estimand: sampling variance of the grand mean under a random-intercept DGP.
What is not identified: a time-series long-run variance.

## What problem is being solved?

Compare iid and cluster bootstrap variances of the mean, and check an
analytic cluster-robust formula.

## What assumptions are required?

iid clusters. iid rows given intercept.

## Why was this method chosen?

The mean is linear, so CRVE is available as an independent check.

## What alternative method could have been used?

Cluster bootstrap of a slope. Different estimand.

## What can go wrong?

Quoting the iid SE. Calling any bootstrap 'robust'.

## How is correctness independently checked?

Analytic CRVE. Ratio lower bound.

## What can legitimately be concluded?

On this DGP the iid bootstrap variance is too small.

## What cannot be concluded?

Coverage rates, or validity under serial correlation.
