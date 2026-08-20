# Higher AUC is not better calibration

Estimand: AUC_B − AUC_A and ECE_B − ECE_A on two scoring rules for one
Bernoulli DGP. What is not identified: which score a decision maker should
use without a loss.

## What problem is being solved?

Make discrimination and calibration disagree on the same draws.

## What assumptions are required?

Bernoulli labels given a logistic probability. Equal-width ECE bins.

## Why was this method chosen?

A constructed pair of scores isolates the two functionals. Fitting two
opaque libraries would confound training protocol with the metric conflict.

## What alternative method could have been used?

Brier decomposition, quantile bins, or reliability diagrams only.

## What can go wrong?

Reporting only AUC. One bin. Recalibrating B and keeping the 'overconfident'
label.

## How is correctness independently checked?

ECE from bin counts. Mann-Whitney AUC without sklearn.

## What can legitimately be concluded?

On this DGP B ranks better and A is better calibrated.

## What cannot be concluded?
A universal ranking of models, or a deployment choice.
