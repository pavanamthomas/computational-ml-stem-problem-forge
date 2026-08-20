# Accuracy conceals a 2% prevalence task

Estimand: ROC-AUC minus PR-AUC on in-sample logistic scores, together with
the majority-classifier identities. DGP: logistic, intercept −4.35, prevalence
near 2%. What is not identified: a deployment threshold.

## What problem is being solved?

Show that accuracy and ROC-AUC are not sufficient summaries when positives
are rare and FN cost dominates.

## What assumptions are required?

Stated costs. In-sample scores for metric geometry. Prevalence near 2%.

## Why was this method chosen?

These are the metrics that appear in model reports. The majority classifier
is the baseline accuracy must beat.

## What alternative method could have been used?

Decision-curve analysis, or proper scoring rules alone.

## What can go wrong?

Threshold 0.5 under asymmetric cost. Quoting accuracy. Treating ROC-AUC as
PR-AUC.

## How is correctness independently checked?

Algebraic majority identities. A five-row confusion fixture without sklearn.
Mann-Whitney AUC versus sklearn on the DGP.

## What can legitimately be concluded?

On this DGP, majority recall is 0 and accuracy equals 1-prevalence. ROC-AUC
exceeds PR-AUC by a material amount.

## What cannot be concluded?

Cost ratios or thresholds for a real screening programme.
