# Accuracy on a rare-event DGP

A majority classifier on prevalence π has accuracy 1-π and recall 0. At
π ≈ 0.02 that accuracy is 0.98.

## What looks right

Accuracy 0.97 and ROC-AUC around 0.8. Both can occur together with PR-AUC
near the prevalence and with recall far below what a FN-heavy cost would
demand.

## Mechanism

Threshold 0.5 on a model whose probabilities sit near 0.02 predicts almost
everyone negative. ROC-AUC can still rank. PR-AUC uses the positive class as
the denominator of precision and is the harsher summary.

## Incorrect candidate

Resample to 50/50, quote accuracy, and stop. Or pick threshold 0.5 when FN
cost is 50× FP cost.

## What remains unknown

The true cost ratio in any application. It is not identified from this DGP.
