# Discrimination is not calibration

ROC-AUC is a ranking functional. ECE is a bin-wise probability-frequency
gap. They can move in opposite directions.

## What looks right

Model B has AUC 0.85 versus A's 0.70. If the report stops there, B 'wins'.

## Mechanism

B uses an extra signal coordinate, so ranking improves, then applies a steep
logit map so predicted probabilities pile up near 0 and 1. Observed
frequencies in those bins do not match 0.02 or 0.98.

## Incorrect candidate

Quote only AUC. Or isotonic-calibrate B and still describe it as the
overconfident scoring rule in this laboratory.

## What remains unknown

The loss function that would select A or B in a decision problem.
