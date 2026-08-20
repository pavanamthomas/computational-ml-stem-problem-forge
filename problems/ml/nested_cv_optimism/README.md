# Inner-CV best score is not a generalisation estimate

Estimand: `inner_best_score - nested_outer_mean` on a weak-signal,
high-dimensional logistic DGP. What is not identified: optimism of an
arbitrary search on observational data.

## What problem is being solved?

Separate a hyperparameter identity from a performance estimate, and show that
the inner best accuracy overstates nested outer accuracy on this DGP.

## What assumptions are required?

iid rows. Two weak signal coordinates. Fixed C grid. Accuracy as the score.

## Why was this method chosen?

`GridSearchCV.best_score_` is the number people quote. Nested CV is the
standard comparator for that quote.

## What alternative method could have been used?

A single held-out test set, or a bias-correction formula. Different estimators.

## What can go wrong?

Reporting inner CV. Scaling outside the CV loop. Confusing selected C with
accuracy.

## How is correctness independently checked?

A handwritten stratified nested loop. The scaler is fit on the training fold
only. `best_score_` is never read.

## What can legitimately be concluded?

On this DGP the inner best accuracy is optimistic relative to nested outer
accuracy. The selected C is not a score.

## What cannot be concluded?

A universal optimism constant, or unbiasedness of nested CV in general.
