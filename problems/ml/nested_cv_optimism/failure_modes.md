# Nested CV optimism

`GridSearchCV.best_score_` is the mean inner-CV accuracy of the winning
hyperparameter, computed on the same rows that were used to choose it.

## What looks right

A number such as 0.78 accuracy after a C-grid search. It is a valid inner-CV
score. It is not a generalisation estimate for the search procedure.

## Mechanism

When p is comparable to n and most coordinates are noise, some C values overfit
fold noise. The max over the grid is biased upward. Nested outer CV evaluates
the whole search on held-out rows.

## Incorrect candidate

Quote `best_score_` in a table of “test accuracy.” Scale X on the full matrix.
Treat the selected C as if it were a performance metric.

## What remains unknown

How large the optimism is for a different grid, a different score, or a
non-iid sample.
