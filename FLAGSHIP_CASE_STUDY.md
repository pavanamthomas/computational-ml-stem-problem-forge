# Flagship case study: entity intercept leakage

This note walks `ml/entity_group_leakage` from formulation to a wrong
candidate. It is a laboratory record, not a tutorial with the traps removed.

## Formulation

Estimand: the difference in mean cross-validated accuracy

```text
Δ = acc_naive_KFold − acc_GroupKFold
```

on a logistic random-intercept DGP. Fifty entities, twelve rows each.
Entity intercept a_g ~ N(0, 2²). A weak covariate x with coefficient 0.25.
The pipeline one-hot encodes entity identity with `handle_unknown='ignore'`
and fits logistic regression.

The deployment question that makes grouping the right split is: **new
entities**, not new rows of a closed set. That question is an assumption,
not a result.

## Hidden traps

1. The naive number is often in a range that looks like a successful
   classifier. It is performance on rows of entities already seen in
   training. The intercept dummy is present in the test fold whenever the
   same entity appeared in the training fold.

2. GroupKFold holds out entire entities. Unseen IDs map to the zero vector.
   The model is reduced to the weak covariate. Accuracy collapses. The
   collapse is the point.

3. A more subtle error: fit the one-hot encoder on the full matrix, then
   run GroupKFold. Every entity has already been seen. Grouped CV no longer
   removes the intercept. The laboratory's reference encoder is inside the
   pipeline and is fit on the training fold only.

4. Interpreting Δ as a causal effect of 'using GroupKFold' is a category
   error. Δ is a validation artefact of an intercept that is not available
   for new entities.

## Reference

`reference_solution.py` draws the DGP with `get_rng(seed)`. It calls
`sklearn.model_selection.cross_val_score` with `KFold(shuffle=True,
random_state=0)` and with `GroupKFold`. GT1 is the mean-accuracy gap. On
seed 2026 that gap is substantially positive (the tests require more than
0.08).

## Independent ground truth

GT2 is not the gap. It is a split invariant: the number of entities that
appear in both train and test. Naive KFold: strictly positive. GroupKFold:
identically zero.

GT3 is a different construction. The verifier does not call
`cross_val_score` and does not use `GroupKFold`. It sorts unique entity IDs
and assigns `fold = rank % n_splits`. It one-hot encodes **training
entities only**, predicts, and averages `pred == y`. A permutation split
that is not sklearn's KFold is used for the naive arm. Accuracy is a mean
of indicators, not `sklearn.metrics.accuracy_score`.

Agreement of the two gaps is judged with an absolute tolerance of 0.12
because the fold constructions differ. Both must be large. A tight match
of two copies of the same splitter would not be an independent check.

## Incorrect candidate

`candidate_solutions/naive_kfold_as_generalisation.yaml` reports a plausible
accuracy from row-wise KFold and sets `groups_respected: false`. The audit
engine's earliest stage is `split_integrity`. The number is not the first
diagnosis.

`deliberately_broken_solve` reports zero gap by copying naive accuracy into
the grouped slot and zeroing the sentinel. Invariants fail. That failure is
required.

## Failure diagnosis

If a submitted answer has a tiny Δ and a large leakage sentinel, the split
was not grouped. If Δ is large and the sentinel is zero for the naive arm,
the naive arm was not naive — the candidate did not implement KFold on rows.
If the encoder was fit on all rows, grouped accuracy will not collapse and
GT1 will fail the minimum-gap invariant.

## Limitations

This DGP has no time index. It does not estimate a mixed-effects predictive
distribution for a new entity (that would draw a new intercept from the
estimated law). It does not observe a production system. If production only
ever scores known entities, a closed-set intercept model can be the right
object — and then the scientific claim must say so, because KFold on rows
still does not describe performance on new entities.

Nothing in this case study is an empirical finding about a firm, a hospital,
or a platform.
