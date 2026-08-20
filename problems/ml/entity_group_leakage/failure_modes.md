# Entity-group leakage: documented failure modes

The DGP is a logistic random-intercept model. The intercept is an entity-level
constant. Entity identity is one-hot encoded. The scientific object is the
**validation split**, not the existence of entity IDs.

## What looks right and is wrong

Naive KFold accuracy of a logistic model with entity dummies is often in a
range that looks like a successful classifier (for example 0.75–0.85). That
number is performance on **rows of entities already seen in training**. It is
not performance on new entities.

## Hidden mechanism

`OneHotEncoder(handle_unknown="ignore")` emits a zero vector for an entity
that did not appear in the training fold. Under GroupKFold every test entity
is unseen, so the intercept is unavailable and the model is reduced to the
weak covariate `x`. Under naive KFold the same entity appears on both sides,
the dummy is present, and the intercept is used.

## Incorrect candidate (typical)

Report `cross_val_score(..., cv=KFold)` as the generalisation estimate, or
claim the grouped score is “too conservative.” A more subtle error: fit the
one-hot encoder on the full matrix, then run GroupKFold — the encoder has
already seen every entity, so grouped CV no longer removes the intercept.

## What the sentinel catches

Count entities in `train ∩ test` per fold. Naive KFold: strictly positive.
GroupKFold: identically zero. If a candidate reports a tiny accuracy gap but
the sentinel is still large, the candidate did not use a grouped split.

## What remains unknown

Whether a production system scores new rows of known entities or genuinely
new entities. That is a deployment fact, not a property of this DGP. The
laboratory does not estimate a mixed-effects predictive distribution.
