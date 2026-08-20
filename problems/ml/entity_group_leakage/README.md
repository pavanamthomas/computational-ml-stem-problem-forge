# Entity intercept leakage under naive KFold

Estimand: `naive_cv_acc - grouped_cv_acc` on a logistic random-intercept DGP
with entity identity encoded as a feature. DGP: 50 entities, 12 rows each,
intercept ~ N(0, 2²), weak covariate coefficient 0.25. What is not identified:
performance on a new population of entities from a single naive CV number.

## What problem is being solved?

A pipeline that one-hot encodes entity identity is evaluated with KFold.
Because the label depends on a persistent entity intercept, any split that
puts the same entity on both sides of the fold boundary lets the dummy carry
the intercept into the test fold. The laboratory measures that gap against
GroupKFold, where every test entity is unseen.

## What assumptions are required?

- The intercept is constant within entity.
- Rows are exchangeable given entity. There is no time index.
- Unseen entity IDs map to a zero dummy vector.
- The deployment question of interest is scoring **new entities**, not new
  rows of a closed entity set.

## Why was this method chosen?

Logistic regression on entity dummies is the direct finite-sample analogue of
estimating intercepts. The leakage is then a fact about the split, not about
a particular tree algorithm memorising IDs.

## What alternative method could have been used?

Leave-one-entity-out, mixed-model prediction with a random intercept drawn
from the estimated distribution, or target encoding with nested CV. Those
are different estimands. They are not used as the reference here.

## What can go wrong?

Fitting the encoder on all rows before splitting. Reporting accuracy without
the leakage sentinel. Interpreting the gap as a causal effect of “using
GroupKFold.”

## How is correctness independently checked?

The verifier assigns grouped folds by sorting unique entity IDs and taking
every k-th entity as the test group. It one-hot encodes **training entities
only**, predicts, and averages `pred == y`. It also counts entity overlap on
a permutation split that is not `sklearn.model_selection.KFold`.

## What can legitimately be concluded?

On this DGP, naive KFold accuracy is not an estimate of accuracy on new
entities. The leakage sentinel is a split invariant, not a restatement of
the accuracy gap.

## What cannot be concluded?

That entity features are always illegitimate. If production only ever scores
known entities, a closed-set intercept model can be the right object — and
then KFold on rows is still a poor description of that object if the claim
is “new entity.” This repository does not observe a production system.
