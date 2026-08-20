"""Inner CV versus nested outer CV.

What problem is being solved?
    Show that the best inner-CV score after a hyperparameter search overstates
    accuracy on new rows when p is comparable to n and most features are noise.

What assumptions are required?
    iid rows, logistic DGP with two weak signal coordinates, a fixed C grid,
    accuracy as the score.

Why was this method chosen?
    GridSearchCV.best_score_ is the number that is commonly quoted. Nested CV
    is the standard correction for that quote.

What alternative method could have been used?
    A held-out test set after a single inner search, or the Tibshirani–Tibshirani
    bias correction. Those are different estimators.

What can go wrong?
    Scaling before splitting; reporting inner CV; treating selected C as a score.

How is correctness independently checked?
    A handwritten nested loop that never reads GridSearchCV.best_score_.

What can legitimately be concluded?
    On this DGP, inner best accuracy is optimistic relative to nested outer
    accuracy.

What cannot be concluded?
    The magnitude of optimism in an observational study, or that nested CV
    yields an unbiased estimator in general.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from problemforge.rng import get_rng

N = 80
P = 40
N_SIGNAL = 2
COEF = 0.35
C_GRID = [0.02, 0.1, 1.0, 50.0]
INNER = 4
OUTER = 4
MIN_OPTIMISM = 0.01


def _sigmoid(z: NDArray[np.float64]) -> NDArray[np.float64]:
    z = np.clip(z, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-z))


def generate_data(seed: int = 2026) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    rng = get_rng(seed)
    X = rng.normal(0.0, 1.0, size=(N, P))
    logits = COEF * X[:, :N_SIGNAL].sum(axis=1)
    y = rng.binomial(1, _sigmoid(logits)).astype(int)
    return X, y


def inner_best_score(X, y) -> tuple[float, float]:
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=400, solver="lbfgs", random_state=0)),
        ]
    )
    search = GridSearchCV(
        pipe,
        param_grid={"lr__C": C_GRID},
        cv=StratifiedKFold(n_splits=INNER, shuffle=True, random_state=0),
        scoring="accuracy",
        refit=True,
    )
    search.fit(X, y)
    chosen = float(search.best_params_["lr__C"])
    return float(search.best_score_), chosen


def nested_outer_score(X, y) -> float:
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=400, solver="lbfgs", random_state=0)),
        ]
    )
    inner = GridSearchCV(
        pipe,
        param_grid={"lr__C": C_GRID},
        cv=StratifiedKFold(n_splits=INNER, shuffle=True, random_state=1),
        scoring="accuracy",
        refit=True,
    )
    outer = StratifiedKFold(n_splits=OUTER, shuffle=True, random_state=2)
    scores = cross_val_score(inner, X, y, cv=outer, scoring="accuracy")
    return float(scores.mean())


def solve(seed: int = 2026) -> dict[str, object]:
    X, y = generate_data(seed)
    inner, chosen_C = inner_best_score(X, y)
    nested = nested_outer_score(X, y)
    optimism = float(inner - nested)
    return {
        "gt1": optimism,
        "gt2": {
            "selected_C": chosen_C,
            "C_is_in_grid": chosen_C in C_GRID,
            "C_is_not_an_accuracy": not (0.0 <= chosen_C <= 1.0 and chosen_C in {0.0, 1.0} or 0 < chosen_C < 1 and chosen_C not in C_GRID),
        },
        "seed": int(seed),
        "diagnostics": {
            "inner_best_score": inner,
            "nested_outer_score": nested,
            "selected_C": chosen_C,
            "C_grid": C_GRID,
        },
    }


def deliberately_broken_solve(seed: int = 2026) -> dict[str, object]:
    """Treat inner best as if it were nested (zero optimism)."""
    out = solve(seed)
    inner = float(out["diagnostics"]["inner_best_score"])  # type: ignore[index]
    out["gt1"] = 0.0
    out["diagnostics"]["nested_outer_score"] = inner  # type: ignore[index]
    out["broken"] = True
    return out
