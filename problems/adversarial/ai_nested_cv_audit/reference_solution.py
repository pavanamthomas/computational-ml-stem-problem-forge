"""Correct nested grouped CV used as the audit reference.

What problem is being solved?
    Produce a reference nested-grouped score and a protocol against which
    candidate YAML answers are diagnosed at the earliest failed stage.

What assumptions are required?
    Random intercepts, GroupKFold, scaling inside the pipeline, nested outer
    accuracy as the reported number.

Why was this method chosen?
    The mistakes are the ones that survive a 'the number looks reasonable'
    review. Stage order is part of the specification.

What alternative method could have been used?
    A rubric with weighted error codes. That would hide which failure is first.

What can go wrong?
    Scoring only claimed_score. Reordering stages ad hoc.

How is correctness independently checked?
    Manual grouped folds. The shipped candidates have known earliest failures.

What can legitimately be concluded?
    On this DGP the three candidates fail at split, scaling, and inner-CV
    reporting respectively.

What cannot be concluded?
    That an audit of free-form Python would catch every leak.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from problemforge.rng import get_rng

N_ENT = 36
N_OBS = 8
C_GRID = [0.1, 1.0, 4.0]
OUTER = 4
INNER = 3

REQUIRED_PROTOCOL = {
    "split": "group_kfold",
    "scaling": "inside_cv",
    "reported_score": "nested_outer",
}


def _sigmoid(z: NDArray[np.float64]) -> NDArray[np.float64]:
    z = np.clip(z, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-z))


def generate(seed: int = 2026):
    rng = get_rng(seed)
    intercepts = rng.normal(0.0, 1.2, size=N_ENT)
    X = []
    y = []
    g = []
    for ent in range(N_ENT):
        x = rng.normal(0.0, 1.0, size=(N_OBS, 4))
        logits = intercepts[ent] + 0.25 * x[:, 0]
        X.append(x)
        y.append(rng.binomial(1, _sigmoid(logits)))
        g.append(np.full(N_OBS, ent))
    return np.vstack(X), np.concatenate(y), np.concatenate(g)


def nested_grouped_score(X, y, groups) -> float:
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=400, solver="lbfgs", random_state=0)),
        ]
    )
    outer = GroupKFold(n_splits=OUTER)
    scores: list[float] = []
    for tr, te in outer.split(X, y, groups):
        inner = GridSearchCV(
            pipe,
            param_grid={"lr__C": C_GRID},
            cv=GroupKFold(n_splits=INNER),
            scoring="accuracy",
        )
        inner.fit(X[tr], y[tr], groups=groups[tr])
        pred = inner.predict(X[te])
        scores.append(float(np.mean(pred == y[te])))
    return float(np.mean(scores))


def solve(seed: int = 2026) -> dict[str, object]:
    X, y, groups = generate(seed)
    score = nested_grouped_score(X, y, groups)
    return {
        "gt1": score,
        "gt2": dict(REQUIRED_PROTOCOL),
        "seed": int(seed),
        "diagnostics": {
            "n": int(len(y)),
            "n_entities": int(len(np.unique(groups))),
            "protocol": dict(REQUIRED_PROTOCOL),
        },
    }


def deliberately_broken_solve(seed: int = 2026) -> dict[str, object]:
    out = solve(seed)
    out["gt1"] = 0.99
    out["gt2"] = {
        "split": "kfold",
        "scaling": "outside_cv",
        "reported_score": "inner_cv_best",
    }
    out["broken"] = True
    return out
