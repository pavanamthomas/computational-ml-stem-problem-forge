"""Entity-intercept DGP and naive vs grouped CV.

What problem is being solved?
    Quantify how much naive KFold overstates accuracy when a strong entity
    intercept is encoded as a feature and the same entities appear in train
    and test.

What assumptions are required?
    Logistic random intercepts, repeated rows per entity, OneHot with unknown
    categories ignored, no temporal dependence.

Why was this method chosen?
    Logistic regression on entity dummies is the finite-sample analogue of
    estimating entity intercepts. The leakage is then a split property, not a
    tree-memorisation curiosity.

What alternative method could have been used?
    Target encoding, mixed-effects prediction, or a tree on raw entity IDs.
    Those leak by related mechanisms; they are not this DGP's reference.

What can go wrong?
    Fitting the encoder on all rows before splitting reintroduces leakage
    under GroupKFold. Reporting accuracy without the sentinel hides the cause.

How is correctness independently checked?
    ``independent_verifier.py`` rebuilds folds from sorted entity ids and
    counts correct classifications by hand.

What can legitimately be concluded?
    On this DGP, naive KFold accuracy is not an estimate of performance on
    new entities.

What cannot be concluded?
    That grouping is always required, or that entity features are illegitimate
    when the deployment unit is the same closed set of entities.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from problemforge.rng import get_rng

N_ENTITIES = 50
N_OBS = 12
INTERCEPT_SD = 2.0
X_COEF = 0.25
N_SPLITS = 5
MIN_GAP = 0.08


def _sigmoid(z: NDArray[np.float64]) -> NDArray[np.float64]:
    z = np.clip(z, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-z))


def generate_data(seed: int = 2026) -> pd.DataFrame:
    """Repeated rows with a dominant entity intercept and a weak covariate."""
    rng = get_rng(seed)
    intercepts = rng.normal(0.0, INTERCEPT_SD, size=N_ENTITIES)
    rows: list[dict[str, float | int]] = []
    for ent in range(N_ENTITIES):
        x = rng.normal(0.0, 1.0, size=N_OBS)
        logits = intercepts[ent] + X_COEF * x
        y = rng.binomial(1, _sigmoid(logits))
        for i in range(N_OBS):
            rows.append(
                {
                    "entity": ent,
                    "x": float(x[i]),
                    "y": int(y[i]),
                    "intercept": float(intercepts[ent]),
                }
            )
    return pd.DataFrame(rows)


def make_pipeline() -> Pipeline:
    pre = ColumnTransformer(
        [
            (
                "ent",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ["entity"],
            ),
            ("x", "passthrough", ["x"]),
        ]
    )
    return Pipeline(
        [
            ("pre", pre),
            (
                "lr",
                LogisticRegression(max_iter=400, solver="lbfgs", random_state=0),
            ),
        ]
    )


def naive_and_grouped_scores(
    frame: pd.DataFrame, n_splits: int = N_SPLITS
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    X = frame[["entity", "x"]]
    y = frame["y"].to_numpy()
    groups = frame["entity"].to_numpy()
    pipe = make_pipeline()
    naive = cross_val_score(
        pipe, X, y, cv=KFold(n_splits=n_splits, shuffle=True, random_state=0)
    )
    grouped = cross_val_score(
        pipe, X, y, cv=GroupKFold(n_splits=n_splits), groups=groups
    )
    return naive.astype(float), grouped.astype(float)


def leaked_entity_counts(
    frame: pd.DataFrame, n_splits: int = N_SPLITS
) -> dict[str, float]:
    """GT2: entities on both sides of a split. Naive must be > 0; grouped = 0."""
    n = len(frame)
    groups = frame["entity"].to_numpy()
    naive_leaked = 0
    naive_fold_leaks: list[int] = []
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=0)
    for train_idx, test_idx in kf.split(np.arange(n)):
        train_ent = set(groups[train_idx].tolist())
        test_ent = set(groups[test_idx].tolist())
        leaked = len(train_ent & test_ent)
        naive_fold_leaks.append(leaked)
        naive_leaked += leaked
    grouped_leaked = 0
    gkf = GroupKFold(n_splits=n_splits)
    for train_idx, test_idx in gkf.split(np.arange(n), groups=groups):
        train_ent = set(groups[train_idx].tolist())
        test_ent = set(groups[test_idx].tolist())
        grouped_leaked += len(train_ent & test_ent)
    return {
        "naive_leaked_entity_occurrences": float(naive_leaked),
        "naive_mean_leaked_entities_per_fold": float(np.mean(naive_fold_leaks)),
        "grouped_leaked_entity_occurrences": float(grouped_leaked),
    }


def solve(seed: int = 2026) -> dict[str, object]:
    frame = generate_data(seed)
    naive, grouped = naive_and_grouped_scores(frame)
    gap = float(naive.mean() - grouped.mean())
    sentinel = leaked_entity_counts(frame)
    return {
        "gt1": gap,
        "gt2": sentinel,
        "seed": int(seed),
        "diagnostics": {
            "naive_cv_acc": float(naive.mean()),
            "grouped_cv_acc": float(grouped.mean()),
            "naive_fold_scores": naive.tolist(),
            "grouped_fold_scores": grouped.tolist(),
            "min_gap_required": MIN_GAP,
        },
    }


def deliberately_broken_solve(seed: int = 2026) -> dict[str, object]:
    """Report naive CV as if it were grouped. Numerically plausible, scientifically false."""
    out = solve(seed)
    naive = float(out["diagnostics"]["naive_cv_acc"])  # type: ignore[index]
    out["gt1"] = 0.0
    out["diagnostics"]["grouped_cv_acc"] = naive  # type: ignore[index]
    out["gt2"] = {
        "naive_leaked_entity_occurrences": 0.0,
        "naive_mean_leaked_entities_per_fold": 0.0,
        "grouped_leaked_entity_occurrences": 0.0,
    }
    out["broken"] = True
    return out
