"""Independent grouped vs naive CV.

This module does not call ``sklearn.model_selection.cross_val_score`` and does
not use ``GroupKFold``. Grouped folds are assigned by sorting unique entity
identifiers and taking ``fold = rank % n_splits``. Accuracy is the mean of
indicator agreements, not ``sklearn.metrics.accuracy_score``.

The DGP is re-implemented from the specification (same seed, same draws),
not imported from ``reference_solution.py``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression
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


def _generate_data(seed: int) -> pd.DataFrame:
    rng = get_rng(seed)
    intercepts = rng.normal(0.0, INTERCEPT_SD, size=N_ENTITIES)
    rows: list[dict[str, float | int]] = []
    for ent in range(N_ENTITIES):
        x = rng.normal(0.0, 1.0, size=N_OBS)
        logits = intercepts[ent] + X_COEF * x
        y = rng.binomial(1, _sigmoid(logits))
        for i in range(N_OBS):
            rows.append({"entity": ent, "x": float(x[i]), "y": int(y[i])})
    return pd.DataFrame(rows)


def _manual_grouped_folds(
    groups: np.ndarray, n_splits: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    unique = np.sort(np.unique(groups))
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for k in range(n_splits):
        test_entities = set(unique[k::n_splits].tolist())
        test_idx = np.flatnonzero(np.isin(groups, list(test_entities)))
        train_idx = np.flatnonzero(~np.isin(groups, list(test_entities)))
        folds.append((train_idx, test_idx))
    return folds


def _manual_naive_folds(
    n: int, n_splits: int, seed: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Disjoint contiguous blocks after a seeded permutation. Not sklearn KFold."""
    rng = get_rng(seed + 17)
    order = rng.permutation(n)
    bounds = np.linspace(0, n, n_splits + 1, dtype=int)
    folds = []
    for k in range(n_splits):
        test_idx = np.sort(order[bounds[k] : bounds[k + 1]])
        mask = np.ones(n, dtype=bool)
        mask[test_idx] = False
        train_idx = np.flatnonzero(mask)
        folds.append((train_idx, test_idx))
    return folds


def _fit_predict_acc(
    frame: pd.DataFrame, train_idx: np.ndarray, test_idx: np.ndarray
) -> float:
    """One-hot on train entities only; unseen test entities become zeros."""
    y = frame["y"].to_numpy()
    x = frame["x"].to_numpy()
    ent = frame["entity"].to_numpy().reshape(-1, 1)
    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    ent_train = enc.fit_transform(ent[train_idx])
    ent_test = enc.transform(ent[test_idx])
    X_train = np.column_stack([ent_train, x[train_idx]])
    X_test = np.column_stack([ent_test, x[test_idx]])
    clf = LogisticRegression(max_iter=400, solver="lbfgs", random_state=0)
    clf.fit(X_train, y[train_idx])
    pred = clf.predict(X_test)
    return float(np.mean(pred == y[test_idx]))


def verify(seed: int = 2026) -> dict[str, object]:
    frame = _generate_data(seed)
    n = len(frame)
    groups = frame["entity"].to_numpy()

    naive_scores = [
        _fit_predict_acc(frame, tr, te)
        for tr, te in _manual_naive_folds(n, N_SPLITS, seed)
    ]
    grouped_scores = [
        _fit_predict_acc(frame, tr, te)
        for tr, te in _manual_grouped_folds(groups, N_SPLITS)
    ]
    gap = float(np.mean(naive_scores) - np.mean(grouped_scores))

    grouped_overlap = 0
    for tr, te in _manual_grouped_folds(groups, N_SPLITS):
        grouped_overlap += len(set(groups[tr].tolist()) & set(groups[te].tolist()))

    naive_overlap = 0
    for tr, te in _manual_naive_folds(n, N_SPLITS, seed):
        naive_overlap += len(set(groups[tr].tolist()) & set(groups[te].tolist()))

    gt2_passed = grouped_overlap == 0 and naive_overlap > 0
    gt3_passed = gap >= MIN_GAP and grouped_overlap == 0

    return {
        "gt1": gap,
        "gt2": {
            "passed": bool(gt2_passed),
            "detail": (
                f"manual naive entity overlap={naive_overlap}, "
                f"manual grouped overlap={grouped_overlap}"
            ),
            "naive_overlap": int(naive_overlap),
            "grouped_overlap": int(grouped_overlap),
        },
        "gt3": {
            "passed": bool(gt3_passed),
            "detail": (
                f"manual gap={gap:.4f} (min {MIN_GAP}); "
                "grouped folds from sorted entity ranks, accuracy by mean(pred==y)"
            ),
            "value": gap,
            "naive_scores": naive_scores,
            "grouped_scores": grouped_scores,
        },
        "gt2_passed": bool(gt2_passed),
        "gt3_passed": bool(gt3_passed),
        "seed": int(seed),
        "diagnostics": {
            "naive_cv_acc": float(np.mean(naive_scores)),
            "grouped_cv_acc": float(np.mean(grouped_scores)),
        },
    }
