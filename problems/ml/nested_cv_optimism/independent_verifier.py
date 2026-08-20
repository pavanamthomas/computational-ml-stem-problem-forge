"""Handwritten nested CV. Does not read GridSearchCV.best_score_."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression
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


def _data(seed: int) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    rng = get_rng(seed)
    X = rng.normal(0.0, 1.0, size=(N, P))
    logits = COEF * X[:, :N_SIGNAL].sum(axis=1)
    y = rng.binomial(1, _sigmoid(logits)).astype(int)
    return X, y


def _stratified_folds(y: NDArray[np.int64], n_splits: int, seed: int) -> list[np.ndarray]:
    rng = get_rng(seed)
    folds = [np.array([], dtype=int) for _ in range(n_splits)]
    for label in (0, 1):
        idx = np.flatnonzero(y == label)
        idx = rng.permutation(idx)
        for i, row in enumerate(idx):
            folds[i % n_splits] = np.append(folds[i % n_splits], row)
    return folds


def _fit_acc(X_tr, y_tr, X_te, y_te, C: float) -> float:
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_tr)
    Xte = scaler.transform(X_te)
    clf = LogisticRegression(C=C, max_iter=400, solver="lbfgs", random_state=0)
    clf.fit(Xtr, y_tr)
    pred = clf.predict(Xte)
    return float(np.mean(pred == y_te))


def _inner_select(X, y, train_idx: np.ndarray, inner_seed: int) -> tuple[float, float]:
    y_tr = y[train_idx]
    inner_folds = _stratified_folds(y_tr, INNER, inner_seed)
    best_C = C_GRID[0]
    best_score = -1.0
    for C in C_GRID:
        accs = []
        for k, inner_te_rel in enumerate(inner_folds):
            inner_te = train_idx[inner_te_rel]
            inner_tr = np.setdiff1d(train_idx, inner_te, assume_unique=False)
            accs.append(_fit_acc(X[inner_tr], y[inner_tr], X[inner_te], y[inner_te], C))
        mean_acc = float(np.mean(accs))
        if mean_acc > best_score:
            best_score = mean_acc
            best_C = C
    return best_C, best_score


def verify(seed: int = 2026) -> dict[str, object]:
    X, y = _data(seed)
    # Inner search on all rows (the optimistic number).
    all_idx = np.arange(len(y))
    chosen_all, inner_best = _inner_select(X, y, all_idx, inner_seed=11)
    outer_folds = _stratified_folds(y, OUTER, seed=22)
    outer_scores = []
    for k, te in enumerate(outer_folds):
        tr = np.setdiff1d(all_idx, te, assume_unique=False)
        C_hat, _ = _inner_select(X, y, tr, inner_seed=30 + k)
        outer_scores.append(_fit_acc(X[tr], y[tr], X[te], y[te], C_hat))
    nested = float(np.mean(outer_scores))
    optimism = float(inner_best - nested)
    c_not_score = chosen_all in C_GRID and not (0.49 <= chosen_all <= 0.51)
    gt2_passed = bool(c_not_score and chosen_all in C_GRID)
    gt3_passed = optimism >= MIN_OPTIMISM
    return {
        "gt1": optimism,
        "gt2": {
            "passed": gt2_passed,
            "detail": (
                f"selected C={chosen_all} is a grid element, not an accuracy "
                f"(inner_best={inner_best:.3f})"
            ),
            "selected_C": chosen_all,
        },
        "gt3": {
            "passed": bool(gt3_passed),
            "detail": "handwritten stratified nested loop; scaler fit on train fold only",
            "value": optimism,
            "outer_scores": outer_scores,
        },
        "gt2_passed": gt2_passed,
        "gt3_passed": bool(gt3_passed),
        "seed": int(seed),
        "diagnostics": {
            "inner_best_score": inner_best,
            "nested_outer_score": nested,
        },
    }
