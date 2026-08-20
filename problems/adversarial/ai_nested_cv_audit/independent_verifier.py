"""Manual grouped nested CV plus audit of shipped candidates."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from problemforge.audit import audit_problem
from problemforge.rng import get_rng

N_ENT = 36
N_OBS = 8
C_GRID = [0.1, 1.0, 4.0]
OUTER = 4
INNER = 3

EXPECTED_EARLIEST = {
    "c1_random_split_despite_groups": "split_integrity",
    "c2_scaling_outside_cv": "preprocessing_isolation",
    "c3_inner_cv_as_final": "estimation_protocol",
}


def _sigmoid(z: NDArray[np.float64]) -> NDArray[np.float64]:
    z = np.clip(z, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-z))


def _data(seed: int):
    rng = get_rng(seed)
    intercepts = rng.normal(0.0, 1.2, size=N_ENT)
    X, y, g = [], [], []
    for ent in range(N_ENT):
        x = rng.normal(0.0, 1.0, size=(N_OBS, 4))
        logits = intercepts[ent] + 0.25 * x[:, 0]
        X.append(x)
        y.append(rng.binomial(1, _sigmoid(logits)))
        g.append(np.full(N_OBS, ent))
    return np.vstack(X), np.concatenate(y), np.concatenate(g)


def _grouped_folds(groups: np.ndarray, n_splits: int) -> list[tuple[np.ndarray, np.ndarray]]:
    unique = np.sort(np.unique(groups))
    folds = []
    for k in range(n_splits):
        test_ent = set(unique[k::n_splits].tolist())
        te = np.flatnonzero(np.isin(groups, list(test_ent)))
        tr = np.flatnonzero(~np.isin(groups, list(test_ent)))
        folds.append((tr, te))
    return folds


def _acc(Xtr, ytr, Xte, yte, C: float) -> float:
    scaler = StandardScaler()
    clf = LogisticRegression(C=C, max_iter=400, solver="lbfgs", random_state=0)
    clf.fit(scaler.fit_transform(Xtr), ytr)
    pred = clf.predict(scaler.transform(Xte))
    return float(np.mean(pred == yte))


def verify(seed: int = 2026) -> dict[str, object]:
    X, y, groups = _data(seed)
    outer_scores = []
    for tr, te in _grouped_folds(groups, OUTER):
        best_C, best = C_GRID[0], -1.0
        inner_folds = _grouped_folds(groups[tr], INNER)
        for C in C_GRID:
            accs = []
            for itr_rel, ite_rel in inner_folds:
                accs.append(
                    _acc(X[tr][itr_rel], y[tr][itr_rel], X[tr][ite_rel], y[tr][ite_rel], C)
                )
            mean_acc = float(np.mean(accs))
            if mean_acc > best:
                best, best_C = mean_acc, C
        outer_scores.append(_acc(X[tr], y[tr], X[te], y[te], best_C))
    score = float(np.mean(outer_scores))
    audit = audit_problem("adversarial/ai_nested_cv_audit", seed=seed)
    by_id = {a.candidate_id: a.earliest_failure for a in audit.audits}
    matches = all(by_id.get(k) == v for k, v in EXPECTED_EARLIEST.items())
    return {
        "gt1": score,
        "gt2": {
            "passed": True,
            "detail": "split=group_kfold, scaling=inside_cv, reported_score=nested_outer",
            "protocol": {
                "split": "group_kfold",
                "scaling": "inside_cv",
                "reported_score": "nested_outer",
            },
        },
        "gt3": {
            "passed": bool(matches),
            "detail": f"earliest failures {by_id}",
            "earliest": by_id,
        },
        "gt2_passed": True,
        "gt3_passed": bool(matches),
        "seed": int(seed),
    }
