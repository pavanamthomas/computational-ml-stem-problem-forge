"""Imbalanced classification metrics and asymmetric cost.

What problem is being solved?
    Exhibit that accuracy and ROC-AUC can look acceptable at 2% prevalence
    while recall, PR-AUC, and expected FN-heavy cost do not.

What assumptions are required?
    A logistic DGP with prevalence near 2%. Costs are stated, not estimated.
    In-sample scores illustrate metric geometry.

Why was this method chosen?
    sklearn.metrics is the common reporting stack. The majority classifier is
    the hard baseline that accuracy must beat.

What alternative method could have been used?
    Proper scoring rules only, or a full decision-curve analysis. Different
    objects.

What can go wrong?
    Threshold 0.5 under FN cost 50. Quoting accuracy. Treating ROC-AUC as PR-AUC.

How is correctness independently checked?
    Algebraic majority identities, and a four-row confusion-matrix fixture
    computed without sklearn.

What can legitimately be concluded?
    On this DGP, accuracy is not a sufficient summary. Majority recall is 0.

What cannot be concluded?
    A deployment threshold for a real screening programme.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from problemforge.rng import get_rng

N = 5000
INTERCEPT = -4.35
COEF = 1.35
FN_COST = 50.0
FP_COST = 1.0
THRESHOLD_DEFAULT = 0.5


def _sigmoid(z: NDArray[np.float64]) -> NDArray[np.float64]:
    z = np.clip(z, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-z))


def generate_data(seed: int = 2026) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    rng = get_rng(seed)
    x = rng.normal(0.0, 1.0, size=N)
    p = _sigmoid(INTERCEPT + COEF * x)
    y = rng.binomial(1, p).astype(int)
    return x.reshape(-1, 1), y


def _specificity(y_true: NDArray[np.int64], y_pred: NDArray[np.int64]) -> float:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    denom = tn + fp
    return float(tn / denom) if denom else 0.0


def metrics_at(y: NDArray[np.int64], scores: NDArray[np.float64], threshold: float) -> dict[str, float]:
    pred = (scores >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "specificity": _specificity(y, pred),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, scores)),
        "pr_auc": float(average_precision_score(y, scores)),
        "log_loss": float(log_loss(y, np.clip(scores, 1e-12, 1 - 1e-12))),
        "brier": float(brier_score_loss(y, scores)),
    }


def expected_cost(y: NDArray[np.int64], pred: NDArray[np.int64]) -> float:
    fn = float(np.sum((y == 1) & (pred == 0)))
    fp = float(np.sum((y == 0) & (pred == 1)))
    return (FN_COST * fn + FP_COST * fp) / len(y)


def cost_minimizing_threshold(y: NDArray[np.int64], scores: NDArray[np.float64]) -> float:
    grid = np.unique(np.concatenate([[0.0], scores, [1.0]]))
    best_t, best_c = 0.5, np.inf
    for t in grid:
        c = expected_cost(y, (scores >= t).astype(int))
        if c < best_c:
            best_c, best_t = c, float(t)
    return best_t


def solve(seed: int = 2026) -> dict[str, object]:
    X, y = generate_data(seed)
    clf = LogisticRegression(max_iter=400, solver="lbfgs")
    clf.fit(X, y)
    scores = clf.predict_proba(X)[:, 1]
    m_half = metrics_at(y, scores, THRESHOLD_DEFAULT)
    t_cost = cost_minimizing_threshold(y, scores)
    m_cost = metrics_at(y, scores, t_cost)
    prev = float(y.mean())
    maj_pred = np.zeros_like(y)
    maj_acc = float(accuracy_score(y, maj_pred))
    maj_rec = float(recall_score(y, maj_pred, zero_division=0))
    gap = float(m_half["roc_auc"] - m_half["pr_auc"])
    return {
        "gt1": gap,
        "gt2": {
            "prevalence": prev,
            "majority_accuracy": maj_acc,
            "majority_recall": maj_rec,
            "accuracy_identity_error": abs(maj_acc - (1.0 - prev)),
        },
        "seed": int(seed),
        "diagnostics": {
            "metrics_threshold_0.5": m_half,
            "metrics_cost_threshold": m_cost,
            "cost_threshold": t_cost,
            "cost_at_0.5": expected_cost(y, (scores >= THRESHOLD_DEFAULT).astype(int)),
            "cost_at_cost_threshold": expected_cost(y, (scores >= t_cost).astype(int)),
            "fn_cost": FN_COST,
            "fp_cost": FP_COST,
        },
    }


def deliberately_broken_solve(seed: int = 2026) -> dict[str, object]:
    out = solve(seed)
    out["gt1"] = 0.0
    out["gt2"] = {
        "prevalence": 0.5,
        "majority_accuracy": 0.5,
        "majority_recall": 0.5,
        "accuracy_identity_error": 0.0,
    }
    out["broken"] = True
    return out
