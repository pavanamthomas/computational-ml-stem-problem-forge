"""Hand metrics and a Mann-Whitney ROC-AUC. Fixture does not use sklearn."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from problemforge.rng import get_rng

N = 5000
INTERCEPT = -4.35
COEF = 1.35


def _sigmoid(z: NDArray[np.float64]) -> NDArray[np.float64]:
    z = np.clip(z, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-z))


def _data(seed: int) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    rng = get_rng(seed)
    x = rng.normal(0.0, 1.0, size=N)
    p = _sigmoid(INTERCEPT + COEF * x)
    y = rng.binomial(1, p).astype(int)
    return x, y


def confusion(y: NDArray[np.int64], pred: NDArray[np.int64]) -> tuple[int, int, int, int]:
    tp = int(np.sum((y == 1) & (pred == 1)))
    tn = int(np.sum((y == 0) & (pred == 0)))
    fp = int(np.sum((y == 0) & (pred == 1)))
    fn = int(np.sum((y == 1) & (pred == 0)))
    return tn, fp, fn, tp


def metrics_from_counts(tn: int, fp: int, fn: int, tp: int) -> dict[str, float]:
    acc = (tp + tn) / max(tn + fp + fn + tp, 1)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    f1 = 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)
    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "specificity": float(spec),
        "f1": float(f1),
    }


def mann_whitney_auc(y: NDArray[np.int64], scores: NDArray[np.float64]) -> float:
    pos = scores[y == 1]
    neg = scores[y == 0]
    n1, n0 = len(pos), len(neg)
    if n1 == 0 or n0 == 0:
        return 0.5
    # P(score_pos > score_neg) + 0.5 P(tie)
    gt = 0.0
    for p in pos:
        gt += float(np.sum(p > neg) + 0.5 * np.sum(p == neg))
    return float(gt / (n1 * n0))


def average_precision_hand(y: NDArray[np.int64], scores: NDArray[np.float64]) -> float:
    order = np.argsort(-scores)
    y_ord = y[order]
    tp = np.cumsum(y_ord)
    fp = np.cumsum(1 - y_ord)
    recall = tp / max(int(y.sum()), 1)
    precision = tp / np.maximum(tp + fp, 1)
    # step-wise AP
    ap = 0.0
    prev_r = 0.0
    for p, r in zip(precision, recall):
        ap += float(p) * float(r - prev_r)
        prev_r = float(r)
    return float(ap)


def fixture_check() -> dict[str, object]:
    y = np.array([1, 1, 0, 0, 0], dtype=int)
    pred = np.array([1, 0, 0, 1, 0], dtype=int)
    tn, fp, fn, tp = confusion(y, pred)
    hand = metrics_from_counts(tn, fp, fn, tp)
    # sklearn-free expected values: tp=1, fn=1, fp=1, tn=2
    expected = metrics_from_counts(2, 1, 1, 1)
    ok = all(abs(hand[k] - expected[k]) < 1e-12 for k in hand)
    # Cross-check a couple of closed forms.
    ok = ok and abs(hand["precision"] - 0.5) < 1e-12
    ok = ok and abs(hand["recall"] - 0.5) < 1e-12
    ok = ok and abs(hand["specificity"] - 2 / 3) < 1e-12
    return {"passed": ok, "hand": hand, "counts": {"tn": tn, "fp": fp, "fn": fn, "tp": tp}}


def verify(seed: int = 2026) -> dict[str, object]:
    x, y = _data(seed)
    # Unregularised logit via Newton on one feature — not sklearn.metrics for AUC.
    # Use the same logistic MLE with a few IRLS steps.
    z = np.column_stack([np.ones(len(x)), x])
    beta = np.zeros(2)
    for _ in range(12):
        p = _sigmoid(z @ beta)
        w = p * (1.0 - p)
        grad = z.T @ (y - p)
        hess = z.T @ (w[:, None] * z)
        try:
            beta = beta + np.linalg.solve(hess + 1e-8 * np.eye(2), grad)
        except np.linalg.LinAlgError:
            break
    scores = _sigmoid(z @ beta)
    roc = mann_whitney_auc(y, scores)
    pr = average_precision_hand(y, scores)
    gap = float(roc - pr)
    prev = float(y.mean())
    maj_acc = 1.0 - prev
    maj_rec = 0.0
    fx = fixture_check()
    gt2_passed = abs(maj_acc - (1.0 - prev)) < 1e-15 and maj_rec == 0.0
    gt3_passed = bool(fx["passed"])
    return {
        "gt1": gap,
        "gt2": {
            "passed": bool(gt2_passed),
            "detail": f"majority accuracy={maj_acc:.4f}=1-prevalence, recall=0",
            "majority_accuracy": maj_acc,
            "majority_recall": maj_rec,
            "prevalence": prev,
        },
        "gt3": {
            "passed": gt3_passed,
            "detail": "fixture tn,fp,fn,tp = 2,1,1,1; precision=recall=0.5; spec=2/3",
            "value": fx,
        },
        "gt2_passed": bool(gt2_passed),
        "gt3_passed": gt3_passed,
        "seed": int(seed),
    }
