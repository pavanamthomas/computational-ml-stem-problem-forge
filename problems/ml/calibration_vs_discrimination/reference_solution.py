"""Two scoring rules: discrimination versus calibration.

What problem is being solved?
    Construct two scores on one DGP such that B ranks better and A is better
    calibrated, so AUC and ECE disagree on 'which model is better'.

What assumptions are required?
    Bernoulli labels given a logistic probability. ECE with equal-width bins.

Why was this method chosen?
    The conflict is the object. A single scalar 'model quality' would hide it.

What alternative method could have been used?
    Brier decomposition, or reliability diagrams with quantile bins.

What can go wrong?
    Reporting only AUC. Using one bin. Isotonic-calibrating B and still
    calling it overconfident.

How is correctness independently checked?
    ECE recomputed from bin counts. AUC via Mann-Whitney U.

What can legitimately be concluded?
    On this DGP, higher AUC is not better ECE.

What cannot be concluded?
    Which score a decision maker should use without a loss function.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import roc_auc_score

from problemforge.rng import get_rng

N = 4000
N_BINS = 10


def _sigmoid(z: NDArray[np.float64]) -> NDArray[np.float64]:
    z = np.clip(z, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-z))


def generate_scores(seed: int = 2026) -> dict[str, NDArray[np.float64]]:
    rng = get_rng(seed)
    x1 = rng.normal(0.0, 1.0, size=N)
    x2 = rng.normal(0.0, 1.0, size=N)
    logit_true = 0.35 * x1 + 0.90 * x2
    p_true = _sigmoid(logit_true)
    y = rng.binomial(1, p_true).astype(float)
    p_a = _sigmoid(0.35 * x1)
    logit_b = 0.35 * x1 + 0.90 * x2
    p_b = _sigmoid(6.0 * logit_b)
    p_a = np.clip(p_a, 1e-6, 1 - 1e-6)
    p_b = np.clip(p_b, 1e-6, 1 - 1e-6)
    return {"y": y, "p_a": p_a, "p_b": p_b, "p_true": p_true}


def reliability_bins(
    y: NDArray[np.float64], p: NDArray[np.float64], n_bins: int = N_BINS
) -> dict[str, list[float]]:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_pred: list[float] = []
    bin_true: list[float] = []
    weights: list[float] = []
    ece = 0.0
    for k in range(n_bins):
        if k == n_bins - 1:
            mask = (p >= edges[k]) & (p <= edges[k + 1])
        else:
            mask = (p >= edges[k]) & (p < edges[k + 1])
        if not np.any(mask):
            continue
        conf = float(p[mask].mean())
        acc = float(y[mask].mean())
        w = float(mask.mean())
        ece += w * abs(acc - conf)
        bin_pred.append(conf)
        bin_true.append(acc)
        weights.append(w)
    return {
        "ece": ece,
        "bin_pred": bin_pred,
        "bin_true": bin_true,
        "weights": weights,
    }


def solve(seed: int = 2026) -> dict[str, object]:
    draws = generate_scores(seed)
    y, p_a, p_b = draws["y"], draws["p_a"], draws["p_b"]
    auc_a = float(roc_auc_score(y, p_a))
    auc_b = float(roc_auc_score(y, p_b))
    rel_a = reliability_bins(y, p_a)
    rel_b = reliability_bins(y, p_b)
    gt1 = float(auc_b - auc_a)
    return {
        "gt1": gt1,
        "gt2": {
            "ece_a": float(rel_a["ece"]),
            "ece_b": float(rel_b["ece"]),
            "ece_gap": float(rel_b["ece"] - rel_a["ece"]),
        },
        "seed": int(seed),
        "diagnostics": {
            "auc_a": auc_a,
            "auc_b": auc_b,
            "ece_a": float(rel_a["ece"]),
            "ece_b": float(rel_b["ece"]),
            "model_a_bin_pred": rel_a["bin_pred"],
            "model_a_bin_true": rel_a["bin_true"],
            "model_b_bin_pred": rel_b["bin_pred"],
            "model_b_bin_true": rel_b["bin_true"],
        },
    }


def deliberately_broken_solve(seed: int = 2026) -> dict[str, object]:
    out = solve(seed)
    out["gt1"] = -abs(float(out["gt1"]))
    out["gt2"] = {"ece_a": 0.0, "ece_b": 0.0, "ece_gap": 0.0}
    out["broken"] = True
    return out
