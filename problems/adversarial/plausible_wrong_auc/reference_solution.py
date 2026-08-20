"""Legitimate AUC versus a leaked but plausible score.

What problem is being solved?
    Produce a wrong AUC in the 0.85–0.95 band via label contamination, and
    show that a feature-permutation sentinel detects it while a legitimate
    logistic fit on X does not look like that number.

What assumptions are required?
    Permitted features are X. Evaluation labels are not a feature.

Why was this method chosen?
    A convex combination of a weak logit and y, plus noise, looks like a
    strong ranker without being exactly equal to y.

What alternative method could have been used?
    Target leakage through a test-set scaler, or row duplication. Different
    contamination channels.

What can go wrong?
    Reporting the leaked AUC. Running the sentinel by permuting y.

How is correctness independently checked?
    Permute X; leaked AUC stays high. Residualise leaked scores on X; leftover
    correlation with y is large.

What can legitimately be concluded?
    On this DGP a 0.9 AUC is achievable by contamination and is not the
    legitimate-feature AUC.

What cannot be concluded?
    That every high AUC in the wild is leaked.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from problemforge.rng import get_rng

N = 900
P = 6
COEF = 0.18


def _sigmoid(z: NDArray[np.float64]) -> NDArray[np.float64]:
    z = np.clip(z, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-z))


def generate(seed: int = 2026) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    rng = get_rng(seed)
    X = rng.normal(0.0, 1.0, size=(N, P))
    logits = COEF * X[:, 0]
    y = rng.binomial(1, _sigmoid(logits)).astype(int)
    return X, y


def leaked_scores(
    y: NDArray[np.int64], legit: NDArray[np.float64], rng: np.random.Generator
) -> NDArray[np.float64]:
    # Noisy mixture with labels: ranking is strong but not a perfect copy of y.
    return 0.22 * legit + 0.62 * y.astype(float) + rng.normal(0.0, 0.32, size=len(y))


def solve(seed: int = 2026) -> dict[str, object]:
    X, y = generate(seed)
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.34, random_state=0, stratify=y
    )
    clf = LogisticRegression(max_iter=400, solver="lbfgs")
    clf.fit(Xtr, ytr)
    legit = clf.predict_proba(Xte)[:, 1]
    auc_legit = float(roc_auc_score(yte, legit))
    rng = get_rng(seed + 3)
    leaked = leaked_scores(yte, legit, rng)
    auc_leaked = float(roc_auc_score(yte, leaked))
    rng_perm = get_rng(seed + 5)
    Xte_perm = rng_perm.permutation(Xte.reshape(-1)).reshape(Xte.shape)
    legit_perm = clf.predict_proba(Xte_perm)[:, 1]
    auc_legit_perm = float(roc_auc_score(yte, legit_perm))
    leaked_perm = leaked_scores(yte, legit_perm, get_rng(seed + 3))
    auc_leaked_perm = float(roc_auc_score(yte, leaked_perm))
    return {
        "gt1": auc_legit,
        "gt2": {
            "auc_leaked": auc_leaked,
            "auc_legit_perm_X": auc_legit_perm,
            "auc_leaked_perm_X": auc_leaked_perm,
        },
        "seed": int(seed),
        "diagnostics": {
            "auc_legitimate": auc_legit,
            "auc_leaked": auc_leaked,
        },
    }


def deliberately_broken_solve(seed: int = 2026) -> dict[str, object]:
    out = solve(seed)
    out["gt1"] = float(out["gt2"]["auc_leaked"])  # type: ignore[index]
    out["broken"] = True
    return out
