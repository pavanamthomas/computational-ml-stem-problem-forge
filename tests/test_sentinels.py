from __future__ import annotations

from problemforge.loader import load_problem_modules
from problemforge.rng import DEFAULT_SEED


def test_leakage_sentinels_entity_and_temporal() -> None:
    ent = load_problem_modules("ml/entity_group_leakage").reference.solve(
        seed=DEFAULT_SEED
    )
    assert ent["gt2"]["naive_leaked_entity_occurrences"] > 0
    assert ent["gt2"]["grouped_leaked_entity_occurrences"] == 0.0

    tmp = load_problem_modules("ml/temporal_feature_leakage").reference.solve(
        seed=DEFAULT_SEED
    )
    assert tmp["gt2"]["leaked_future_source_fraction"] > 0.99
    assert tmp["gt2"]["lagged_future_source_fraction"] == 0.0


def test_majority_classifier_special_case() -> None:
    out = load_problem_modules("ml/imbalance_metrics_threshold").reference.solve(
        seed=DEFAULT_SEED
    )
    assert out["gt2"]["majority_recall"] == 0.0
    assert out["gt2"]["accuracy_identity_error"] < 1e-12


def test_naive_logsumexp_overflow_is_the_special_case() -> None:
    out = load_problem_modules("numerical/logsumexp_stability").reference.solve(
        seed=DEFAULT_SEED
    )
    assert not out["diagnostics"]["naive_is_finite"]
