from __future__ import annotations

from problemforge.loader import Registry
from problemforge.validate import validate_all


def test_twelve_complete_problems() -> None:
    registry = Registry()
    ids = sorted(r.qualified_id for r in registry)
    assert len(ids) == 12
    expected = {
        "ml/entity_group_leakage",
        "ml/temporal_feature_leakage",
        "ml/nested_cv_optimism",
        "ml/imbalance_metrics_threshold",
        "ml/calibration_vs_discrimination",
        "numerical/logsumexp_stability",
        "numerical/ill_conditioned_normal_equations",
        "stats/monte_carlo_ci_coverage",
        "stats/bootstrap_dependence_trap",
        "stem/qp_kkt_verification",
        "adversarial/plausible_wrong_auc",
        "adversarial/ai_nested_cv_audit",
    }
    assert set(ids) == expected


def test_short_id_and_qualified_id_resolve() -> None:
    registry = Registry()
    a = registry.get("entity_group_leakage")
    b = registry.get("ml/entity_group_leakage")
    assert a.path == b.path


def test_all_specs_validate() -> None:
    reports = validate_all()
    failed = [r.problem_id for r in reports if not r.ok]
    assert failed == [], failed
    for r in reports:
        spec = r.spec
        assert spec is not None
        assert [u.unit_id for u in spec.ground_truth_units] == ["GT1", "GT2", "GT3"]
        assert spec.ground_truth_units[0].kind == "numerical"
        assert spec.ground_truth_units[1].kind == "invariant"
