from __future__ import annotations

from problemforge.audit import audit_problem
from problemforge.loader import load_problem_modules
from problemforge.rng import DEFAULT_SEED
from problemforge.runner import run_problem


def test_earliest_failures() -> None:
    report = audit_problem("adversarial/ai_nested_cv_audit")
    by_id = {a.candidate_id: a.earliest_failure for a in report.audits}
    assert by_id["c1_random_split_despite_groups"] == "split_integrity"
    assert by_id["c2_scaling_outside_cv"] == "preprocessing_isolation"
    assert by_id["c3_inner_cv_as_final"] == "estimation_protocol"


def test_c1_not_diagnosed_as_inner_cv_only() -> None:
    report = audit_problem("adversarial/ai_nested_cv_audit")
    c1 = next(a for a in report.audits if a.candidate_id.startswith("c1"))
    assert c1.earliest_failure == "split_integrity"
    assert c1.failed_stages[0] == "split_integrity"


def test_seed_and_runner() -> None:
    mods = load_problem_modules("adversarial/ai_nested_cv_audit")
    assert mods.reference.solve(seed=DEFAULT_SEED)["gt1"] == mods.reference.solve(
        seed=DEFAULT_SEED
    )["gt1"]
    assert run_problem("adversarial/ai_nested_cv_audit").ok
