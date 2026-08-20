from __future__ import annotations

from problemforge.loader import Registry
from problemforge.validate import check_tolerance


def test_absolute_just_inside_and_outside() -> None:
    policy = {"kind": "absolute", "abs": 0.05}
    assert check_tolerance(1.05, 1.0, policy).passed
    assert not check_tolerance(1.05 + 1e-9, 1.0, policy).passed


def test_relative_just_inside_and_outside() -> None:
    policy = {"kind": "relative", "rel": 0.1}
    assert check_tolerance(1.1, 1.0, policy).passed
    assert not check_tolerance(1.1 + 1e-9, 1.0, policy).passed


def test_monte_carlo_se_boundaries() -> None:
    policy = {"kind": "monte_carlo_se", "se_mult": 3.0}
    se = 0.01
    assert check_tolerance(0.95 + 0.03, 0.95, policy, monte_carlo_se=se).passed
    assert not check_tolerance(0.95 + 0.03 + 1e-9, 0.95, policy, monte_carlo_se=se).passed


def test_every_problem_has_named_tolerance() -> None:
    for rec in Registry():
        from problemforge.loader import load_yaml_spec

        spec = load_yaml_spec(rec.path)
        assert spec.tolerance_policy.kind in {
            "absolute",
            "relative",
            "monte_carlo_se",
            "kkt_residual",
            "invariant",
            "mixed",
        }
