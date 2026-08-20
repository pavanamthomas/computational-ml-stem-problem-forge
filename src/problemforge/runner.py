"""Run a reference solver and an independent verifier.

GT1 is the primary numerical object. GT2 is an invariant that is not the same
computation as GT1. GT3 is produced by the verifier on a different code path.
Agreement is judged by the problem's tolerance policy, not by exact float
equality, unless the policy says so.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from problemforge.loader import ProblemModules, Registry, load_problem_modules
from problemforge.rng import DEFAULT_SEED
from problemforge.validate import (
    check_tolerance,
    result_matches_expected_schema,
    validate_problem,
)


@dataclass
class UnitCheck:
    unit_id: str
    passed: bool
    detail: str
    reference_value: Any = None
    verifier_value: Any = None


@dataclass
class RunReport:
    problem_id: str
    seed: int
    ok: bool
    reference: dict[str, Any]
    verifier: dict[str, Any]
    unit_checks: list[UnitCheck] = field(default_factory=list)
    invariant_failures: list[str] = field(default_factory=list)
    schema_errors: list[str] = field(default_factory=list)

    def summary_row(self) -> dict[str, Any]:
        gt1 = self.reference.get("gt1")
        return {
            "problem_id": self.problem_id,
            "ok": self.ok,
            "seed": self.seed,
            "gt1": gt1,
            "n_unit_checks": len(self.unit_checks),
            "n_failed_units": sum(not c.passed for c in self.unit_checks),
            "n_invariant_failures": len(self.invariant_failures),
        }


def _require_dict(result: Any, who: str) -> dict[str, Any]:
    if not isinstance(result, dict):
        raise TypeError(f"{who} must return a dict, got {type(result)!r}")
    if "gt1" not in result:
        raise KeyError(f"{who} result missing gt1")
    if "gt2" not in result:
        raise KeyError(f"{who} result missing gt2")
    return result


def _numeric(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, dict) and "value" in value:
        inner = value["value"]
        if isinstance(inner, (int, float)) and not isinstance(inner, bool):
            return float(inner)
    return None


def _compare_gt1(
    mods: ProblemModules, ref: dict[str, Any], ver: dict[str, Any]
) -> UnitCheck:
    policy = mods.spec.tolerance_policy
    rv = _numeric(ref["gt1"])
    vv = _numeric(ver["gt1"])
    if rv is None or vv is None:
        passed = ref["gt1"] == ver["gt1"]
        return UnitCheck(
            unit_id="GT1",
            passed=passed,
            detail="non-numeric GT1 compared by equality"
            if passed
            else f"GT1 mismatch: reference={ref['gt1']!r} verifier={ver['gt1']!r}",
            reference_value=ref["gt1"],
            verifier_value=ver["gt1"],
        )
    se = None
    if policy.kind == "monte_carlo_se":
        se = float(ver.get("monte_carlo_se") or ref.get("monte_carlo_se") or 0.0)
        if "monte_carlo_se" in ver:
            se = float(ver["monte_carlo_se"])
        elif "monte_carlo_se" in ref:
            se = float(ref["monte_carlo_se"])
        else:
            diagnostics = ver.get("diagnostics") or ref.get("diagnostics") or {}
            se = float(diagnostics.get("monte_carlo_se", 0.0))
    result = check_tolerance(vv, rv, policy, monte_carlo_se=se)
    return UnitCheck(
        unit_id="GT1",
        passed=result.passed,
        detail=result.message,
        reference_value=rv,
        verifier_value=vv,
    )


def _compare_gt2(ref: dict[str, Any], ver: dict[str, Any]) -> UnitCheck:
    """GT2 is an invariant. The verifier must confirm it, not recompute GT1."""
    passed = bool(ver.get("gt2_passed", True))
    detail = str(ver.get("gt2_detail", "GT2 invariant checked by verifier"))
    if "gt2" in ver and isinstance(ver["gt2"], dict) and "passed" in ver["gt2"]:
        passed = bool(ver["gt2"]["passed"])
        detail = str(ver["gt2"].get("detail", detail))
    return UnitCheck(
        unit_id="GT2",
        passed=passed,
        detail=detail,
        reference_value=ref.get("gt2"),
        verifier_value=ver.get("gt2"),
    )


def _compare_gt3(ver: dict[str, Any]) -> UnitCheck:
    passed = bool(ver.get("gt3_passed", True))
    detail = str(ver.get("gt3_detail", "GT3 independent check"))
    if "gt3" in ver and isinstance(ver["gt3"], dict) and "passed" in ver["gt3"]:
        passed = bool(ver["gt3"]["passed"])
        detail = str(ver["gt3"].get("detail", detail))
    return UnitCheck(
        unit_id="GT3",
        passed=passed,
        detail=detail,
        reference_value=None,
        verifier_value=ver.get("gt3"),
    )


def run_problem(
    problem_id: str,
    *,
    seed: int = DEFAULT_SEED,
    registry: Registry | None = None,
) -> RunReport:
    registry = registry or Registry()
    validation = validate_problem(problem_id, registry=registry)
    validation.raise_if_invalid()
    mods = load_problem_modules(problem_id, registry=registry)
    reference = _require_dict(mods.reference.solve(seed=seed), "reference_solution.solve")
    verifier = _require_dict(mods.verifier.verify(seed=seed), "independent_verifier.verify")
    if "gt3" not in verifier:
        raise KeyError("independent_verifier.verify result missing gt3")

    schema_errors = result_matches_expected_schema(
        reference, mods.record.path / "expected_schema.json"
    )
    unit_checks = [
        _compare_gt1(mods, reference, verifier),
        _compare_gt2(reference, verifier),
        _compare_gt3(verifier),
    ]
    invariant_failures: list[str] = []
    inv_result = mods.invariants.check_invariants(reference, verifier, seed=seed)
    if isinstance(inv_result, list):
        invariant_failures = [str(x) for x in inv_result if x]
    elif isinstance(inv_result, dict):
        if not inv_result.get("passed", True):
            invariant_failures = [str(x) for x in inv_result.get("failures", ["invariant failed"])]
    elif inv_result is False:
        invariant_failures = ["check_invariants returned False"]

    ok = (
        all(c.passed for c in unit_checks)
        and not invariant_failures
        and not schema_errors
    )
    return RunReport(
        problem_id=mods.record.qualified_id,
        seed=seed,
        ok=ok,
        reference=reference,
        verifier=verifier,
        unit_checks=unit_checks,
        invariant_failures=invariant_failures,
        schema_errors=schema_errors,
    )


def run_all(
    *, seed: int = DEFAULT_SEED, registry: Registry | None = None
) -> list[RunReport]:
    registry = registry or Registry()
    return [run_problem(r.qualified_id, seed=seed, registry=registry) for r in registry]
