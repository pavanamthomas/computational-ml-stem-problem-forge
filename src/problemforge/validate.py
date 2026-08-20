"""Tolerance checks and specification validation.

A problem is valid only if the YAML parses, the three GT units exist, the
required files are present, and ``expected_schema.json`` lists the keys that
``solve()`` actually returns.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from problemforge.loader import (
    Registry,
    load_problem_modules,
    load_yaml_spec,
    required_files_present,
)
from problemforge.schema import ProblemSpec, TolerancePolicy


@dataclass
class ToleranceResult:
    passed: bool
    error: float
    bound: float
    kind: str
    message: str


def check_tolerance(
    value: float,
    reference: float,
    policy: TolerancePolicy | dict[str, Any],
    *,
    monte_carlo_se: float | None = None,
) -> ToleranceResult:
    """Compare ``value`` to ``reference`` under a named policy.

    Boundary convention used by tests: ``error <= bound`` passes (just inside
    or exact); ``error > bound`` fails (just outside).
    """
    if isinstance(policy, dict):
        policy = TolerancePolicy.model_validate(policy)
    kind = policy.kind
    error = abs(float(value) - float(reference))
    if kind == "absolute":
        if policy.abs is None:
            raise ValueError("absolute tolerance requires abs")
        bound = float(policy.abs)
    elif kind == "relative":
        if policy.rel is None:
            raise ValueError("relative tolerance requires rel")
        scale = max(abs(float(reference)), 1e-15)
        error = error / scale
        bound = float(policy.rel)
    elif kind == "monte_carlo_se":
        if monte_carlo_se is None:
            raise ValueError("monte_carlo_se policy requires the SE value")
        if policy.se_mult is None:
            raise ValueError("monte_carlo_se policy requires se_mult")
        if monte_carlo_se < 0:
            raise ValueError("Monte Carlo SE must be non-negative")
        bound = float(policy.se_mult) * float(monte_carlo_se)
    elif kind == "kkt_residual":
        if policy.abs is None:
            raise ValueError("kkt_residual tolerance requires abs")
        bound = float(policy.abs)
    elif kind in {"invariant", "mixed"}:
        if policy.abs is None:
            raise ValueError(f"{kind} tolerance requires abs as the default bound")
        bound = float(policy.abs)
    else:
        raise ValueError(f"unknown tolerance kind {kind!r}")
    passed = error <= bound + 1e-15
    side = "inside" if passed else "outside"
    message = (
        f"{kind}: |value-reference|={error:.6g} is {side} bound={bound:.6g}"
    )
    return ToleranceResult(
        passed=passed, error=error, bound=bound, kind=kind, message=message
    )


def _as_plain(obj: Any) -> Any:
    if hasattr(obj, "item") and callable(obj.item):
        try:
            return obj.item()
        except (ValueError, AttributeError):
            pass
    if isinstance(obj, dict):
        return {str(k): _as_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_as_plain(v) for v in obj]
    return obj


def result_matches_expected_schema(result: dict[str, Any], schema_path: Path) -> list[str]:
    """Minimal JSON-schema subset: required keys and declared types.

    Full JSON Schema is not a dependency. The file still documents the output
    contract; this checker enforces ``required`` and ``properties.*.type``.
    """
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    required = schema.get("required", [])
    for key in required:
        if key not in result:
            errors.append(f"missing required key {key!r}")
    properties = schema.get("properties", {})
    for key, spec in properties.items():
        if key not in result:
            continue
        expected_type = spec.get("type")
        if expected_type is None:
            continue
        value = result[key]
        if not _json_type_matches(value, expected_type):
            errors.append(
                f"key {key!r} expected type {expected_type!r}, got {type(value).__name__}"
            )
    return errors


def _json_type_matches(value: Any, expected: str | list[str]) -> bool:
    expected_list = [expected] if isinstance(expected, str) else list(expected)
    mapping = {
        "number": (int, float),
        "integer": (int,),
        "string": (str,),
        "object": (dict,),
        "array": (list, tuple),
        "boolean": (bool,),
        "null": type(None),
    }
    if isinstance(value, bool) and "boolean" not in expected_list and "integer" not in expected_list:
        # bool is a subclass of int; do not count it as number unless asked.
        if "number" in expected_list:
            return False
    for name in expected_list:
        py = mapping.get(name)
        if py is None:
            continue
        if name == "null" and value is None:
            return True
        if name != "null" and isinstance(value, py) and not (
            name in {"number", "integer"} and isinstance(value, bool)
        ):
            return True
        if name == "number" and isinstance(value, (int, float)) and not isinstance(value, bool):
            return True
    return False


@dataclass
class ValidationReport:
    problem_id: str
    ok: bool
    spec: ProblemSpec | None
    missing_files: list[str] = field(default_factory=list)
    schema_errors: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def raise_if_invalid(self) -> None:
        if not self.ok:
            parts = self.missing_files + self.schema_errors + self.notes
            raise ValueError(
                f"invalid problem {self.problem_id}: " + "; ".join(parts)
            )


def validate_problem(
    problem_id: str, registry: Registry | None = None
) -> ValidationReport:
    registry = registry or Registry()
    record = registry.get(problem_id)
    missing = required_files_present(record.path)
    schema_errors: list[str] = []
    spec: ProblemSpec | None = None
    try:
        spec = load_yaml_spec(record.path)
    except Exception as exc:  # noqa: BLE001 — report as schema error
        schema_errors.append(str(exc))
    if spec is not None:
        if spec.domain != record.domain:
            schema_errors.append(
                f"domain {spec.domain!r} does not match directory {record.domain!r}"
            )
        mods = None
        try:
            mods = load_problem_modules(problem_id, registry=registry)
            for attr in ("solve",):
                if not hasattr(mods.reference, attr):
                    schema_errors.append(f"reference_solution missing {attr}()")
            if not hasattr(mods.verifier, "verify"):
                schema_errors.append("independent_verifier missing verify()")
            if not hasattr(mods.invariants, "check_invariants"):
                schema_errors.append("invariants missing check_invariants()")
            ref_src = (record.path / "reference_solution.py").read_text(encoding="utf-8")
            ver_src = (record.path / "independent_verifier.py").read_text(encoding="utf-8")
            if ref_src.strip() == ver_src.strip():
                schema_errors.append(
                    "independent_verifier.py is identical to reference_solution.py"
                )
        except Exception as exc:  # noqa: BLE001
            schema_errors.append(f"import failed: {exc}")
    ok = not missing and not schema_errors
    return ValidationReport(
        problem_id=record.qualified_id,
        ok=ok,
        spec=spec,
        missing_files=missing,
        schema_errors=schema_errors,
    )


def validate_all(registry: Registry | None = None) -> list[ValidationReport]:
    registry = registry or Registry()
    return [validate_problem(r.qualified_id, registry=registry) for r in registry]
