"""Strict schema for ``problem.yaml``.

The schema is the contract between a problem author and the runner. A missing
ground-truth unit is a specification error, not a runtime warning.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

Difficulty = Literal["L1", "L2", "L3", "L4", "EXPERT", "ADVERSARIAL"]
GTKind = Literal["numerical", "invariant", "independent_check", "protocol"]
GTUnitId = Literal["GT1", "GT2", "GT3"]
ToleranceKind = Literal[
    "absolute",
    "relative",
    "monte_carlo_se",
    "kkt_residual",
    "invariant",
    "mixed",
]


class GroundTruthUnit(BaseModel):
    """One independently checkable ground-truth unit."""

    model_config = ConfigDict(extra="forbid")

    unit_id: GTUnitId
    name: str
    description: str
    kind: GTKind


class TolerancePolicy(BaseModel):
    """How numerical disagreement is judged.

    Monte Carlo coverage problems must use ``monte_carlo_se``. Exact algebraic
    identities may use a tight absolute tolerance. Mixing those without naming
    the kind is rejected.
    """

    model_config = ConfigDict(extra="forbid")

    kind: ToleranceKind
    abs: float | None = None
    rel: float | None = None
    se_mult: float | None = None
    notes: str = ""

    @field_validator("abs", "rel", "se_mult")
    @classmethod
    def _non_negative(cls, value: float | None) -> float | None:
        if value is not None and value < 0:
            raise ValueError("tolerance magnitudes must be non-negative")
        return value


class ProblemSpec(BaseModel):
    """Validated problem specification.

    Required fields match the laboratory authoring standard. Extra keys are
    forbidden so that a typo cannot hide an unvalidated claim.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    title: str
    domain: str
    difficulty: Difficulty
    skills: list[str] = Field(min_length=1)
    estimated_compute: str
    seed_policy: str
    problem_statement: str
    inputs: dict[str, Any]
    outputs: dict[str, Any]
    assumptions: list[str] = Field(min_length=1)
    constraints: list[str] = Field(min_length=1)
    ground_truth_units: list[GroundTruthUnit] = Field(min_length=3)
    validation_strategy: str | list[str]
    tolerance_policy: TolerancePolicy
    failure_modes: list[str] = Field(min_length=1)

    @field_validator("id")
    @classmethod
    def _id_shape(cls, value: str) -> str:
        if not value or value != value.strip():
            raise ValueError("id must be a non-empty stripped string")
        if " " in value:
            raise ValueError("id must not contain spaces")
        return value

    @field_validator("ground_truth_units")
    @classmethod
    def _require_three_units(
        cls, units: list[GroundTruthUnit]
    ) -> list[GroundTruthUnit]:
        ids = [u.unit_id for u in units]
        if ids != ["GT1", "GT2", "GT3"]:
            raise ValueError(
                "ground_truth_units must be exactly GT1, GT2, GT3 in that order"
            )
        kinds = {u.kind for u in units}
        if units[0].kind != "numerical":
            raise ValueError("GT1 must be a numerical/object primary answer")
        if units[1].kind != "invariant":
            raise ValueError("GT2 must be an invariant, not a repeat of GT1")
        if units[2].kind not in {"independent_check", "protocol"}:
            raise ValueError(
                "GT3 must be an independent_check or a protocol audit"
            )
        if len(kinds) < 2:
            raise ValueError("GT units must not all share the same kind")
        return units
