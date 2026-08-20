from __future__ import annotations

from pydantic import ValidationError
import pytest

from problemforge.schema import ProblemSpec


def test_missing_gt_units_rejected() -> None:
    payload = {
        "id": "x",
        "title": "t",
        "domain": "ml",
        "difficulty": "L1",
        "skills": ["a"],
        "estimated_compute": "1s",
        "seed_policy": "s",
        "problem_statement": "p",
        "inputs": {},
        "outputs": {},
        "assumptions": ["a"],
        "constraints": ["c"],
        "ground_truth_units": [
            {
                "unit_id": "GT1",
                "name": "n",
                "description": "d",
                "kind": "numerical",
            }
        ],
        "validation_strategy": "v",
        "tolerance_policy": {"kind": "absolute", "abs": 0.1},
        "failure_modes": ["f"],
    }
    with pytest.raises(ValidationError):
        ProblemSpec.model_validate(payload)


def test_extra_key_rejected() -> None:
    payload = {
        "id": "x",
        "title": "t",
        "domain": "ml",
        "difficulty": "EXPERT",
        "skills": ["a"],
        "estimated_compute": "1s",
        "seed_policy": "s",
        "problem_statement": "p",
        "inputs": {},
        "outputs": {},
        "assumptions": ["a"],
        "constraints": ["c"],
        "ground_truth_units": [
            {"unit_id": "GT1", "name": "n", "description": "d", "kind": "numerical"},
            {"unit_id": "GT2", "name": "i", "description": "d", "kind": "invariant"},
            {
                "unit_id": "GT3",
                "name": "c",
                "description": "d",
                "kind": "independent_check",
            },
        ],
        "validation_strategy": "v",
        "tolerance_policy": {"kind": "absolute", "abs": 0.1},
        "failure_modes": ["f"],
        "todo": "no extra keys",
    }
    with pytest.raises(ValidationError):
        ProblemSpec.model_validate(payload)


def test_gt1_must_be_numerical() -> None:
    payload = {
        "id": "x",
        "title": "t",
        "domain": "ml",
        "difficulty": "L3",
        "skills": ["a"],
        "estimated_compute": "1s",
        "seed_policy": "s",
        "problem_statement": "p",
        "inputs": {},
        "outputs": {},
        "assumptions": ["a"],
        "constraints": ["c"],
        "ground_truth_units": [
            {"unit_id": "GT1", "name": "n", "description": "d", "kind": "invariant"},
            {"unit_id": "GT2", "name": "i", "description": "d", "kind": "invariant"},
            {
                "unit_id": "GT3",
                "name": "c",
                "description": "d",
                "kind": "independent_check",
            },
        ],
        "validation_strategy": "v",
        "tolerance_policy": {"kind": "absolute", "abs": 0.1},
        "failure_modes": ["f"],
    }
    with pytest.raises(ValidationError):
        ProblemSpec.model_validate(payload)
