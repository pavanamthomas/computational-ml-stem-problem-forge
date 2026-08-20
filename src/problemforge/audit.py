"""Audit AI-generated or submitted candidate solutions.

The audit does not stop at “the number is wrong.” It walks a fixed stage
order and reports the earliest substantive failure:

1. split integrity (grouped vs random, temporal vs shuffled)
2. preprocessing isolation (scaling/encoding fitted outside the training fold)
3. estimation protocol (inner CV reported as generalisation)
4. leakage / contamination sentinels
5. numerical claim against GT1

A candidate that used a random split on grouped data is diagnosed as a split
failure even if its reported score also happens to be numerically off.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from problemforge.loader import Registry, load_problem_modules
from problemforge.rng import DEFAULT_SEED

STAGE_ORDER = (
    "split_integrity",
    "preprocessing_isolation",
    "estimation_protocol",
    "leakage_sentinel",
    "numerical_claim",
)


@dataclass
class CandidateAudit:
    candidate_id: str
    path: Path
    earliest_failure: str | None
    failed_stages: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    claimed: dict[str, Any] = field(default_factory=dict)
    passed: bool = False


@dataclass
class AuditReport:
    problem_id: str
    n_candidates: int
    audits: list[CandidateAudit] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        if self.n_candidates == 0:
            return True
        return all(a.earliest_failure is not None or a.passed for a in self.audits)


def _load_candidate(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        data = yaml.safe_load(text)
    elif path.suffix.lower() == ".json":
        data = json.loads(text)
    else:
        raise ValueError(f"unsupported candidate suffix: {path.suffix}")
    if not isinstance(data, dict):
        raise ValueError(f"candidate {path} is not a mapping")
    return data


def _protocol_of(candidate: dict[str, Any]) -> dict[str, Any]:
    proto = candidate.get("protocol", {})
    if not isinstance(proto, dict):
        raise ValueError("protocol must be a mapping")
    return proto


def _required_protocol(mods) -> dict[str, Any]:
    if hasattr(mods.invariants, "REQUIRED_PROTOCOL"):
        req = getattr(mods.invariants, "REQUIRED_PROTOCOL")
        if isinstance(req, dict):
            return req
    if hasattr(mods.reference, "REQUIRED_PROTOCOL"):
        req = getattr(mods.reference, "REQUIRED_PROTOCOL")
        if isinstance(req, dict):
            return req
    return {
        "split": "group_kfold",
        "scaling": "inside_cv",
        "reported_score": "nested_outer",
    }


def audit_candidate(
    candidate: dict[str, Any],
    *,
    required: dict[str, Any],
    reference_gt1: Any | None = None,
    path: Path | None = None,
) -> CandidateAudit:
    cid = str(candidate.get("id") or candidate.get("candidate_id") or "unnamed")
    proto = _protocol_of(candidate)
    notes: list[str] = []
    failed: list[str] = []

    split = str(proto.get("split", "")).lower()
    required_split = str(required.get("split", "")).lower()
    if required_split and split and split != required_split:
        failed.append("split_integrity")
        notes.append(
            f"split={split!r} violates required split={required_split!r}"
        )
    if proto.get("groups_respected") is False:
        if "split_integrity" not in failed:
            failed.append("split_integrity")
        notes.append("groups_respected is False")

    scaling = str(proto.get("scaling", "")).lower()
    required_scaling = str(required.get("scaling", "")).lower()
    if required_scaling and scaling and scaling != required_scaling:
        failed.append("preprocessing_isolation")
        notes.append(
            f"scaling={scaling!r} violates required scaling={required_scaling!r}"
        )
    if proto.get("scaler_fit_on") in {"full_data", "train_plus_test", "all_rows"}:
        if "preprocessing_isolation" not in failed:
            failed.append("preprocessing_isolation")
        notes.append(f"scaler_fit_on={proto.get('scaler_fit_on')!r}")

    reported = str(proto.get("reported_score", "")).lower()
    required_reported = str(required.get("reported_score", "")).lower()
    if required_reported and reported and reported != required_reported:
        failed.append("estimation_protocol")
        notes.append(
            f"reported_score={reported!r} violates required {required_reported!r}"
        )
    if reported in {"inner_cv_best", "inner_cv", "gridsearch_best_score"}:
        if "estimation_protocol" not in failed:
            failed.append("estimation_protocol")
        notes.append("inner CV score reported as generalisation")

    leak_flag = proto.get("leakage") or candidate.get("leakage")
    if leak_flag in {True, "present", "label_in_score", "future_in_feature"}:
        failed.append("leakage_sentinel")
        notes.append(f"leakage flag {leak_flag!r}")
    if proto.get("uses_test_labels") is True:
        if "leakage_sentinel" not in failed:
            failed.append("leakage_sentinel")
        notes.append("uses_test_labels=True")

    claimed_score = candidate.get("claimed_score", candidate.get("gt1"))
    if reference_gt1 is not None and claimed_score is not None:
        try:
            delta = abs(float(claimed_score) - float(reference_gt1))
            # A large numerical gap is a failure only if no earlier protocol
            # stage already fired. It is still recorded.
            rel = delta / max(abs(float(reference_gt1)), 1e-8)
            if delta > 0.05 and rel > 0.05:
                failed.append("numerical_claim")
                notes.append(
                    f"claimed_score={claimed_score} vs reference GT1={reference_gt1}"
                )
        except (TypeError, ValueError):
            notes.append("claimed_score is not numeric; skipped numerical_claim")

    # Honour an author-supplied earliest_failure only after computing ours.
    earliest = None
    for stage in STAGE_ORDER:
        if stage in failed:
            earliest = stage
            break
    passed = earliest is None
    return CandidateAudit(
        candidate_id=cid,
        path=path or Path("."),
        earliest_failure=earliest,
        failed_stages=failed,
        notes=notes,
        claimed=candidate,
        passed=passed,
    )


def audit_problem(
    problem_id: str,
    *,
    seed: int = DEFAULT_SEED,
    registry: Registry | None = None,
) -> AuditReport:
    registry = registry or Registry()
    mods = load_problem_modules(problem_id, registry=registry)
    cand_dir = mods.record.path / "candidate_solutions"
    notes: list[str] = []
    if not cand_dir.is_dir():
        return AuditReport(
            problem_id=mods.record.qualified_id,
            n_candidates=0,
            notes=["no candidate_solutions/ directory"],
        )
    paths = sorted(
        [
            p
            for p in cand_dir.iterdir()
            if p.suffix.lower() in {".yaml", ".yml", ".json"} and p.is_file()
        ]
    )
    if not paths:
        return AuditReport(
            problem_id=mods.record.qualified_id,
            n_candidates=0,
            notes=["candidate_solutions/ contains no YAML/JSON files"],
        )
    required = _required_protocol(mods)
    reference_gt1 = None
    try:
        ref = mods.reference.solve(seed=seed)
        reference_gt1 = ref.get("gt1")
        if isinstance(reference_gt1, dict):
            reference_gt1 = reference_gt1.get("value", reference_gt1)
    except Exception as exc:  # noqa: BLE001
        notes.append(f"reference solve failed during audit: {exc}")

    audits = [
        audit_candidate(
            _load_candidate(p),
            required=required,
            reference_gt1=reference_gt1,
            path=p,
        )
        for p in paths
    ]
    if hasattr(mods.invariants, "audit_candidates"):
        extra = mods.invariants.audit_candidates(audits, seed=seed)
        if extra:
            notes.append(str(extra))
    return AuditReport(
        problem_id=mods.record.qualified_id,
        n_candidates=len(audits),
        audits=audits,
        notes=notes,
    )
