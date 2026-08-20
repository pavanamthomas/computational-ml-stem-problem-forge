"""Command-line interface for the problem forge."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Sequence

from problemforge.audit import audit_problem
from problemforge.loader import Registry
from problemforge.rng import DEFAULT_SEED
from problemforge.runner import run_problem
from problemforge.validate import validate_all, validate_problem


def _print(msg: str) -> None:
    sys.stdout.write(msg + "\n")


def cmd_list(_: argparse.Namespace) -> int:
    registry = Registry()
    rows = list(registry)
    _print(f"{len(rows)} problems")
    _print(f"{'id':<36} {'difficulty':<14} {'domain':<14} title")
    _print("-" * 88)
    for rec in rows:
        _print(
            f"{rec.qualified_id:<36} {rec.difficulty:<14} {rec.domain:<14} {rec.title}"
        )
    return 0


def cmd_validate(ns: argparse.Namespace) -> int:
    if ns.problem_id in {"all", "*"}:
        reports = validate_all()
    else:
        reports = [validate_problem(ns.problem_id)]
    rc = 0
    for rep in reports:
        status = "OK" if rep.ok else "FAIL"
        _print(f"{status}  {rep.problem_id}")
        for item in rep.missing_files + rep.schema_errors + rep.notes:
            _print(f"    - {item}")
            rc = 1
    return rc


def cmd_run(ns: argparse.Namespace) -> int:
    report = run_problem(ns.problem_id, seed=ns.seed)
    status = "OK" if report.ok else "FAIL"
    _print(f"{status}  {report.problem_id}  seed={report.seed}")
    _print(f"  GT1 reference: {report.reference.get('gt1')}")
    _print(f"  GT1 verifier:  {report.verifier.get('gt1')}")
    for check in report.unit_checks:
        mark = "pass" if check.passed else "FAIL"
        _print(f"  {check.unit_id} [{mark}] {check.detail}")
    for fail in report.invariant_failures:
        _print(f"  invariant: {fail}")
    for err in report.schema_errors:
        _print(f"  schema: {err}")
    return 0 if report.ok else 1


def cmd_audit(ns: argparse.Namespace) -> int:
    report = audit_problem(ns.problem_id, seed=ns.seed)
    _print(f"{report.problem_id}: {report.n_candidates} candidate(s)")
    for note in report.notes:
        _print(f"  note: {note}")
    for aud in report.audits:
        if aud.passed:
            _print(f"  {aud.candidate_id}: PASS")
        else:
            _print(
                f"  {aud.candidate_id}: earliest_failure={aud.earliest_failure}"
            )
            for n in aud.notes:
                _print(f"      - {n}")
    if report.n_candidates == 0:
        return 0
    # An audit run is successful if every wrong candidate is diagnosed.
    undiagnosed = [
        a for a in report.audits if (not a.passed) and a.earliest_failure is None
    ]
    return 1 if undiagnosed else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="problemforge",
        description="Computational ML/STEM problem forge",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_list = sub.add_parser("list", help="list registered problems")
    p_list.set_defaults(func=cmd_list)

    p_val = sub.add_parser("validate", help="validate problem.yaml and files")
    p_val.add_argument("problem_id")
    p_val.set_defaults(func=cmd_validate)

    p_run = sub.add_parser("run", help="run reference + independent verifier")
    p_run.add_argument("problem_id")
    p_run.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p_run.set_defaults(func=cmd_run)

    p_audit = sub.add_parser("audit", help="audit candidate solutions")
    p_audit.add_argument("problem_id")
    p_audit.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p_audit.set_defaults(func=cmd_audit)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    ns = parser.parse_args(argv)
    func = getattr(ns, "func")
    return int(func(ns))


def dumps_report(obj: Any) -> str:
    return json.dumps(obj, default=str, indent=2)
