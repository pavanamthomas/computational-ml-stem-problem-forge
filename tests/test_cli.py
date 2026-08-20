from __future__ import annotations

from problemforge.cli import main
from problemforge.loader import Registry
from problemforge.runner import run_problem


def test_cli_list(capsys) -> None:
    rc = main(["list"])
    captured = capsys.readouterr()
    assert rc == 0
    assert "entity_group_leakage" in captured.out
    assert str(len(list(Registry()))) in captured.out.split()[0] or "12 problems" in captured.out


def test_cli_validate_all(capsys) -> None:
    rc = main(["validate", "all"])
    assert rc == 0
    captured = capsys.readouterr()
    assert "OK" in captured.out


def test_cli_run_logsumexp(capsys) -> None:
    rc = main(["run", "logsumexp_stability"])
    captured = capsys.readouterr()
    assert rc == 0, captured.out
    assert "GT1" in captured.out


def test_cli_audit_without_candidates(capsys) -> None:
    rc = main(["audit", "logsumexp_stability"])
    assert rc == 0
    captured = capsys.readouterr()
    assert "candidate" in captured.out.lower()


def test_cli_audit_nested(capsys) -> None:
    rc = main(["audit", "ai_nested_cv_audit"])
    captured = capsys.readouterr()
    assert rc == 0, captured.out
    assert "split_integrity" in captured.out
    assert "preprocessing_isolation" in captured.out
    assert "estimation_protocol" in captured.out


def test_run_logsumexp_ok() -> None:
    report = run_problem("numerical/logsumexp_stability")
    assert report.ok
    assert len(report.unit_checks) == 3
