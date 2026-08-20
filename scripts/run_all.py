"""Run every problem laboratory and write regenerable artefacts.

All numbers printed here are computed from stated DGPs or numerical
identities. They are not empirical findings about a real population.

Usage, from the repository root::

    python scripts/run_all.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from problemforge.audit import audit_problem
from problemforge.loader import Registry
from problemforge.runner import run_all
from problemforge.validate import validate_all


def _print_header(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def _plot_leakage_gap(report, fig_path: Path) -> None:
    naive = report.reference["diagnostics"]["naive_cv_acc"]
    grouped = report.reference["diagnostics"]["grouped_cv_acc"]
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.bar(["Naive KFold", "GroupKFold"], [naive, grouped], color=["#b45309", "#1d4ed8"])
    ax.set_ylabel("CV accuracy")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Entity intercept DGP: naive vs grouped CV")
    ax.axhline(0.5, color="grey", linewidth=0.8, linestyle="--")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=140)
    plt.close(fig)


def _plot_reliability(report, fig_path: Path) -> None:
    diag = report.reference["diagnostics"]
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="identity")
    ax.plot(
        diag["model_a_bin_pred"],
        diag["model_a_bin_true"],
        marker="o",
        label=f"A ECE={diag['ece_a']:.3f} AUC={diag['auc_a']:.3f}",
    )
    ax.plot(
        diag["model_b_bin_pred"],
        diag["model_b_bin_true"],
        marker="s",
        label=f"B ECE={diag['ece_b']:.3f} AUC={diag['auc_b']:.3f}",
    )
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Mean observed frequency")
    ax.set_title("Discrimination is not calibration")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=140)
    plt.close(fig)


def main() -> None:
    fig_dir = ROOT / "outputs" / "figures"
    tab_dir = ROOT / "outputs" / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)

    _print_header("Validate specifications")
    validations = validate_all()
    for v in validations:
        mark = "OK" if v.ok else "FAIL"
        print(f"  {mark}  {v.problem_id}")
        for item in v.missing_files + v.schema_errors:
            print(f"      {item}")
    if any(not v.ok for v in validations):
        raise SystemExit("specification validation failed")

    _print_header("Run reference + independent verifier")
    reports = run_all()
    rows = [r.summary_row() for r in reports]
    frame = pd.DataFrame(rows)
    out_csv = tab_dir / "run_summary.csv"
    frame.to_csv(out_csv, index=False)
    for r in reports:
        mark = "OK" if r.ok else "FAIL"
        print(f"  {mark}  {r.problem_id}  GT1={r.reference.get('gt1')!r}")
        if not r.ok:
            for c in r.unit_checks:
                if not c.passed:
                    print(f"      {c.unit_id}: {c.detail}")
            for f in r.invariant_failures:
                print(f"      invariant: {f}")
            for s in r.schema_errors:
                print(f"      schema: {s}")

    _print_header("Candidate audits")
    registry = Registry()
    for rec in registry:
        aud = audit_problem(rec.qualified_id)
        print(f"  {aud.problem_id}: {aud.n_candidates} candidate(s)")
        for item in aud.audits:
            if item.passed:
                print(f"      {item.candidate_id}: PASS")
            else:
                print(
                    f"      {item.candidate_id}: earliest={item.earliest_failure}"
                )

    by_id = {r.problem_id: r for r in reports}
    leak = by_id.get("ml/entity_group_leakage")
    if leak is not None and "diagnostics" in leak.reference:
        _plot_leakage_gap(leak, fig_dir / "entity_group_leakage.png")
        print(f"wrote {fig_dir / 'entity_group_leakage.png'}")
    cal = by_id.get("ml/calibration_vs_discrimination")
    if cal is not None and "diagnostics" in cal.reference:
        _plot_reliability(cal, fig_dir / "calibration_vs_discrimination.png")
        print(f"wrote {fig_dir / 'calibration_vs_discrimination.png'}")

    print()
    print(f"wrote {out_csv}")
    if any(not r.ok for r in reports):
        raise SystemExit("one or more problems failed verification")


if __name__ == "__main__":
    main()
