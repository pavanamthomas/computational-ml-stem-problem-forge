from __future__ import annotations

from problemforge.loader import Registry, load_problem_modules
from problemforge.rng import DEFAULT_SEED


def test_reference_and_verifier_files_differ() -> None:
    for rec in Registry():
        ref = (rec.path / "reference_solution.py").read_text(encoding="utf-8")
        ver = (rec.path / "independent_verifier.py").read_text(encoding="utf-8")
        assert ref.strip() != ver.strip(), rec.qualified_id
        assert "def solve" in ref
        assert "def verify" in ver


def test_broken_reference_fails_invariants_when_present() -> None:
    checked = 0
    for rec in Registry():
        mods = load_problem_modules(rec.qualified_id)
        broken_fn = getattr(mods.reference, "deliberately_broken_solve", None)
        if broken_fn is None:
            continue
        broken = broken_fn(seed=DEFAULT_SEED)
        honest = mods.reference.solve(seed=DEFAULT_SEED)
        ver = mods.verifier.verify(seed=DEFAULT_SEED)
        failures = mods.invariants.check_invariants(broken, ver, seed=DEFAULT_SEED)
        assert failures, rec.qualified_id
        assert honest["gt1"] != broken["gt1"] or failures
        checked += 1
    assert checked >= 10
