from __future__ import annotations

from problemforge.loader import Registry, load_problem_modules
from problemforge.rng import DEFAULT_SEED


def test_same_seed_regenerates_gt1_for_every_problem() -> None:
    for rec in Registry():
        mods = load_problem_modules(rec.qualified_id)
        a = mods.reference.solve(seed=DEFAULT_SEED)["gt1"]
        b = mods.reference.solve(seed=DEFAULT_SEED)["gt1"]
        assert a == b, rec.qualified_id


def test_second_seed_changes_stochastic_gt1() -> None:
    """Problems whose GT1 depends on draws must move when the seed moves."""
    movers = [
        "ml/entity_group_leakage",
        "numerical/logsumexp_stability",
        "stats/monte_carlo_ci_coverage",
    ]
    for pid in movers:
        mods = load_problem_modules(pid)
        a = mods.reference.solve(seed=DEFAULT_SEED)["gt1"]
        b = mods.reference.solve(seed=DEFAULT_SEED + 1)["gt1"]
        assert a != b, pid
