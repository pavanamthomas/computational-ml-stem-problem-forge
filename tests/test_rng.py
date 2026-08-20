from __future__ import annotations

import pytest

from problemforge.rng import DEFAULT_SEED, get_rng


def test_default_seed_is_2026() -> None:
    assert DEFAULT_SEED == 2026


def test_same_seed_same_draw() -> None:
    a = get_rng(2026).normal()
    b = get_rng(2026).normal()
    assert a == b


def test_different_seed_different_draw() -> None:
    a = get_rng(2026).normal()
    b = get_rng(2027).normal()
    assert a != b


def test_bool_seed_rejected() -> None:
    with pytest.raises(ValueError):
        get_rng(True)  # type: ignore[arg-type]


def test_negative_seed_rejected() -> None:
    with pytest.raises(ValueError):
        get_rng(-1)


def test_none_uses_default() -> None:
    assert get_rng(None).normal() == get_rng(DEFAULT_SEED).normal()
