"""Seeded NumPy generators.

What problem is being solved?
    Reproducible draws for every DGP and Monte Carlo arm in this laboratory.

What assumptions are required?
    Callers pass an integer seed. Boolean values are rejected because ``bool``
    is a subclass of ``int`` in Python and would otherwise silently coerce.

Why was this method chosen?
    ``numpy.random.Generator`` is the current NumPy RNG API. A single helper
    keeps the default seed in one place.

What alternative method could have been used?
    ``RandomState``, Python's ``random`` module, or a bit-generator spawned
    from ``SeedSequence`` for independent streams. Independent streams are
    created by passing a different integer, not by advancing a shared generator.

What can go wrong?
    Re-using one generator across two arms that should be independent couples
    their draws. Hashing strings into seeds is not used here.

How is correctness independently checked?
    Tests assert that the same seed regenerates the same first draw, and that
    ``bool`` and negative seeds raise ``ValueError``.

What can legitimately be concluded?
    Given the same seed, the same Generator sequence is produced on this NumPy
    version.

What cannot be concluded?
    Cross-library reproducibility (R, MATLAB) or bit-stability across NumPy
    major versions is not claimed.
"""

from __future__ import annotations

import numpy as np
from numpy.random import Generator

DEFAULT_SEED = 2026


def get_rng(seed: int | None = None) -> Generator:
    """Return a ``Generator`` bound to ``seed`` (default ``2026``)."""
    if seed is None:
        seed = DEFAULT_SEED
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise ValueError(f"seed must be a non-boolean integer, got {type(seed)!r}")
    seed_int = int(seed)
    if seed_int < 0:
        raise ValueError("seed must be non-negative")
    return np.random.default_rng(seed_int)
