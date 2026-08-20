"""Computational ML/STEM problem forge.

This package loads problem laboratories under ``problems/``, validates their
specifications, runs a reference solver against an independent verifier, and
audits candidate solutions for the earliest substantive protocol failure.

Nothing here is an empirical finding about an observational population. Ground
truth is defined on a stated data-generating process or a stated numerical
identity.
"""

from __future__ import annotations

from problemforge.rng import DEFAULT_SEED, get_rng
from problemforge.schema import ProblemSpec

__version__ = "0.1.0"

__all__ = ["DEFAULT_SEED", "ProblemSpec", "get_rng"]
