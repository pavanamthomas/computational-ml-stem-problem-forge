# Reproducibility

Default seed is `2026`, via `problemforge.rng.get_rng`. Boolean seeds are
rejected because `bool` is a subclass of `int` in Python.

## Commands

```bash
python -m pip install -e .
python -m pytest
python scripts/run_all.py
python -m problemforge list
python -m problemforge validate entity_group_leakage
python -m problemforge run entity_group_leakage
python -m problemforge audit ai_nested_cv_audit
```

`scripts/run_all.py` sets `MPLBACKEND=Agg` when launched as in CI, writes
`outputs/tables/run_summary.csv`, and writes two figures. Those files are
regenerable and are not the source of truth. The source of truth is the
code plus the tests.

## What same seed implies

Same NumPy `Generator` sequence on this NumPy version, therefore the same
DGP draws, therefore the same GT1 for a deterministic solver. Monte Carlo
coverage is stochastic as an estimand and still deterministic given the seed.

## What is not claimed

Bit-stability across NumPy major versions, agreement with R's RNG, or
reproducibility of a GPU kernel.
