# Data policy

This repository is a computational laboratory. It does not ship observational
microdata, survey extracts, or proprietary files.

## What is used

Every numerical claim is generated in code from a stated data-generating
process or a stated numerical identity. Randomness is controlled through
`problemforge.rng.get_rng` (`numpy.random.Generator`, default seed `2026`).

No file in `data/` is required. No download script is required.

## What is not claimed

Simulated coverage, leakage gaps, AUC values, and residuals describe
procedures under a known DGP or identity. They are not estimates for a real
population, a clinic, a trading book, or a deployed model.

## Regeneration

Figures and tables under `outputs/` are disposable. They are written by
`python scripts/run_all.py` and are ignored by git except for `.gitkeep`
placeholders. A clean clone plus the commands in the README regenerates them.

## Third-party code

The package depends on NumPy, pandas, SciPy, scikit-learn, matplotlib,
PyYAML, pydantic, and pytest under their respective licences. This repository
does not copy textbook exam questions or copyrighted worked examples into
`problems/` or `docs/`.
