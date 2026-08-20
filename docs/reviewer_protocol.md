# Reviewer protocol

Review the specification before the number.

1. Name the estimand in one sentence. If you cannot, the problem is not ready.
2. Name the DGP or identity. If the DGP is 'sklearn's default', reject it.
3. Check that GT2 is not a copy of GT1.
4. Open `independent_verifier.py` and confirm it is not the same file as the
   reference (the test suite hashes this; still read it).
5. Confirm a deliberate-failure path exists and that invariants fail on it.
6. For Monte Carlo claims, confirm the tolerance is an SE rule.
7. For leakage claims, confirm a sentinel that does not reuse the primary
   metric.
8. For candidate audits, confirm earliest-stage diagnosis, not only a score
   delta.
9. Run `python -m problemforge validate <id>` and `python -m problemforge run <id>`.
10. Record remaining unknowns as limitations, not as future marketing.

Reject causal language for a predictive metric. Reject hard-coded GT1.
Reject empty `failure_modes.md`.
