# Problem authoring standard

A problem is a laboratory, not a quiz item. It has a named estimand, a named
DGP or identity, a reference solver, an independent verifier, and tests that
fail when the wrong procedure is used.

## Directory

```text
problems/<domain>/<problem_id>/
  problem.yaml
  README.md
  reference_solution.py
  independent_verifier.py
  invariants.py
  tests/test_*.py
  failure_modes.md
  expected_schema.json
  candidate_solutions/   # optional except for audit problems
```

`problem.yaml` is validated strictly by `problemforge.schema.ProblemSpec`.
Extra keys are forbidden. Ground-truth units must be exactly GT1, GT2, GT3
in that order, with kinds `numerical`, `invariant`, and
`independent_check` or `protocol`.

## The eight questions

`README.md` and the reference module docstring answer, in this order:

1. What problem is being solved?
2. What assumptions are required?
3. Why was this method chosen?
4. What alternative method could have been used?
5. What can go wrong?
6. How is correctness independently checked?
7. What can legitimately be concluded?
8. What cannot be concluded?

Name the estimand. Name the DGP. Name what is not identified.

## Independence

`independent_verifier.py` must not be a copy of `reference_solution.py`.
GT3 is a different code path: a different splitter construction, a
closed-form identity, a different solver, or a different reduction. Sharing
the DGP formula is allowed; sharing the solver is not.

## Difficulty

`L1`–`L4` are computational intensity and subtlety, not a grade. Recruiter-facing
flagship cases in this corpus are `EXPERT` and `ADVERSARIAL`.

## What is forbidden

- Placeholder TODOs in shipped solvers.
- Hard-coded numerical answers that should be computed.
- Treating simulated draws as empirical.
- Causal language for a predictive metric.
- Marketing adjectives in the problem statement.

The registry discovers `problem.yaml` recursively. No central index file is
required to add the 101st problem.
