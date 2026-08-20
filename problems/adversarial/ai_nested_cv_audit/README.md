# Earliest protocol failure in nested-CV candidates

Estimand: nested grouped outer accuracy, and the earliest failed protocol
stage of three written candidates. DGP: entity intercepts. What is not
identified: an auditor for arbitrary source code.

## What problem is being solved?

Diagnose subtle nested-CV mistakes in stage order, not as a single wrong
scalar.

## What assumptions are required?

GroupKFold. Scaling inside the pipeline. Nested outer accuracy as the report.

## Why was this method chosen?

These three mistakes survive a glance at a plausible accuracy. Stage order
is specified so that a double fault is not scored as 'inner CV'.

## What alternative method could have been used?

A weighted rubric. That hides which failure is first.

## What can go wrong?

Scoring only `claimed_score`. Reordering stages after seeing the candidates.

## How is correctness independently checked?

Manual grouped nested loops. Known earliest failures on the shipped YAML.

## What can legitimately be concluded?

The three candidates fail at split, scaling, and inner-CV reporting.

## What cannot be concluded?

That every Python leak would be caught.
