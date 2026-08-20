# Earliest failure, not the longest list

A candidate that randomly splits grouped rows and also quotes inner CV is a
**split** failure first. The numeric gap is stage five.

## Shipped candidates

1. `c1_random_split_despite_groups` — KFold despite entity intercepts.
2. `c2_scaling_outside_cv` — StandardScaler fit on all rows, grouped split.
3. `c3_inner_cv_as_final` — grouped nested protocol, but `best_score_` reported.

## Incorrect meta-candidate

An auditor that returns 'wrong number' for all three.

## What remains unknown

Free-form Python solutions that do not declare a protocol block.
