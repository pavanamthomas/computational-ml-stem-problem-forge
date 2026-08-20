# Convex QP checked by KKT residuals, not solver status

Estimand: primal objective and KKT residual of a 4-variable SPD QP.
What is not identified: SLSQP's global reliability.

## What problem is being solved?

Solve a strictly convex QP and verify optimality conditions independently.

## What assumptions are required?

Q SPD, Gx <= h, x >= 0.

## Why was this method chosen?
The KKT map is small enough to assemble. That is the point.

## What alternative method could have been used?

An interior-point QP solver. Same KKT object.

## What can go wrong?

Trusting `success=True`. Not recomputing the objective.

## How is correctness independently checked?

NNLS recovery of multipliers. Feasibility of inequalities and bounds.

## What can legitimately be concluded?

On this instance x is feasible and approximately stationary.

## What cannot be concluded?

Anything about nonconvex programs.
