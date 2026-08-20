# Do not trust solver status

SLSQP can report success while complementary slackness is sloppy, and it can
report failure on a problem that is still nearly stationary. The laboratory
assembles residuals from (Q, c, G, h, x).

## What looks right

`message: Optimization terminated successfully.`

## Mechanism

KKT for `min 0.5 x'Qx + c'x s.t. Gx <= h, x >= 0`:
stationarity `Qx + c + G'λ - μ = 0`, primal feasibility, `λ,μ >= 0`, and
`λ_i (Gx-h)_i = 0`, `μ_j x_j = 0`.

## Incorrect candidate

Return `res.fun` without recomputing `0.5 x'Qx + c'x`. Shift x and keep the
success flag.

## What remains unknown

Nonconvex QPs and solver-specific constraint scaling.
