# Log-sum-exp overflow

`exp(710)` overflows float64. `log(sum(exp(x)))` then returns `+inf`.

## What looks right

A softmax implementation that 'works' on small logits in a unit test.

## Mechanism

Max subtraction: `m + log(sum(exp(x-m)))` with `m = max(x)`. Subtracting the
min makes the largest term overflow still.

## Incorrect candidate

Clip logits to [-20, 20] and call the clip a stable log-sum-exp. That changes
the mathematical value.

## What remains unknown

float32 GPU kernels and mixed-precision training.
