# Naive log-sum-exp overflows; max-subtraction does not

Estimand: log Σ exp(x_i) for a float64 vector with entries near 710.
What is not identified: GPU float32 behaviour.

## What problem is being solved?

Evaluate log-sum-exp where the naive formula overflows.

## What assumptions are required?

IEEE-754 float64. Scalar reduction.

## Why was this method chosen?

Max subtraction is the standard identity, not a clip.

## What alternative method could have been used?

`scipy.special.logsumexp` — used as GT3.

## What can go wrong?

Naive `log(sum(exp(x)))`. Subtracting the min. Clipping x.

## How is correctness independently checked?

Translation identity. SciPy. A loop reduction. Naive non-finiteness.

## What can legitimately be concluded?

On this vector the naive formula is non-finite and the stable value matches
SciPy.

## What cannot be concluded?

Stability of unrelated kernels.
