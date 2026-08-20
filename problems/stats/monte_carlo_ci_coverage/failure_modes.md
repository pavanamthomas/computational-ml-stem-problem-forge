# Coverage is not a point mass at 0.95

With R=2000, SE of a Bernoulli(0.95) mean is about 0.0049. A realised phat
of 0.946 is ordinary.

## What looks right

A test `assert coverage == 0.95`. It is the wrong test.

## Mechanism

Coverage is an expectation. The Monte Carlo estimator has variance
p(1-p)/R. The laboratory uses that SE as the tolerance.

## Incorrect candidate

R=50, phat=0.90, declare the t-interval broken.

## What remains unknown

Coverage under a t_3 parent, or for a bootstrap interval.
