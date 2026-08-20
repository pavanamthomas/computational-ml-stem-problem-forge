# A 0.90 AUC that cannot come from X

The wrong score is a noisy function of the evaluation labels. It is not
`score = y`, which would be an AUC of 1 and would not pass as a plausible
modelling result.

## What looks right

Held-out ROC-AUC ≈ 0.90. The number is in the range people screenshot.

## Mechanism

`0.22 * p_hat(X) + 0.78 * y + noise`, mapped through a sigmoid. Permuting the
permitted columns barely changes the leaked AUC. Permuting X collapses the
legitimate AUC toward 0.5.

## Incorrect candidate

Publish the leaked number. Or 'test' leakage by shuffling y, which destroys
the leak and the legitimate signal together.

## What remains unknown

Other contamination channels (row identity, test-set scaling).
