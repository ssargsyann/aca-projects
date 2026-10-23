# Sampling and numerical contracts

All filters are nonmutating and operate along the final vocabulary axis, including batched inputs. Logits must be floating, nonempty, contain at least one finite entry per row, and contain neither NaN nor positive infinity. Negative infinity is permitted for masked candidates.

## Temperature

For T>0, compute shifted float64 scores `(logits - row_max) / T` and normalize with softmax after filtering. Subtracting the maximum before dividing avoids positive overflow at tiny T; negligible candidates may legitimately underflow to zero probability. T must be finite and nonnegative. **T=0 is an explicit deterministic argmax extension**, with the first maximum ID winning ties; it is not a near-zero shortcut. Negative/NaN/infinite temperatures fail. Transformers' temperature warper itself requires strictly positive T, so reference equivalence is tested for T>0 only.

## Top-K

K must be an integer >=0. K=0 disables filtering; K>=vocabulary size keeps the whole current support. Otherwise discard scores strictly below the K-th largest score. Ties at the threshold are retained, so there may be more than K eligible tokens (and K=1 can leave tied maxima). Existing masked entries remain masked.

## Top-P

P must lie in (0,1]; P=1 disables filtering. Sort scores descending with stable token-ID tie order, calculate their softmax/CDF, and keep the smallest sorted prefix whose CDF **reaches or exceeds P**, including the boundary token. At least one candidate survives. Exact-CDF ties use that explicit minimal-prefix rule; implementations with a shifted `CDF > P` mask can differ on exact boundaries. Finite-precision cutoffs and ties can also change discrete membership. Tests cover the exact uniform boundary and match the installed reference on nonboundary random logits.

## Composition and interpretation

Sampling applies temperature, then Top-K, then Top-P, followed by final normalization and one multinomial draw. Top-P's CDF is over the temperature-scaled distribution conditioned on Top-K survivors. For example, the measured first step has 40 Top-P candidates versus 12 after K=50 then P=0.9. Applying P before temperature, as the historical loop did, defines a different policy.

Greedy ignores stochastic filters and uses raw argmax. Its recorded processed-policy entropy is zero by construction; that does not mean the model was certain about a factual answer. Eligibility counts finite scores, while positive-probability counts reflect possible numerical underflow. Entropy is in natural-log units (nats). Selected sampling probabilities are normalized after filtering, whereas `base_model_logprob` comes from the unmodified model distribution.

The same first-step-logit comparison isolates policy transformations. Later steps see different generated contexts, so their descriptive entropy averages are not controlled estimates of a universal policy effect. Token distinctness and model likelihood do not establish correctness or human-rated quality.

Primary references: [Transformers generation utilities](https://huggingface.co/docs/transformers/v4.57.1/en/internal/generation_utils), [The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751). The experiment does not reproduce that paper's evaluation.
