# PPO batch construction and clipped update

A packed PPO batch contains prompts, sampled actions, action masks, rollout-time log probabilities and values, generalized advantages, fixed returns, and valid-next sets. It is collected once from the current actor-value model and is not regenerated during optimization epochs.

For every sampled token, the policy loss is the negative minimum of the unclipped likelihood-ratio advantage and the ratio clipped to the declared interval. When configured and every scalar completion reward in the fixed rollout batch is zero, the policy loss is instead zero for every optimization epoch and microbatch; value loss, entropy, reference-policy divergence, and valid-next auxiliary loss remain independently active according to their coefficients. The value loss is the larger squared error between the return and either the new prediction or, when value clipping is enabled, the rollout prediction plus a clipped change. Entropy is averaged over sampled-token distributions. Clip fraction and explained variance remain observable.

The total loss combines policy loss, the weighted value loss, negative weighted entropy, reference-policy divergence, and valid-next auxiliary loss. Declared epochs and microbatches reuse fixed rollout statistics, gradients are clipped before optimizer steps, and only the current actor and value head are updated.
