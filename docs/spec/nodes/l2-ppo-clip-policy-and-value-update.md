# PPO-Clip policy and value update

For each on-policy token trajectory, a learned scalar value function estimates expected future verifier reward from every sampled action state. Intermediate rewards are zero and the terminal sampled action receives the completion's scalar verifier reward. Generalized advantage estimation uses configurable discount and trace-decay factors to produce per-token advantages and value targets without changing reward semantics.

The policy objective applies PPO's clipped likelihood-ratio surrogate to sampled tokens. A separately weighted value objective optionally clips value changes around rollout-time predictions, and an entropy bonus remains explicit. Advantage normalization preserves a nonzero constant signal when variance is negligible instead of erasing it.

The complete update may also include the existing reference-policy divergence penalty and valid-next-token auxiliary objective as separately configured terms. Multiple optimization epochs and microbatches reuse one fixed on-policy batch; old log probabilities, old values, advantages, and returns remain fixed throughout those epochs. Only the current actor and value function are updated.
