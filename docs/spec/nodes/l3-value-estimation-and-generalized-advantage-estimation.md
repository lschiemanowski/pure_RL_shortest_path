# Value estimation and generalized advantage estimation

A scalar value head reads the transformer's final normalized hidden state at every sampled action position and predicts expected discounted verifier return. It is initialized and optimized only for PPO training; policy sampling, reference scoring, and evaluation remain actor-only. Rollout collection stores detached value predictions alongside action log probabilities.

For a trajectory with terminal completion reward on its final action and zero intermediate rewards, temporal-difference residuals are computed as \(\delta_t=r_t+\gamma V_{t+1}-V_t\), with zero bootstrap after termination. Reverse accumulation \(A_t=\delta_t+\gamma\lambda A_{t+1}\) produces generalized advantages, and \(R_t=A_t+V_t\) produces fixed value targets. Padding never contributes.

Advantages are normalized across real sampled actions when their finite standard deviation exceeds the declared numerical tolerance. Otherwise the original advantages are retained, so an equal nonzero terminal signal can still train the policy.
