# Validation-gated frontier advancement

Validation measures the policy on freshly generated problems from the current
frontier stage. Validation uses its own random stream, applies the declared
evaluation sampling protocol, and does not update the policy or contribute
examples to subsequent training rollouts.

After each scheduled validation, the protocol compares shortest-path success
with the configured advancement threshold. A qualifying result increments the
frontier's advancement streak; a nonqualifying result resets the streak to zero.
Once the streak reaches the configured patience and a succeeding stage exists,
the frontier advances by exactly one stage and the streak resets.

Advancement immediately recenters the asymmetric training distribution on the
new frontier. There is no current/next-stage interpolation fraction,
maximum-mixture condition, or separate mixture-readiness gate. Earlier stages
remain available through the backward tail, while future exposure is controlled
entirely by the sharply decaying forward tail.

The frontier never retreats and never advances beyond the final stage. Every
validation decision retains its example count, frontier, shortest-path success,
threshold, qualification result, and resulting streak. Every advancement record
identifies the completed frontier, new frontier, training step, and validation
result that triggered the transition.

A validation that does not advance the frontier still records the decision and leaves the training distribution centered on the existing frontier.
