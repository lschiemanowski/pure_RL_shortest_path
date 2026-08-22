# Append-only evidence and comparable records

Training writes schema-versioned JSON Lines records by appending complete records and synchronizing them to disk. Every event identifies its run, completed step, resolved-configuration digest, and selected algorithm. Shared records retain rollout counts, verifier reward components, shaping diagnostics, sampling settings, and optimizer diagnostics. GRPO records its policy, divergence, valid-next, and zero-variance metrics; PPO additionally records value loss, entropy, clip fraction, explained variance, and their coefficients.

For sterile-repetition shaping, training and representative-completion evidence retains both the all-transition and off-answer rates, the selected mode and coefficient, and the resulting applied penalty. Both diagnostics remain present when the coefficient is zero or only one mode affects reward.

Independent evaluation records retain the policy identity, graph-problem and problem-set identities, sampling protocol, declared minimum-reasoning requirement, and exact format, valid-path, and shortest-path counts and rates. Curriculum decisions and reference-policy updates are distinct lifecycle events rather than being inferred from nearby measurements.

At the declared interval, representative completions are selected by taking the first sampled completion for each of the first declared number of problems. Each record preserves graph and query semantics, prompt and completion tokens and text, parsed and verified facts, and separate reward components. This deterministic rule does not select examples by success.

Configuration comparison recursively reports stable field paths whose values differ. It exposes experimental differences without deciding that the corresponding evidence is scientifically comparable.

When requested, a GRPO training record stores one full-batch pre-clipping diagnostic per update epoch: norms for the policy objective, coefficient-weighted KL term, coefficient-weighted valid-next term, and their combined gradient, plus all pairwise cosine similarities.
