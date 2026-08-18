# On-policy rollout collection

For each supplied training problem, the fixed current actor generates the configured number of independent completions before any parameter update. GRPO retains adjacent completions as one comparison group; PPO treats them as independent trajectories even when execution keeps them adjacent.

At every generated position, the next token is sampled categorically from the complete softmax of the current actor's raw full-vocabulary logits. Training requires temperature one and top-p one; the configuration rejects values that would rescale or truncate this distribution. Sampling applies no graph-legality mask, forced delimiter, verifier intervention, or post-hoc token edit. A completion ends only when it samples the termination symbol or exhausts the declared maximum length; reaching the length limit does not append an artificial termination symbol.

Each retained sample identifies its problem, group position, exact completion tokens, and whether termination was sampled. After verification and packing, PPO snapshots detached action log probabilities and scalar values before its first optimization epoch. Batching, left padding, causal attention caching, and seeded categorical sampling may alter execution but not the defined distribution or sample order.
