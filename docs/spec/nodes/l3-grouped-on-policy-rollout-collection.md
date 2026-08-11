# Grouped on-policy rollout collection

For each supplied training problem, the fixed current policy generates \(K\)
independent completions. Those completions are retained adjacently as one
comparison group, and no parameter update occurs until collection of the whole
rollout batch is complete.

At every generated position, the next token is sampled categorically from the
complete softmax of the current policy's raw full-vocabulary logits. Training
requires temperature one and top-p one; the configuration rejects values that
would rescale or truncate this distribution. Sampling applies no graph-legality
mask, forced delimiter, verifier intervention, or post-hoc token edit. A
completion ends only when it samples the termination symbol or exhausts the
declared maximum length; reaching the length limit does not append an artificial
termination symbol.

Each retained sample identifies its problem, comparison group, exact completion
tokens, and whether termination was sampled. Batching, left padding, causal
attention caching, and seeded categorical sampling may alter execution but not
the defined distribution or group order.
