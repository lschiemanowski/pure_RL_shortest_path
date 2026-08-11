# Rollout sampling

Rollout sampling receives training problems supplied by the experiment protocol
and produces a declared number of policy completions for each problem.
Completions for the same problem form a comparison group. Sampling uses declared
maximum completion length and terminates when the policy emits the termination
symbol or reaches that limit. Training fixes temperature and top-p to one, so
each token is drawn from the policy's complete softmax distribution. Top-p is
also called nucleus sampling: values below one would retain only the smallest
high-probability token set reaching that cumulative probability, but training
does not apply this truncation.

Token selection operates on the policy's complete next-token scores without
task-specific legality masks, forced tokens, or post-hoc edits. Graph
verification occurs only after a completion has been sampled and cannot
influence token selection within that rollout. Batching and cached evaluation
may improve execution efficiency but do not change the sampling semantics.
