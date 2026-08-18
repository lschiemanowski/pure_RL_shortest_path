# Transformer policy

The transformer policy is the trainable actor in the system. It contains no task-specific verifier, shortest-path oracle, demonstrated solution, or mechanism for looking up a correct next action.

Its transformer model owns the declared decoder-only causal architecture, random initialization, and learned parameters that identify the current actor. During PPO training it may expose final normalized token representations to a separate training-only value function; policy sampling and evaluation remain actor-only.

Its token-prefix scoring interface maps serialized token prefixes to next-token scores over the complete task vocabulary. Training and evaluation use the same scoring interface; the invoking process owns sampling, token constraints, and termination behavior.
