# Pure-RL shortest-path research system

The system is a local command-line research environment for controlled
experiments in which a transformer initialized from scratch is trained by
reinforcement learning to find shortest paths in synthetic connected,
undirected, unweighted graphs. A researcher launches training and evaluation
runs and later inspects their metrics, checkpoints, and generated solutions.

Reinforcement learning is the model-training mechanism. During training, the
current policy receives serialized graphs with source–target queries and freely
samples reasoning traces and final paths. The system evaluates those sampled
completions using automatically computed graph feedback and uses that feedback
to update the policy. Training uses no pretrained weights, demonstrated
solutions, teacher reasoning, or supervised target completions. Verifier
reward, reward shaping, and auxiliary graph-derived objectives are individually
named and configurable, so each experiment states precisely which learning
signals it uses.

Each run preserves enough information to reproduce and audit its result,
including task and vocabulary semantics, configuration, random seeds,
checkpoints, metrics, and representative completions. Evaluation uses fresh or
fixed independently generated examples and reports formatting success,
valid-path success, and shortest-path success separately. Comparisons across
curriculum stages, graph scales, model variants, and learning signals avoid
known confounds such as introducing previously untrained node labels only at
larger graph sizes.
