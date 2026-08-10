# Pure-RL shortest-path research system

The system is a local command-line research environment for controlled
experiments into whether a transformer can learn to find shortest paths through
reinforcement learning. A researcher launches training and evaluation runs and
later inspects their metrics, checkpoints, and generated solutions.

Experiments use synthetic connected, undirected, unweighted graphs with
source–target queries. Each problem is serialized under declared task and
vocabulary semantics, and generated reasoning traces and final paths can be
assessed exactly for format correctness, path validity, path length, and
optimality. A transformer policy initialized from scratch receives the
serialized problem and freely generates its completion without task-specific
token masks, forced actions, or access to a shortest-path oracle.

Reinforcement learning is the model-training mechanism. The current policy
samples its own completions, the system derives declared graph feedback from
them, and that feedback is used to update the policy. Training uses no
pretrained weights, demonstrated solutions, teacher reasoning, or supervised
target completions. Verifier reward, reward shaping, and auxiliary graph-derived
objectives are individually named and configurable, so each experiment states
precisely which learning signals it uses.

Each run preserves enough information to reproduce and audit its result,
including task and vocabulary semantics, configuration, random seeds, source
revision, run lineage, checkpoints, metrics, and representative completions.
Evaluation measures selected policies without updating them, using fresh or
fixed independently generated problems, and reports formatting success,
valid-path success, and shortest-path success separately. Comparisons across
curriculum stages, graph scales, model variants, and learning signals use
compatible evidence and expose known confounds such as introducing previously
untrained node labels only at larger graph sizes.
