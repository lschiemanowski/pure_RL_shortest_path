# Run configuration and provenance

A run configuration resolves the task distribution, vocabulary, completion protocol including its minimum reasoning length, transformer architecture, rollout settings, selected optimization algorithm and its algorithm-specific settings, reward and auxiliary-learning signals, curriculum, evaluation schedule, and artifact locations before training begins. Invalid or mutually incompatible settings are rejected before model updates occur. GRPO remains the default for configurations that omit an algorithm selection.

Each run records its resolved configuration, random seeds, source revision, and a stable run identity. Independent random streams are used for training-problem generation and evaluation-problem generation. A resumed or derived run records its relationship to the originating run and checkpoint rather than presenting itself as an unrelated experiment.

Sterile-repetition shaping is declared by an independently named nonnegative coefficient and an explicit choice between all-transition and off-answer mode. A zero coefficient disables its effect on reward without disabling either diagnostic.
