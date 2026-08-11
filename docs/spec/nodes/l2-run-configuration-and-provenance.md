# Run configuration and provenance

A run configuration resolves the task distribution and vocabulary, transformer
architecture, rollout and optimization settings, reward and auxiliary-learning
signals, curriculum, evaluation schedule, and artifact locations before
training begins. Invalid or mutually incompatible settings are rejected before
model updates occur.

Each run records its resolved configuration, random seeds, source revision, and
a stable run identity. Independent random streams are used for training-problem
generation and evaluation-problem generation. A resumed or derived run records
its relationship to the originating run and checkpoint rather than presenting
itself as an unrelated experiment.
