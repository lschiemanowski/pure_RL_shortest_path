# Command-line run orchestration

The command line exposes fresh training, checkpoint resumption, and checkpoint evaluation as distinct operations. Fresh training accepts one experiment TOML. Resumption accepts a checkpoint, an optional compatible TOML for an explicitly derived run, and an operational stopping-step override. Evaluation accepts a checkpoint and a separate evaluation TOML. Scientific settings are not duplicated as command-line flags.

Each training step samples curriculum problems, collects unconstrained on-policy completions, verifies them, and packs sampled actions. The selected algorithm then performs either the existing grouped GRPO update or PPO batch preparation followed by a clipped actor-value update. Scheduled reference refresh, representative completions, actor-only evaluation, curriculum gating, checkpoints, and evidence are shared.

Standalone evaluation restores only the selected actor and model configuration. Its TOML declares one or more named graph-problem sets with exact example and generation seeds plus a common greedy or identified stochastic sampling protocol. It writes exact metrics and every completion to a new evaluation directory without constructing an optimizer or value head and without mutating the checkpoint.

Normal completion writes a final checkpoint and terminal event. A first interrupt is deferred to the next completed-step boundary, which writes an interruption checkpoint and event before propagating the interrupt. The package entry point remains a small dispatcher; experiment semantics and artifact behavior reside in the run implementation.

At GRPO steps that coincide with the evaluation interval, orchestration requests objective-gradient diagnostics for that update. Other steps omit this additional measurement.
