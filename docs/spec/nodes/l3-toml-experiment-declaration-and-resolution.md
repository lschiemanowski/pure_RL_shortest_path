# TOML experiment declaration and resolution

A fresh training run begins from one versioned TOML experiment declaration. Its root `algorithm` selects `"grpo"` or `"ppo"` and defaults to GRPO when omitted. Named sections describe task and minimum-reasoning semantics, curriculum, transformer architecture, rollout collection, reward shaping, shared optimizer settings, GRPO settings, PPO settings, evaluation, checkpoints, runtime, and artifacts. The resolved record retains both algorithm sections so comparisons remain explicit.

The reward section independently declares `coverage_coefficient`, `sterile_repetition_coefficient`, and `sterile_repetition_mode`, whose value is either `"all"` or `"off_answer"`. Zero coefficients remain explicit resolved values rather than removing the corresponding diagnostics from experiment evidence.

Loading converts the TOML data into typed configurations. It validates the selected algorithm, applies PPO defaults for clipping, discount, trace decay, value and entropy coefficients, epochs, microbatching, and disabled all-zero-reward policy suppression, and applies the GRPO group-size compatibility rule only when GRPO is selected. The PPO section may enable `suppress_zero_reward_policy_updates` as a boolean. Unknown keys, invalid types, impossible completion budgets, and incompatible settings are rejected before a model or run directory is created.

The run retains both the input TOML file verbatim and a canonical resolved configuration record. The command line does not provide a parallel collection of scientific hyperparameter flags. Any supported operational override is included in the resolved record rather than silently changing the declaration.
