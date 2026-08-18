# TOML experiment declaration and resolution

A fresh training run begins from one versioned TOML experiment declaration. Its
named sections describe the task vocabulary and positive
`minimum_reason_tokens` requirement, curriculum stages and mixture,
transformer architecture, rollout collection, reward shaping, auxiliary
objective, GRPO update, evaluation schedule, checkpoint schedule, runtime, and
artifact location. The minimum-reasoning requirement defaults to one when
omitted for compatibility. Ordered values such as curriculum stages retain
their declared order.

The reward section independently declares `coverage_coefficient`,
`sterile_repetition_coefficient`, and `sterile_repetition_mode`, whose value is
either `"all"` or `"off_answer"`. Zero coefficients remain explicit resolved
values rather than removing the corresponding diagnostics from experiment
evidence.

Loading converts the TOML data into the typed configurations used by the task,
policy, trainer, curriculum, and evaluation implementations. Unknown keys,
missing required values, invalid types, and mutually incompatible settings are
rejected before a model or run directory is created. Completion budgets must
accommodate the declared minimum reasoning segment plus the shortest
structurally valid answer. Defaults inserted during resolution become explicit
in the resolved configuration.

The run retains both the input TOML file verbatim and a canonical resolved
configuration record. The command line does not provide a parallel collection
of scientific hyperparameter flags. Any supported operational override is
included in the resolved record rather than silently changing the declaration.
