# Experiment protocol and evidence

The experiment protocol and evidence system turns task, policy, and trainer
configurations into reproducible training and evaluation runs. It records the
task and vocabulary semantics, model and learning configuration, random seeds,
source revision, run lineage, metrics, checkpoints, and representative
completions needed to resume or audit a run.

Evaluation measures a selected policy on fresh or fixed independently generated
problems without updating it. It reports formatting, valid-path, and
shortest-path performance separately and retains the identities of the policy,
problem distribution, and evaluation protocol behind each result. Comparisons
across curriculum stages, graph scales, model variants, and learning signals use
compatible evidence and expose rather than conceal relevant experimental
differences.
