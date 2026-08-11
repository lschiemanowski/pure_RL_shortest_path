# Independent evaluation

An evaluation identifies the policy checkpoint, graph-problem configuration,
completion-sampling settings, number of examples, and either an independent
random seed or a fixed evaluation-set identity. Evaluation problems are not
reused as training rollouts, and evaluation does not update the policy,
optimizer, reference policy, or curriculum state.

Completions are interpreted and verified using the same declared task semantics
as training. Results report the evaluation denominator and formatting success,
valid-path success, and shortest-path success separately. Path-length and
reasoning-trace diagnostics may be reported in addition, but they do not replace
those primary outcomes.

Optional intervention evaluations, such as replacing a generated reasoning
trace before producing the final answer, are identified separately from
ordinary evaluation. Their intervention, comparison condition, and resulting
metrics remain explicit rather than being merged into the policy's unmodified
performance.
