# Policy evaluation and outcome aggregation

An evaluation protocol identifies the policy checkpoint and complete
completion-sampling configuration, including the sampling seed for a stochastic
evaluation and the positive minimum number of reasoning tokens required by the
completion protocol. Ordinary and intervention evaluations are distinct. An
intervention protocol must identify both the intervention and its comparison
condition; ordinary evaluation refuses to apply an implicit intervention.

Aggregation requires exactly one completion for each problem in the evaluation
set and preserves their order. It parses and verifies each completion using the
same task implementation and declared minimum-reasoning requirement used by
training. The primary result records the evaluation denominator and exact
formatting-success, valid-path-success, and shortest-path-success counts
together with their rates. Mean path length and mean excess length are computed
only over valid paths and remain secondary diagnostics.

The result retains the checkpoint, problem-set identity, graph-problem
configuration, sampling protocol, individual completions, and verifier facts
behind the aggregate. The ordinary evaluation operation receives no optimizer,
reference policy, training random stream, or curriculum state and performs no
parameter update. Intervention completions must be produced explicitly before
aggregation, preventing their metrics from being presented as unmodified policy
performance.
