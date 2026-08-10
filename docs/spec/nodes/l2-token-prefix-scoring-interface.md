# Token-prefix scoring interface

The scoring interface accepts batches of token prefixes together with an
optional padding mask and optional cached attention state. It returns next-token
scores over the complete configured vocabulary at every supplied position and,
when requested, the updated attention state. It rejects an effective prefix
longer than the model's configured context limit.

Scores are causal: a position depends only on its non-padding prefix. Full-prefix
and incremental cached evaluation represent the same policy for the same
effective input. The interface returns scores only; the invoking process owns
token selection, score masking or forcing, and stopping decisions. It does not
parse completions, consult graph verification, or access shortest-path answers.
