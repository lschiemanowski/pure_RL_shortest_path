# Causal prefix scoring

The scoring operation accepts a nonempty batch of token prefixes, an optional
boolean mask over the complete effective prefix, and an optional key/value cache
with one compatible entry per transformer layer. It rejects incompatible cache
shapes, masks, or an effective prefix exceeding the configured context limit.

The operation returns logits over the complete vocabulary at every newly
supplied position and, when requested, the updated per-layer cache. Attention at
each position includes only unmasked keys at that position or earlier. Scoring a
prefix at once, one token at a time, or in cached multi-token chunks represents
the same policy up to numerical tolerance.

The interface returns only logits and optional attention state. It does not
sample or force tokens, apply task-legality masks, decide termination, parse a
completion, invoke graph verification, or access shortest-path answers.
