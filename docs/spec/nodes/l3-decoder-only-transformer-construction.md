# Decoder-only transformer construction

The transformer configuration declares positive vocabulary, context, model,
layer, attention-head, and feed-forward dimensions together with positive
rotary-position and normalization constants. Model width is divisible by the
head count, and each attention-head dimension is even so rotary positions are
well-defined. Construction rejects a configuration that violates these
relationships.

The model uses token embeddings, pre-normalized causal self-attention with
rotary positions, gated SiLU feed-forward blocks, a final RMS normalization,
and a full-vocabulary linear output whose weights are tied to the token
embeddings. Linear and embedding matrices are sampled from a zero-mean normal
distribution with standard deviation 0.02, while RMS normalization scales begin
at one. Construction accepts no checkpoint or pretrained parameter source.

Given the same configuration and Torch random-generator state, construction
produces identical learned parameters. The immutable configuration and model
state dictionary provide the architecture and parameter identity needed by
checkpoint and experiment records.
