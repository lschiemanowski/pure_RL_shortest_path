# Decoder-only transformer construction

The transformer configuration declares positive vocabulary, context, model, layer, attention-head, and feed-forward dimensions together with positive rotary-position and normalization constants. Model width is divisible by the head count, and each attention-head dimension is even so rotary positions are well-defined. Construction rejects a configuration that violates these relationships.

The model uses token embeddings, pre-normalized causal self-attention with rotary positions, gated SiLU feed-forward blocks, final RMS normalization, and a tied full-vocabulary output. Ordinary actor scoring retains its logits-and-cache interface. A separate full-prefix training operation returns the identical actor logits together with the final normalized tensor supplied to the language-model head, allowing a value head without changing sampling or evaluation.

Linear and embedding matrices are initialized from the declared zero-mean normal distribution, RMS scales begin at one, and construction accepts no pretrained source. Given the same configuration and Torch random state, construction produces identical learned parameters.
