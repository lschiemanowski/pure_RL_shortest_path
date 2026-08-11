# Transformer model

A transformer configuration declares the vocabulary size, context limit, model
width, layer count, attention-head count, feed-forward width, rotary-position
base, and normalization tolerance of a decoder-only causal model. Construction
validates the dimensional relationships required by the declared architecture
and rejects incompatible configurations.

Construction creates token embeddings, causal transformer blocks, and
full-vocabulary output scores, and initializes every learned parameter from
scratch without loading pretrained weights. Learned matrices use declared
random initialization while normalization scales use declared fixed initial
values. Given the same configuration and random state, initialization is
reproducible. The declared configuration and resulting learned parameters
together identify the transformer model supplied to the surrounding training
and checkpoint workflows.
