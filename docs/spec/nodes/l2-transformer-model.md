# Transformer model

A transformer configuration declares the vocabulary size, context limit, model
width, layer count, attention-head count, feed-forward width, rotary-position
base, and normalization tolerance of a decoder-only causal model. Construction
validates the dimensional relationships required by the declared architecture
and rejects incompatible configurations.

Construction creates token embeddings, causal transformer blocks, and
full-vocabulary output scores, and randomly initializes every learned parameter
without loading pretrained weights. Given the same configuration and random
state, initialization is reproducible. The declared configuration and resulting
learned parameters together identify the transformer model supplied to the
surrounding training and checkpoint workflows.
