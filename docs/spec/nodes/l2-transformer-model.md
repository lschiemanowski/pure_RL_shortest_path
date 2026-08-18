# Transformer model

A transformer configuration declares the vocabulary size, context limit, model width, layer count, attention-head count, feed-forward width, rotary-position base, and normalization tolerance of a decoder-only causal model. Construction validates the dimensional relationships required by the declared architecture and rejects incompatible configurations.

Construction creates token embeddings, causal transformer blocks, final normalization, and full-vocabulary output scores, and initializes every learned parameter from scratch without loading pretrained weights. The forward operation may return the same final normalized token representations used by the language-model head when a training-only auxiliary head requests them; ordinary actor scoring omits them. Given the same configuration and random state, initialization is reproducible.
