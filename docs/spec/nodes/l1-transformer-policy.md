# Transformer policy

The transformer policy is the trainable system that maps a serialized graph and
source–target query to an autoregressively generated reasoning trace and final
path. It begins from randomly initialized parameters and generates its
completion without task-specific token masks, forced actions, or access to a
shortest-path oracle.

The policy exposes the same generation behavior to training and evaluation,
with sampling choices controlled by the surrounding experiment. Its
configuration and learned state can be saved and restored together so a
checkpoint identifies a reproducible policy rather than an unexplained
collection of weights.
