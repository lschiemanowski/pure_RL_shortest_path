# Checkpointing and resumption

A checkpoint contains the current and reference-policy parameters, optimizer
state, completed training step, curriculum position, interpolation and
advancement state, resolved model configuration, and the random-generator
states needed to continue the run. Periodic, curriculum-transition, and final
checkpoints are distinguished by their purpose, and checkpoint replacement is
atomic so that an interrupted write does not destroy the last usable state.

Resumption validates that the requested task vocabulary and model configuration
are compatible with the checkpoint. Changes that would reinterpret learned
parameters or previously generated experience are rejected. After restoration,
training continues from the recorded optimizer, curriculum, and random state;
any intentionally permitted configuration change and the checkpoint from which
the run resumed remain visible in the run provenance.
