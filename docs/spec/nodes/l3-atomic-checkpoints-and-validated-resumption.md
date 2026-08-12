# Atomic checkpoints and validated resumption

Every checkpoint is a versioned record containing the current policy, frozen
reference policy, optimizer state, completed training step, curriculum frontier
and advancement streak, resolved configuration and digest, run identity, and
all named and framework random-generator states required to continue training.
It identifies whether it was created periodically, at a curriculum transition,
after interruption, or at normal completion.

A checkpoint is first written to a temporary sibling and then atomically
installed at its step- and purpose-specific path. The convenient latest path is
updated by a second atomic replacement, leaving the named checkpoint usable if
that update is interrupted.

Loading validates the checkpoint schema and configuration digest. Same-run
resumption requires the recorded configuration and originating source revision;
an operational stopping-step override is recorded separately. A derived run
may change future training settings, but it retains parent lineage and rejects
changes to the task vocabulary, model architecture, master seed, or ordered
curriculum stages.

After validation, resumption restores the current and reference policies,
optimizer, curriculum state, named random streams, and applicable Torch device
random state before producing new experience. A changed optimizer configuration
in a derived run is applied explicitly after restoring its accumulated state.
