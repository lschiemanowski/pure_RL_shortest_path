# Atomic checkpoints and validated resumption

Every checkpoint is a versioned record containing the selected algorithm, current and frozen-reference actor parameters, optimizer state, and PPO value-head parameters when applicable, plus the completed step, curriculum state, resolved configuration and digest, run identity, and named random states required to continue training. Its purpose distinguishes periodic, curriculum-transition, interruption, and final checkpoints.

A checkpoint is first written to a temporary sibling and then atomically installed at its step- and purpose-specific path. The convenient latest path is updated by a second atomic replacement, leaving the named checkpoint usable if that update is interrupted.

Loading validates the checkpoint schema, presence of PPO value parameters, and the configuration digest against either the serialized configuration record or its normalized typed form. A valid older record may therefore acquire newly introduced defaults without being rejected or losing its original digest. Same-run resumption requires the recorded configuration. A derived run may change future training settings but rejects changes to the selected algorithm, task vocabulary, completion protocol, model architecture, master seed, or ordered curriculum stages.

After validation, resumption restores the current and reference actors, optional value head, optimizer, curriculum state, named random streams, and applicable Torch device random state before producing new experience. A changed compatible optimizer configuration in a derived run is applied explicitly after restoring its accumulated state.
