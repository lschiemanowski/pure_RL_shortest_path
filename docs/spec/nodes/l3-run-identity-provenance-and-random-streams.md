# Run identity, provenance, and random streams

A fresh run receives a stable run identifier and a new artifact directory. Its
provenance record contains the source and resolved-configuration digests, source
revision and source-tree status, start time, relevant Python and PyTorch
versions, requested and selected device, numerical precision, and exact launch
command. Starting a fresh run in an existing run directory is rejected. The selected optimization algorithm remains part of the resolved configuration used to identify the run.

One declared master seed deterministically produces separately named random
streams for model initialization, training-problem construction, rollout
sampling, curriculum validation, evaluation-problem construction, and
evaluation sampling. Seed derivation is stable and does not depend on
process-local hashing. The resolved seeds are recorded explicitly so that
independence between streams can be inspected rather than inferred.

Runs from a modified source tree are rejected by default. If explicitly
permitted for exploratory work, the dirty status and an exact source patch are
retained with the run rather than associating its result only with the last
commit.
