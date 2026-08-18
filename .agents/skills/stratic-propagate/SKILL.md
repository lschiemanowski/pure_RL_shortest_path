---
name: stratic-propagate
description: Propagate a requested Stratic description change downward into affected descendant contracts, contractual dependencies, and covered code. Use when a human selects a stable description node and asks to change its contract, when a spec-led propagation task must be resumed, or when descriptions should lead an attributed graph-plus-code changeset through review, Git commit, synchronization, and completion.
---

# Stratic Propagate

Keep semantic judgment in the agent and durable workflow state in Stratic. Start
from the human-selected stable node and proposed contract change. Expand only
through explicitly confirmed branches; never treat a whole-graph scan as routine
propagation. Do not edit managed descriptions, relationships, spans, links, or
reconciliation evidence directly. Stage native graph mutations and let the
workflow materialize their checked-in representation.

## Prepare the task

Run the bundled workflow from the repository root:

```bash
python .agents/skills/stratic-propagate/scripts/propagate.py \
  --repository . --database .stratic/graph.sqlite3 --actor "$USER" \
  prepare --base HEAD --node <stable-node-id> \
  --request "Describe the requested contract change" --budget 30
```

Preparation resolves the base commit, rejects dirty managed graph files and
another active propagation task, creates and claims a persistent task, begins
its native changeset, and returns the root's immediate candidate expansions.
Resume an existing task with `prepare --task <task-id>`. Keep using the returned
task and changeset; do not create a parallel changeset.

## Inspect, stage, and decide descriptions

Inspect each required node before deciding it:

```bash
python .agents/skills/stratic-propagate/scripts/propagate.py \
  --repository . --database .stratic/graph.sqlite3 --actor "$USER" \
  inspect-node --task <task-id> --node <node-id>
```

The response includes the requested change, current candidate description and
relationship sidecar, reasons for inclusion, immediate children, outgoing
contractual or dependency-bearing links, and covered code for L3 nodes. The
first inspection consumes one persisted budget unit; repeat inspection is free.

When a contract or relationship changes, stage one native mutation object or an
atomic JSON array:

```bash
python .agents/skills/stratic-propagate/scripts/propagate.py \
  --repository . --database .stratic/graph.sqlite3 --actor "$USER" \
  stage-graph --task <task-id> --rationale "Revise the selected contract" \
  --mutation-file /tmp/graph-mutations.json
```

The helper validates the replayed candidate before persisting it. It accepts the
normal description, hierarchy, realization, reference, span, and link
operations. `description.create` is allowed only below an inspected required
parent. Evidence creation belongs to `record`, not manual staging.

For every created or revised implemented or partial L3, add or refresh precise
description-to-code mappings as part of the same candidate. Treat `covers` as
frontier ownership only. Select meaningful prose sentences or paragraphs and
link each to the smallest stable source range or ranges that realize it; one
prose span may have several code anchors. Merge adjacent spans with identical
links, preserve stable span and code-anchor UUIDs when their meaning persists,
and use disambiguating selector context when quotes repeat. Permit a file-level
link only when a precise range would be misleading, and record the exception in
the node rationale. Do not finish propagation with an unmapped L3 or an
imprecise code-linked span merely because the prose and code decisions are
otherwise complete.

After staging any created description or revision to a description's level,
scope, or body, use the sibling `$stratic-review-description` skill with the
current task and node before deciding it. Address `revision_requested` by
staging and reviewing a new candidate. Route `needs_input` to the user without
silently choosing product intent. Relationship, mapping, realization, and
covered-path-only mutations do not require this prose review.

Choose one outcome for every required node:

- `revised`: the node has one or more staged native mutations.
- `confirmed_unchanged`: its contract remains accurate.
- `pruned`: its branch is unaffected and must not expand.

```bash
python .agents/skills/stratic-propagate/scripts/propagate.py \
  --repository . --database .stratic/graph.sqlite3 --actor "$USER" \
  decide-node --task <task-id> --node <node-id> \
  --outcome revised --propagate \
  --rationale "The changed contract affects its detailed descendants"
```

Add `--propagate` only after deciding that the branch needs examination. It adds
the node's immediate children, outgoing non-informational references and span
links, and, for an L3, its matched covered code. Omitting it stops expansion at
that node. A pruned branch cannot propagate. The workflow rejects decisions that
would orphan an already staged mutation.

## Inspect, stage, and decide code

Inspect each required path in bounded windows:

```bash
python .agents/skills/stratic-propagate/scripts/propagate.py \
  --repository . --database .stratic/graph.sqlite3 --actor "$USER" \
  inspect-code --task <task-id> --path <path>
```

If `next_offset` is present, repeat with `--offset <next-offset>`. Stage an
attributed write from a prepared file, from the current worktree, or a deletion:

```bash
python .agents/skills/stratic-propagate/scripts/propagate.py \
  --repository . --database .stratic/graph.sqlite3 --actor "$USER" \
  stage-code --task <task-id> --path <path> --from-worktree \
  --rationale "Implement the propagated contract"
```

Then record `revised` or `confirmed_unchanged` with `decide-code`. Revised code
must have an attributed staged file mutation, and staged code must be revised.
Do not change files outside the computed frontier. Record unresolved limits with
`note-uncertainty`; uncertainty is persisted on both the task and changeset and
does not authorize a wider scan.

## Record and package the exact candidate

After every required node and path has a decision, append spec-led evidence and
materialize the generated graph:

```bash
python .agents/skills/stratic-propagate/scripts/propagate.py \
  --repository . --database .stratic/graph.sqlite3 --actor "$USER" \
  record --task <task-id> \
  --rationale "Propagate the requested contract through the approved frontier"
```

Recording rejects incomplete decisions, unmatched staged mutations, changed
paths outside the approved frontier, and an invalid impact check. It appends one
UUIDv7 reconciliation record and materializes without `--adopt-matching`.
It also rejects every created or rewritten description contract without a
current approved semantic review; candidate, neighborhood, or guide changes
make an earlier approval stale.

Use the same task-owned changeset for `capture`, `validate`, and `submit` with
optimistic revisions. Review and commit the exact candidate tree through Git,
then `finalize` it against that commit. Run
`stratic-authority check-generated`, synchronize the accepted commit, confirm
zero stale nodes, and complete the task with the accepted changeset ID. Do not
complete the task if review, validation, commit, finalization, synchronization,
or generated-state verification fails. Do not perform an authority cutover as
part of routine propagation.
