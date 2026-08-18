---
name: stratic-reconcile
description: Reconcile code changes with a Stratic description graph. Use when covered repository code changed, Stratic reports stale or directly affected descriptions, a reconciliation task must be resumed, or a code-led change needs revise-or-confirm decisions, precise description-to-code mapping maintenance, native graph mutations, bounded upward escalation, immutable evidence, and a Git-first changeset.
---

# Stratic Reconcile

Keep semantic judgment in the agent and durable workflow state in Stratic. The
checked-in `docs/spec/` representation remains the accepted authority until an
explicit cutover, but routine reconciliation must produce its managed files by
replaying attributed native mutations. Do not edit descriptions, relationships,
or reconciliation records directly. `docs/spec/bootstrap.yaml` remains
hand-authored configuration.

## Prepare the frontier

1. Ensure the base commit is present in the projection database.
2. If synchronization already created a stale-frontier task, prepare that task
   directly with `--task <task-id>` instead of `--base`. The helper claims a
   pending task or resumes one already claimed by the same actor.
3. Run the bundled preparation helper from the repository root:

```bash
python .agents/skills/stratic-reconcile/scripts/prepare.py \
  --repository . --database .stratic/graph.sqlite3 \
  --actor "$USER" --base HEAD --budget 20
```

For a deliberate new in-scope file, add one explicit
`--assign-uncovered PATH=NODE` argument naming the existing L3 that should own
it. Preparation validates that node and adds the path only to its bounded
frontier; recording still refuses to proceed until a staged native mutation
adds valid coverage and precise semantic mappings. Never use this option to
silence an unexplained ownership gap.

Without `--task`, the helper discovers only changed in-scope code, maps it to
covering L3 nodes, and creates and claims a persistent task. Both modes also
begin or resume the task-owned native changeset and return its identity with
frontier metadata, changed paths, and a patch digest. Stop if preparation
reports uncovered code, an empty frontier, or another active task; do not
broadly rescope the work.

Preparation resolves the Git base first and validates the accepted descriptions,
relationships, and exact code selectors against that committed tree. It then
discovers the worktree diff and frontier separately. A changed source range may
therefore make its accepted selector stale in the worktree without blocking task
creation. The accepted selector must have resolved at the base, and managed graph
files must still match the base; never repair this condition by editing YAML.

If a managed graph file was edited directly, restore it to the generated state
and express the intended change through `stage.py`. Materialization refuses to
adopt dirty manual edits during routine work.

## Make decisions

Inspect one required node at a time so context loading and budget consumption
remain bounded and resumable:

```bash
python .agents/skills/stratic-reconcile/scripts/inspect_node.py \
  --repository . --database .stratic/graph.sqlite3 \
  --actor "$USER" --task <task-id> --node <node-id>
```

The helper returns that node's triggering diff, base description, replayed
native candidate description and relationship, parent, typed node references,
and incoming and outgoing anchored links, including exact code-anchor selectors
and resolved ranges where present. Selectors made stale by the worktree appear in
`unresolved_code_anchors` with their stable identities and failure reason. The
first inspection charges one task
budget unit and at most 8,000 patch bytes. If `next_offset` is non-null, repeat the command
with `--offset <next-offset>`; continuation windows omit repeated graph context.
Each new window charges one task budget unit and persists its offset. The node
becomes reviewed only after its final window, and repeated inspection after that
is free.

If the contract changed, stage each description, hierarchy, realization,
reference, span, or link mutation on the task-owned changeset:

```bash
python .agents/skills/stratic-reconcile/scripts/stage.py \
  --repository . --database .stratic/graph.sqlite3 \
  --actor "$USER" --task <task-id> \
  --rationale "Revise the detailed contract" \
  --mutation-json '{"operation":"description.revise","node_id":"<uuid>","changes":{"body":"# Revised contract\n"}}'
```

The helper rejects creation, immutable evidence, invalid candidate graphs,
unreviewed nodes, and mutations outside the bounded frontier. Reinspect the node
after staging when the resulting candidate or its links need review. Supported
routine operations are `description.revise`, `description.retire`,
`hierarchy.set`, `realization.set`, `reference.set`, `reference.remove`,
`span.set`, `span.remove`, `link.set`, and `link.remove`. For a large or atomic
multi-operation change, pass a UTF-8 JSON object or non-empty array with
`--mutation-file <path>` instead of `--mutation-json`.

Staging applies the complete supplied mutation or mutation array before checking
its selectors against worktree code. Every new or replaced exact selector must
already resolve once; an unchanged selector inherited from the valid base may
remain stale temporarily so several repairs can be staged incrementally. Use the
same stable code-anchor ID with `link.set` to replace its selector. Staging never
drops an inherited mapping or treats it as repaired implicitly.

### Maintain precise mappings

Treat file-level `covers` as frontier ownership, not sufficient semantic
mapping. While reviewing an implemented or partial L3, map each
implementation-bearing contract passage to the smallest stable source range or
ranges that realize it. Stage missing, stale, or needlessly coarse mappings even
when the prose remains accurate.

- Select a meaningful sentence or paragraph, not isolated words. Merge adjacent
  prose spans when they have identical targets and relationship kinds.
- Link one prose span to every independently relevant source range. Prefer a
  definition, declaration, branch, or cohesive block over a whole file.
- Use file-level links only when a precise range would be misleading, such as a
  generated artifact or genuinely diffuse declarative behavior, and state that
  exception in the decision rationale.
- Reuse stable span and code-anchor UUIDs when the same semantic relationship
  persists. Mint UUIDv7 identities only for genuinely new spans or source
  ranges.
- Make every prose and code selector resolve exactly once. Add short `prefix`
  or `suffix` context when its `quote` is ambiguous. A precise code anchor
  records its path, selector, resolved offsets, one-based UTF-16 line and column
  coordinates, and the SHA-256 of its quote.
- Reinspect after `span.set` or `link.set` and verify the candidate relationship
  contains the intended one-to-many mappings. Before recording, require every
  code-linked prose span to be precise unless an explicit exception applies.

Mapping-only mutations still require the node outcome `revised`; leave
`abstraction_changed` false unless the contract itself changed. Do not leave an
affected L3 unmapped merely because its wording was confirmed.

### Review changed description prose

After staging any created description or revision to a description's level,
scope, or body, use the sibling `$stratic-review-description` skill before
deciding the node. Supply the current task and node identities. Address a
`revision_requested` judgment by staging a new candidate and reviewing it again;
route `needs_input` to the user without silently choosing product intent. Do not
substitute a self-authored rationale for the recorded semantic review. Mapping,
realization, hierarchy-parent, reference, span, and link-only mutations do not
require this prose review.

Choose exactly one outcome:

- `revised`: edit the node because its contract or semantic mappings changed.
- `confirmed_unchanged`: leave the node untouched and explain why the change
  does not alter its contract.

Set `abstraction_changed` true only when the parent contract may need revision.
Record each decision immediately:

```bash
python .agents/skills/stratic-reconcile/scripts/decide.py \
  --repository . --database .stratic/graph.sqlite3 \
  --actor "$USER" --task <task-id> --node <node-id> \
  --outcome revised --abstraction-changed \
  --rationale "The node contract and its parent abstraction changed"
```

Omit `--abstraction-changed` when the parent remains accurate. A node with any
staged mutation must be decided `revised`; every revised node must have a staged
mutation. A true decision adds the parent and incoming contractual or
dependency-bearing node or anchor
referrers to the persisted required frontier. Inspect and decide every added
node, continuing until each branch stops on false or reaches L0. Include a `revised` decision for
every node whose description or relationship sidecar was edited in the same
change. Never use an edit as a substitute for an explicit decision.

## Record and validate

After staging all required mutations, append and validate immutable evidence:

```bash
python .agents/skills/stratic-reconcile/scripts/record.py \
  --repository . --database .stratic/graph.sqlite3 \
  --actor "$USER" --task <task-id> \
  --rationale "Reconcile the code change with its contracts"
```

The helper reads decisions from the task cursor, refuses incomplete or
unmatched decisions and mutations, appends one UUIDv7
`reconciliation.record` mutation, validates the complete native candidate, and
automatically materializes canonical descriptions, relationships, evidence, and
`docs/spec/generated-state.json`. It does not use `--adopt-matching`.
Before appending evidence, it requires every exact selector in the complete
candidate to resolve once against the worktree; any inherited stale mapping left
unrepaired stops recording.
It also requires a current approved semantic review for every created or
rewritten description contract. Candidate, neighborhood, or guide changes make
an earlier approval stale.
If later validation finds a defect before capture, `stage.py` accepts a
corrective mutation only for an already reviewed node decided `revised` and
automatically rematerializes the attested mutation-prefix candidate.

## Package the result

Use the changeset returned by `prepare.py`; do not begin a second changeset.
Run `capture`, `validate`, and `submit` with each returned revision. Review and
commit the exact candidate tree through Git, then run `finalize`. Run
`stratic-authority check-generated`, synchronize the accepted commit, verify
that it has no stale nodes, and complete the task with the accepted changeset
ID.

Do not complete the task when validation, review, commit, finalization, or sync
fails. Persist the current frontier and cursor, or complete it with an explicit
error when continuation is unsafe.
