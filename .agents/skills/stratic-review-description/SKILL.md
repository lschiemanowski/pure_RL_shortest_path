---
name: stratic-review-description
description: Semantically review a staged Stratic description against the repository's shared description style guide and its immediate graph neighborhood. Use before approving any created description or revision to a description's level, scope, or body; when a reconciliation, propagation, implementation, completion, or bootstrap workflow requests a description approval; or when an existing review became stale after candidate, context, or guide changes.
---

# Stratic Review Description

Review meaning and abstraction placement rather than matching prose to mechanical
rules. Do not edit the candidate. Return evidence to the writing workflow so it
can revise the candidate or request a consequential product decision.

## Load the exact review context

Run from the repository root:

```bash
stratic-description-review \
  --repository . --database .stratic/graph.sqlite3 --actor "$USER" \
  context --task <task-id> --node <node-id>
```

Read the complete `guide.content`, candidate node, base node when present,
parent, children, siblings, related descriptions, task request, and review
identity. Review only this bounded context. Do not rely on an earlier copy of
the guide or an earlier candidate inspection.

## Make a semantic judgment

Consider the following as qualitative lenses, not independent pass/fail rules:

- whether the node is understandable by itself at its stated level;
- whether it answers the question owned by that level;
- whether it adds decisions instead of expanding its parent through repetition;
- whether its parent remains true when lower-level details change;
- whether it owns one coherent responsibility with adequate context and bounds;
- whether direct children elaborate responsibility-bearing prose;
- whether it describes an enduring target rather than delivery state;
- whether an L3's implementation-bearing claims are specific enough to map to
  the linked code evidence.

Use the guide's examples as calibration, not templates. Do not reject prose for
word count, sentence form, headings, vocabulary frequency, or isolated phrases.
Quote the candidate or graph context when identifying a concern.

Choose one verdict:

- `approved` when the candidate coheres with the guide and neighborhood;
- `revision_requested` when the writer can repair a concrete concern without a
  new product decision;
- `needs_input` when authority, intent, scope, or abstraction cannot be resolved
  from the bounded evidence.

Write a UTF-8 JSON object with a non-empty `summary` and an `observations` array.
Each observation may record a `lens`, quoted `evidence`, `reasoning`, and a
specific `suggestion`. Do not manufacture observations merely to populate the
array.

```json
{
  "verdict": "revision_requested",
  "summary": "The contract mixes L2 behavior with L3 representation details.",
  "observations": [
    {
      "lens": "level_fit",
      "evidence": ["The lease deadline is stored in the expires_at column."],
      "reasoning": "The column representation is an L3 decision.",
      "suggestion": "Keep the expiry guarantee here and move storage representation below."
    }
  ]
}
```

## Record the judgment

Persist the review against the exact guide, candidate, and context digests:

```bash
stratic-description-review \
  --repository . --database .stratic/graph.sqlite3 --actor "$USER" \
  record --task <task-id> --node <node-id> \
  --review-file /tmp/stratic-description-review.json
```

Return `revision_requested` to the writing workflow for revision and a fresh
review. Return `needs_input` to the user without changing the candidate's intent.
Proceed to the workflow decision only after an `approved` review. Any relevant
candidate, neighborhood, or guide change makes the approval stale and requires a
new context and judgment.
