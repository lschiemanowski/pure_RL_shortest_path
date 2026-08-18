# Description style guide

This document defines the shared style for Stratic description hierarchies. The
executable leased-queue example in the Stratic repository illustrates it.
Description-writing workflows require a current semantic approval from the
shared review skill before recording created or rewritten contracts; no generic
prose linter reduces these conventions to mechanical writing rules.

## The purpose of each level

Each lower level answers a different question. It should add decisions that do
not belong in its parent rather than restating the parent in more words.

| Level | Question | Include | Exclude |
|---|---|---|---|
| L0 Intent | Who uses the system, what form does it take, and what does successful use look like? | Actors, externally visible form, outcomes, representative usage, system-wide constraints | Internal components, exact interface syntax, algorithms |
| L1 Architecture | Where do major responsibilities lie? | Components, boundaries, ownership, data flow | Function behavior and storage mechanics |
| L2 Contract | What behavior does one component promise? | Inputs, outputs, transitions, invariants, failures | Control flow and source layout |
| L3 Detail | How is that contract realized here? | Algorithms, representations, edge cases, implementation constraints | Product rationale already owned above |
| Code | What executes? | Source and tests | Repeated prose contracts |

The example intentionally uses every level because it is teaching the
boundaries. A production graph should not invent a node merely to fill a level;
introduce a node only when it owns a coherent responsibility at that
abstraction.

## Write clear, self-contained descriptions

Each node is both an explanation and an enduring target contract. It should be
understandable on its own at its stated abstraction level. A reader may follow
its children for greater detail, but should not need them merely to learn what
the subject is, why it exists, or what it promises.

Completeness and clarity come before brevity. Use as much prose as needed to
establish the subject, relevant context, behavior, representative examples, and
important boundaries. Control context cost through bounded scopes, appropriate
abstraction, and removal of repetition—not by omitting explanatory material.

- Orient the reader before presenting dense behavioral claims. Name the subject
  and its role in terms appropriate to the level.
- Give each node one bounded scope. Split it when unrelated contracts repeatedly
  change for different reasons, not merely because the description has grown
  beyond a preferred length.
- Do not optimize for a fixed word or paragraph count. Revise for comprehension
  first, then remove repetition and details owned by lower levels.
- At L0, identify the human or software actors and the system's externally
  visible form: for example, a library, command-line application, background
  service, or network server. State whether interaction is local or networked
  and one-shot or long-running when that distinction shapes how the system is
  used. This defines the product boundary, not its internal decomposition.
- At L0, include a short representative example when the purpose and interaction
  are not immediately concrete. Describe the user's situation and the visible
  outcome in domain language; leave command syntax and internal components to
  lower levels.
- State enduring behavior in present tense: “A claim selects the oldest
  available job.”
- State externally meaningful constraints explicitly: “Acknowledgement removes
  a job only when the requesting worker owns its unexpired lease.”
- Name exclusions when they define the boundary: “Concurrent writers are
  outside this example's scope.”
- Normative prose may describe intended behavior that is still partial or
  unimplemented. State that behavior as an enduring target contract, and record
  its realization separately. Keep delivery plans, progress, unresolved
  questions, discarded alternatives, and revision history in tasks, rationale,
  design notes, or Git.
- Do not mention a class, file, or function above L3 merely because the current
  implementation uses it.

The parent must remain true when details change. For example, an L1 queue engine
can say that lifecycle rules validate transitions and persistence preserves
their results. Adding a new transition then changes the L2 and L3 contracts
without requiring that architectural statement to enumerate it.

## Make hierarchy links discoverable in the prose

Every direct child should elaborate a responsibility-bearing phrase in its
parent. Anchor the complete meaningful phrase, sentence, or paragraph—not a
heading, comma, connective, or repeated keyword.

For example, an L1 sentence stating that a queue engine owns the job lifecycle
and coordinates each state change with durable storage can link to lifecycle
and repository children. The `parent` field defines the hierarchy; span links
make the prose navigable and stable identities live in relationship sidecars.

Use a single primary parent. A relationship that crosses branches is a typed
reference:

- `informational` supports navigation but does not propagate impact.
- `contractual` means the source relies on the target's promised behavior.
- `dependency-bearing` means changes in the target may require the source to be
  reconsidered.

## Map L3 claims to exact code

An L3 `covers` selector assigns frontier ownership; it does not prove that the
prose describes the code. Each implementation-bearing paragraph therefore
links to the smallest stable source ranges that realize it.

Prefer a definition, condition, transition block, schema declaration, or test
case. One prose span may link to several ranges when its behavior is distributed
between implementation and tests. Reuse stable span and code-anchor IDs while
the semantic relationship remains the same. Add `prefix` or `suffix` context
when a quote occurs more than once.

Avoid these mappings:

- a whole file when one cohesive block realizes the claim;
- a punctuation mark or isolated connective;
- a function name that provides no evidence of the described behavior;
- one convenient source range when independent parts of the claim are realized
  elsewhere.

## Keep realization separate from correctness

`implemented`, `partial`, and `unimplemented` say how much of the target
contract exists in the selected code version. They do not say whether the prose
has been reconciled with recent changes.

- Use `implemented` only when the complete scoped contract is materially
  present.
- Use `partial` when a useful subset exists and the unreached contract remains
  intentional.
- Use `unimplemented` when the node records target behavior without material
  implementation.

Do not delete future intent merely to make a node appear implemented, and do
not represent absent implementation as staleness.

## Use changes to test the decomposition

A useful hierarchy produces a bounded semantic frontier. A change should begin
at its owning contract, follow relevant children and typed references, and
prune dependencies whose representations and guarantees remain valid.

If a small change routinely reaches every node, the upper contracts are likely
too detailed, references may be over-classified, or scopes may mix unrelated
responsibilities. If relevant nodes are missed, the hierarchy, references, or
L3 mappings are too weak.
