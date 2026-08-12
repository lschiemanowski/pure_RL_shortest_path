# Outcome verification

Outcome verification evaluates a completion against its graph and query. It
reports whether the completion is well-formed and whether its final answer is a
simple path that begins at the source, ends at the target, contains only active
vertices, and follows graph edges. For a valid path it reports its length,
independently computes the true shortest distance, and determines optimality and
excess length.

For a parsed reasoning trace, verification also reports legal and illegal walk
transitions, the ordered legal directed transitions, restarts, visited nodes and
edges, whether the trace reaches the target, and how its traversed edges cover
the final answer. These facts are deterministic and shared by training and
evaluation. Verification never assigns a scalar reward or trusts a
policy-generated claim about correctness.
