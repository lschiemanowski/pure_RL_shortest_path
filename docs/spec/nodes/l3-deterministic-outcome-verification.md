# Deterministic outcome verification

Verification recomputes the true source–target distance from the labeled graph.
An answer is a valid path exactly when it starts at the source, ends at the
target, contains only active labels, repeats no vertex, and every consecutive
pair is an edge. A valid path is shortest exactly when its edge count equals the
recomputed distance; its excess length is measured against that distance.

Reasoning verification starts at the query source and classifies generated
moves as legal or illegal graph transitions. A restart begins a new walk at its
following active node without counting as a graph edge. Verification records
the ordered legal directed transitions across all walks, restarts, visited
nodes, the distinct legally traversed undirected edges, target reachability, and
the overlap between traversed reasoning edges and final-answer edges. Restart
markers and illegal moves do not appear in the ordered legal-transition list.

All reported facts are deterministic and use the same implementation in
training and evaluation. Verification does not compute a scalar reward and does
not trust model-generated statements about correctness.
