# Deterministic outcome verification

Verification recomputes the true source–target distance from the labeled graph.
An answer is a valid path exactly when it starts at the source, ends at the
target, contains only active labels, repeats no vertex, and every consecutive
pair is an edge. A valid path is shortest exactly when its edge count equals the
recomputed distance; its excess length is measured against that distance.

Reasoning verification starts at the query source and classifies generated
moves as legal or illegal graph transitions. A restart begins a new walk at its
following active node without counting as a graph edge. Verification records
walk transitions, restarts, visited nodes, legally traversed edges, target
reachability, and the overlap between traversed reasoning edges and final-answer
edges.

Let \(e_1,\ldots,e_T\) be the ordered legal directed reasoning transitions,
excluding restarts and illegal moves. This sequence is global across all walks,
so a restart does not erase repetition history. Let \(D_t\) be the set of
distinct undirected graph edges traversed through position \(t\). If \(\tau(t)\)
is the preceding occurrence of the same directed transition, occurrence \(t\)
is sterile exactly when \(\tau(t)\) exists and

\[
D_{t-1}=D_{\tau(t)}.
\]

Thus a transition may be reused without being classified as sterile when the
interval since its preceding use discovered a new graph edge. The all-transition
rate is the number of sterile occurrences divided by \(T\), or zero when
\(T=0\). For the off-answer rate, verification removes from both numerator and
denominator transitions whose undirected edge belongs to the submitted valid
answer; if there is no valid answer, no edges are removed. An empty denominator
produces rate zero. Neither definition asks which edges belong to any
graph-theoretic shortest path.

All reported facts are deterministic and use the same implementation in
training and evaluation. Verification does not compute a scalar reward and does
not trust model-generated statements about correctness.
