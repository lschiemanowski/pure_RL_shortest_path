# Graph problem distribution

A graph-problem configuration declares a number of vertices, a number of edges,
and an admissible shortest-distance interval. Every generated instance is a
simple connected undirected unweighted graph with exactly those vertex and edge
counts, plus distinct active source and target vertices whose distance lies in
the declared interval.

Given the same configuration and random state, generation produces the same
sequence of instances. It operates on abstract vertex identities before any
vocabulary labels are assigned. Invalid parameter combinations are rejected
before generation. If bounded sampling cannot produce an instance whose
source–target distance satisfies the requested interval, generation fails
explicitly; it never returns an instance outside the declared vertex, edge, or
distance constraints.
