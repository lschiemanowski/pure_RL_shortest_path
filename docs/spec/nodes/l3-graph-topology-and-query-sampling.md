# Graph topology and query sampling

A graph-problem request declares the number of vertices n, number of edges m,
admissible shortest-distance interval [d_min,d_max], and maximum number of
sampling attempts. Construction
validates these parameters before consuming random state.

Each attempt constructs a graph over abstract vertices 0,...,n-1 by
sampling a uniform random labeled spanning tree and then sampling unused edges
until the graph contains exactly m edges. It computes shortest distances by
breadth-first search, collects all ordered source–target pairs whose distance
lies in the declared interval, and samples one qualifying pair uniformly. If an
attempt has no qualifying pair, construction samples another graph; exhaustion
produces an explicit failure.

The graph topology and query are determined before vocabulary labels are
assigned. Given the same configuration and random-generator state,
construction produces the same result.
