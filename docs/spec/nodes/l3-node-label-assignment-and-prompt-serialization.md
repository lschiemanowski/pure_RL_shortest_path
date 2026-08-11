# Node-label assignment and prompt serialization

Each abstract graph receives a fresh injective mapping from its vertices into
the complete configured node-label pool. The active labels are sampled uniformly
without replacement from that pool, independently of curriculum stage. The
mapping is applied consistently to edges, source, target, completion
interpretation, and verification.

After labeling, edge order is shuffled and each undirected edge is independently
assigned either orientation for serialization. The prompt begins with the graph
edge sequence, then gives the source–target query, and ends by opening the
reasoning segment at the source node. It contains no path, target completion,
shortest distance, or other solution information. Generated policy tokens begin
after the final source token.
