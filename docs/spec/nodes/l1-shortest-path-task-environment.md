# Shortest-path task environment

The shortest-path task environment defines the problems on which policies are
trained and evaluated. It produces synthetic connected, undirected, unweighted
graphs with source–target queries, represents each problem for the policy,
interprets generated reasoning traces and final paths, and derives the exact
graph facts needed to assess a completion.

The environment distinguishes abstract graph structure from the labels used to
represent vertices, so changes in graph size or required distance are not
confused with vocabulary exposure. It reports facts such as format correctness,
path validity, path length, and optimality without deciding how those facts are
combined into rewards or other learning objectives. Training and evaluation use
the same declared task semantics.
