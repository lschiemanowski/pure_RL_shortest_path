# Task and vocabulary representation

The task vocabulary contains a declared set of structural symbols and a fixed
pool of node labels. Each abstract graph receives a fresh injective assignment
of its vertices to labels sampled from the entire configured pool. The same
assignment is applied consistently to the graph, query, and completion
semantics, and it preserves the underlying shortest-path problem.

Serialization presents the complete edge set and source–target query without
revealing a solution or shortest distance. Edge order and orientation carry no
meaning and may vary independently. The vocabulary and serialization semantics
are identical in training and evaluation and are recorded as part of the task
configuration.
