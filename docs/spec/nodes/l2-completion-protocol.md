# Completion protocol

A completion contains a reasoning segment followed by a final-answer segment
and explicit termination. The reasoning segment represents one or more graph
walks separated by restart markers, while the final answer is a nonempty
sequence of node labels intended to describe a path from the query source to
its target.

Parsing either returns the reasoning walks and answer path in structured form
or reports a format failure. It determines only syntactic well-formedness;
whether transitions are legal and whether the answer is correct or shortest
belong to outcome verification.
