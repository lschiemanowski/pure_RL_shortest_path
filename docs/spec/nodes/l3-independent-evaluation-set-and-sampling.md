# Independent evaluation set and sampling

An evaluation problem set declares one graph-problem configuration and a
nonempty ordered sequence of matching examples. It is identified by exactly one
independent generation seed or stable fixed-set identity. Seeded construction
uses a private random stream and is reproducible from the declared task and
vocabulary semantics; fixed sets validate that every supplied example matches
their declared graph scale and distance bounds.

Evaluation generates one unconstrained completion per problem. Greedy generation
is the default protocol and accepts no unused temperature, top-p, or sampling
seed. A separately declared stochastic protocol requires positive temperature,
top-p in \((0,1]\), and an independent sampling seed. Top-p sampling retains the
smallest descending-probability prefix whose cumulative mass reaches the
threshold, including the token that crosses it, renormalizes that prefix, and
samples from it.

Generation consumes raw full-vocabulary policy scores without legality masks,
forced tokens, verifier intervention, or post-hoc repair. It stops a completion
when the policy emits the termination symbol or the declared maximum length is
reached, without appending an artificial termination symbol. Batching, padding,
and cached scoring do not change greedy results. Evaluation runs without
gradients and restores the policy's preceding train-or-evaluation mode.
