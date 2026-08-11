# Valid-next-token auxiliary loss

An auxiliary graph-aware state machine computes a set \(V_t\) of locally valid
actions before each sampled completion token. During reasoning, legal actions
are graph neighbors of the current node, a restart marker after a nonempty walk,
and the end-reasoning marker after at least one legal reasoning move. A restart
must be followed by any active node label. The answer protocol then requires the
begin-answer marker, the query source, unvisited graph neighbors forming a
simple path, the end-answer marker upon reaching the target, and finally the
termination symbol.

If the sampled action is outside \(V_t\), the auxiliary environment rejects it
and retains its preceding state for the next position. This does not mask,
replace, or otherwise alter the policy's sampled completion or its verifier
reward.

For all sampled positions \(T\) with a nonempty valid set, the loss is

\[
L_{\mathrm{valid}}
= -\frac{1}{|T|}\sum_{t\in T}
\log\!\left(\sum_{a\in V_t}p_\theta(a\mid h_t)\right).
\]

It rewards probability mass on the complete valid set rather than demonstrating
one target action or shortest path. Its coefficient is independently
configurable and may be zero.
