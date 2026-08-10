# Shortest-path task environment

The shortest-path task environment defines the problems on which policies are
trained and evaluated while keeping task meaning and correctness independent of
any policy architecture or learning algorithm.

Its graph problem distribution produces reproducible abstract graph and query
instances with declared size, density, and shortest-distance constraints.

The task and vocabulary representation maps each abstract instance into the
policy's input space without allowing label exposure to encode curriculum
difficulty. The completion protocol defines how freely generated token
sequences are interpreted as reasoning traces and final paths.

Outcome verification derives exact format, path-validity, length, optimality,
and reasoning-trace facts from an instance and completion. The environment
reports those facts without combining them into rewards or other learning
objectives, and training and evaluation use the same declared task semantics.
