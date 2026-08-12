# Asymmetric frontier-centered stage distribution

Let the ordered curriculum contain stages \(0,\ldots,N-1\), and let \(f\) be
the current integer frontier. Before constructing each training problem, the
protocol independently samples its stage \(i\) according to

\[
p(i\mid f)=\frac{w_i(f)}{\sum_{j=0}^{N-1}w_j(f)},
\]

where

\[
w_i(f)=
\begin{cases}
\exp\!\left(-\dfrac{f-i}{\tau_{\mathrm{past}}}\right), & i<f,\\[6pt]
1, & i=f,\\[6pt]
\exp\!\left(-\dfrac{i-f}{\tau_{\mathrm{future}}}\right), & i>f.
\end{cases}
\]

The positive decay scales satisfy
\(\tau_{\mathrm{past}}>\tau_{\mathrm{future}}\), so preceding stages form a
broader rehearsal tail while succeeding stages lose probability much more
rapidly. The frontier is the unique modal stage. All stages retain positive
probability, and normalization automatically accounts for the beginning and end
of the finite curriculum. A configuration whose forward tail underflows a
stage's representable probability to zero is rejected.

After sampling a stage, the existing graph-problem generator independently
constructs an example from that stage's declared topology and distance
constraints. Node labels continue to be sampled from the complete global label
pool. Completions for one problem form a GRPO comparison group; the curriculum
distribution is sampled once for the problem, not separately for its
completions.

The training random stream determines all stage draws and graph examples
reproducibly. Each training batch records the frontier, decay-derived normalized
stage probabilities, sampled stage of every problem, and realized problem count
from every stage. Failure to construct a problem from its sampled stage is
reported explicitly rather than replacing it with a problem from another stage.
