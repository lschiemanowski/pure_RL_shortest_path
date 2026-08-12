# Reward computation and auxiliary objective

Reward computation assigns one scalar reward to each sampled completion using
facts supplied by outcome verification. Let \(d^*\) be the true shortest
distance and \(\ell\) the submitted path length. The base reward is

\[
r_{\mathrm{base}} =
\begin{cases}
0, & \text{if the completion is malformed},\\
0.05, & \text{if it is well-formed but does not give a valid path},\\
1, & \text{if it gives a shortest path},\\
0.5\,d^*/\ell, & \text{if it gives a valid but non-shortest path}.
\end{cases}
\]

A configured reasoning-coverage bonus and sterile-repetition penalty may be
added:

\[
r = r_{\mathrm{base}}
  + \lambda_{\mathrm{coverage}} c
  - \mathbf{1}_{\mathrm{valid}}
    \lambda_{\mathrm{sterile}}\rho_m,
\]

where \(c\) is the fraction of edges in the final answer path that were also
traversed in the reasoning trace. The configured mode
\(m\in\{\mathrm{all},\mathrm{off\mbox{-}answer}\}\) chooses either the
sterile-repetition rate over all legal reasoning transitions or the rate after
excluding transitions on undirected edges in the submitted valid answer. The
penalty is zero for malformed and invalid answers, and its nonnegative
coefficient may be zero. Both repetition rates, the selected mode, each shaping
coefficient, and the total reward remain separately observable.

The optional valid-next-token objective is separate from scalar rollout reward.
At a sampled position \(t\), let \(V_t\) be the complete set of next tokens
permitted by the task and completion state. Its loss is

\[
L_{\mathrm{valid}}
= -\frac{1}{|T|}\sum_{t\in T}
\log\!\left(\sum_{a\in V_t}p_\theta(a\mid h_t)\right),
\]

where \(T\) contains positions for which such a set is defined. This objective
does not provide a demonstrated completion or select one preferred action; it
encourages the policy to place probability mass somewhere inside the locally
valid set. Its coefficient is independently configurable and may be zero.
Experiment evidence distinguishes verifier-reward-only training from runs using
this auxiliary supervision.
