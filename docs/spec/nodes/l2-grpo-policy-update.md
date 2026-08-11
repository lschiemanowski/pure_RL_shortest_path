# GRPO policy update

For each training problem, Group Relative Policy Optimization compares a group
of \(K\) sampled completions with rewards \(r_1,\ldots,r_K\). Their normalized
advantages are

\[
A_i =
\begin{cases}
(r_i-\bar r)/(\sigma_r+\varepsilon), & \sigma_r>\varepsilon,\\
0, & \text{otherwise}.
\end{cases}
\]

A group whose rewards have effectively no variation therefore contributes no
policy advantage.

For sampled token \(a_{i,t}\), let

\[
\rho_{i,t}
= \frac{\pi_\theta(a_{i,t}\mid h_{i,t})}
       {\pi_{\mathrm{old}}(a_{i,t}\mid h_{i,t})}.
\]

The clipped sequence-normalized policy objective is

\[
L_{\mathrm{policy}}
= -\frac{1}{K}\sum_i\frac{1}{T_i}\sum_t
\min\!\left(
  \rho_{i,t}A_i,
  \operatorname{clip}(\rho_{i,t},1-\epsilon,1+\epsilon)A_i
\right).
\]

Only tokens actually sampled in the completion contribute; they are policy
actions rather than supervised target tokens.

The complete update combines this objective with a configurable divergence
penalty against a frozen reference policy and the optional valid-next-token
loss:

\[
L = L_{\mathrm{policy}}
  + \beta_{\mathrm{KL}}L_{\mathrm{KL}}
  + \beta_{\mathrm{valid}}L_{\mathrm{valid}}.
\]

The update may use declared microbatches and multiple optimization epochs,
clips gradients before the optimizer step, and updates only the current policy.
Refreshing the frozen reference is explicit and does not alter the already
collected rollout batch.
