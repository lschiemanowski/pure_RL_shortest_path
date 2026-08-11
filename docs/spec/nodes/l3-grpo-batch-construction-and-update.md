# GRPO batch construction and update

Batch construction concatenates each prompt with the completion actually
sampled from it. Next-token labels are the shifted concatenated sequence, but
the action mask selects only completion positions; prompt tokens therefore
provide context and never act as supervised targets. The packed batch retains
rollout rewards, adjacent group identities, and the valid-action sets computed
for the same sampled positions.

Before any optimizer step, the current policy and frozen reference policy score
every sampled action. These old-policy and reference log probabilities remain
fixed for every update epoch. Rewards are normalized within each fixed-size
comparison group using the population standard deviation; a group with standard
deviation at or below the declared epsilon receives zero advantages.

At each sampled token, GRPO uses the clipped likelihood-ratio objective. Policy
and sampled-token KL terms are first averaged over the actions in each
completion and then equally over completions. For
\(\Delta=\log\pi_{\mathrm{ref}}-\log\pi_\theta\), the nonnegative sampled KL
estimator is

\[
e^\Delta-\Delta-1.
\]

The valid-next-token loss is averaged globally over positions having a nonempty
valid set. The complete loss is

\[
L=L_{\mathrm{policy}}+\beta_{\mathrm{KL}}L_{\mathrm{KL}}
  +\beta_{\mathrm{valid}}L_{\mathrm{valid}}.
\]

Microbatches preserve these global normalizations. Their gradients accumulate
over the whole rollout batch, followed by one global gradient clipping operation
and one optimizer step per update epoch. Only the current policy is updated.
