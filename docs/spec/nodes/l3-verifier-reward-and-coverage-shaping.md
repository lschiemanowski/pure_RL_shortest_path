# Verifier reward and coverage shaping

Reward computation consumes deterministic outcome-verifier facts. It assigns
base reward zero to a malformed completion, \(0.05\) to a well-formed completion
without a valid answer path, one to a shortest answer path, and
\(0.5d^*/\ell\) to a valid non-shortest path of length \(\ell\), where \(d^*\)
is the independently computed shortest distance.

For a valid answer only, reasoning coverage \(c\) is the fraction of distinct
answer edges that also occur among legally traversed reasoning edges. Malformed
and invalid answers receive coverage zero. The total reward is

\[
r = r_{\mathrm{base}} + \lambda_{\mathrm{coverage}}c.
\]

The nonnegative shaping coefficient is configured independently and may be
zero. Every rollout records the base reward, coverage, coefficient, and total
reward separately, so verifier-reward-only and shaped experiments remain
distinguishable.
