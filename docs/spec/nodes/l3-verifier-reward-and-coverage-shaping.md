# Verifier reward and reasoning shaping

Reward computation consumes deterministic outcome-verifier facts. It assigns
base reward zero to a malformed completion, \(0.05\) to a well-formed completion
without a valid answer path, one to a shortest answer path, and
\(0.5d^*/\ell\) to a valid non-shortest path of length \(\ell\), where \(d^*\)
is the independently computed shortest distance.

For a valid answer only, reasoning coverage \(c\) is the fraction of distinct
answer edges that also occur among legally traversed reasoning edges. Malformed
and invalid answers receive coverage zero.

Outcome verification also supplies the sterile-repetition rates
\(\rho_{\mathrm{all}}\) and \(\rho_{\mathrm{off\mbox{-}answer}\). The configured
mode \(m\) selects one without changing either diagnostic. The total reward is

\[
r = r_{\mathrm{base}}
  + \lambda_{\mathrm{coverage}}c
  - \mathbf{1}_{\mathrm{valid}}
    \lambda_{\mathrm{sterile}}\rho_m.
\]

Both shaping coefficients are nonnegative, configured independently, and may be
zero. The repetition penalty is zero for malformed or invalid answers. Every
rollout records the base reward, coverage bonus inputs, both repetition rates,
selected repetition mode, coefficients, applied penalty, and total reward
separately, so verifier-reward-only and shaped experiments remain
distinguishable.
