# Sterile-repetition reward penalty

This reward term measures unproductive repetition in the reasoning trace. Let
\(e_1,\ldots,e_T\) be the ordered legal directed transitions supplied by outcome
verification, and let \(D_t\) be the set of distinct undirected graph edges
traversed through position \(t\). If \(\tau(t)\) is the preceding occurrence of
the same directed transition, occurrence \(t\) is sterile exactly when

\[
\tau(t)\text{ exists}\quad\text{and}\quad D_{t-1}=D_{\tau(t)}.
\]

The history is global across reasoning walks, so a restart does not erase it.
A repeated transition is not sterile when a new undirected edge was discovered
since its preceding occurrence. Restarts and illegal moves are absent from the
input transition sequence and therefore affect neither count.

The all-transition rate \(\rho_{\mathrm{all}}\) divides sterile occurrences by
\(T\), returning zero for an empty sequence. The off-answer rate
\(\rho_{\mathrm{off\mbox{-}answer}}\) first excludes from both numerator and
denominator transitions whose undirected edge belongs to the submitted valid
answer. If the answer is invalid, no edges are excluded; an empty denominator
again gives zero. This uses the submitted answer rather than oracle knowledge
of which graph edges belong to any shortest path.

The configured mode \(m\in\{\mathrm{all},\mathrm{off\mbox{-}answer}\}\) selects
one rate without suppressing either diagnostic. For a nonnegative independently
configured coefficient, the final reward is

\[
r = r_{\mathrm{covered}}
  - \mathbf{1}_{\mathrm{valid}}
    \lambda_{\mathrm{sterile}}\rho_m.
\]

Malformed and invalid answers therefore retain repetition diagnostics but
receive no penalty. Every rollout records both rates, the selected mode and
coefficient, the applied penalty, and total reward separately.
