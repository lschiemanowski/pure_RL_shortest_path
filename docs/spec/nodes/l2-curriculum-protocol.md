# Curriculum protocol

The curriculum protocol defines an ordered sequence of graph-problem
configurations and a current frontier within that sequence. At each training
step, it independently samples the stage of every training problem from a
declared asymmetric probability distribution over all curriculum stages. The
distribution is peaked at the frontier, has a broad tail over preceding stages,
and attenuates sharply over succeeding stages. The trainer receives the
resulting problems but does not choose or alter their distribution.

The protocol evaluates the current policy on independently generated validation
problems from the frontier stage. It advances the frontier by one stage only
after shortest-path performance reaches a configured threshold for the required
number of consecutive validations. Frontier advancement is monotone and stops
at the final stage. Training and validation use separate random streams,
validation examples never become training rollouts, and every validation result
and frontier transition is recorded.
