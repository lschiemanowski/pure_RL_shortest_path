# Curriculum protocol

The curriculum protocol defines an ordered sequence of graph-problem
configurations. At each training step it selects the active stage and any
declared mixture of problems from the following stage, then supplies those
problems to the reinforcement-learning trainer. The trainer does not choose or
alter this distribution.

The protocol evaluates the current policy on independently generated validation
problems. Once shortest-path performance reaches a configured interpolation
threshold, it may gradually increase the fraction of next-stage training
problems up to a declared maximum. It advances fully to the next stage only
after the mixture is ready and the configured validation threshold has been met
for the required number of consecutive validations. Training and validation use
separate random streams, and validation examples never become training
rollouts. Each transition records the completed stage and new curriculum
position.
