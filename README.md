# Pure-RL shortest paths

This repo contains code to train a small transformer to find shortest paths in
graphs. The transformer is initialized with random weights. The aim is for the
model to learn a search algorithm using chain of thought (CoT), without being
given an algorithm to follow. Reinforcement learning provides feedback on its attempts, but no
instructions for how to organize the search.

Given an undirected, unweighted graph and a pair of nodes, the model generates
walks through the graph as its CoT, followed by an answer path. A `JUMP` token
allows it to start a new walk. The answer is checked against breadth-first search.

For example, consider a graph with two routes from node 0 to node 4:
`0–1–2–4` and `0–3–4`. The input is:

```text
BOS EDGES 0 1 1 2 2 4 0 3 3 4 QUERY 0 4 BEGIN_REASON 0
```

After `EDGES`, each pair of node labels describes an undirected edge.
`QUERY 0 4` asks for a shortest path from 0 to 4. The input ends with
`BEGIN_REASON 0`, supplying the starting node for the first reasoning walk.
Each node label and each marker is a single token.

One correct completion would be:

```text
1 2 4 JUMP 0 3 4 END_REASON BEGIN_ANSWER 0 3 4 END_ANSWER EOS
```

Here, the reasoning first follows `0–1–2–4`, then restarts at 0 and follows
`0–3–4`. The final answer is `0–3–4`, a shortest path of length 2.

## Training

We trained a model with 12 layers and 12.6M parameters using GRPO on randomly
generated graphs. Training starts with small graphs and short paths, increasing
the difficulty as validation improves. No pretrained weights or example
solutions are used.

The reward favors shortest paths and gives partial credit for valid longer
paths. We also reward covering answer edges during exploration and use an
auxiliary loss that encourages locally legal next tokens. These signals leave
the choice of search strategy to the model.

Training uses the stages below, numbered from zero. Each stage specifies the
number of nodes and edges and the allowed shortest-path lengths, measured in
edges. Problems are sampled mainly from the current stage, with some sampling
from earlier and later stages.

Every 100 training steps, we evaluate on 256 problems from the current stage
using greedy decoding. Promotion requires at least 80% shortest-path success
(205/256) in three consecutive evaluations. A result below the threshold resets
the count. The table gives the step at which we entered each stage; promotion
means passing the preceding stage's criterion.

| Stage | Nodes | Edges | Shortest-path length | Promoted into stage at step |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 8 | 7 | 1 | Initial stage |
| 1 | 8 | 9 | 2 | 600 |
| 2 | 12 | 14 | 2–3 | 11,100 |
| 3 | 16 | 20 | 3 | 49,500 |
| 4 | 24 | 32 | 3–4 | 54,000 |
| 5 | 32 | 44 | 4 | 87,000 |
| 6 | 48 | 64 | 4–5 | 90,000 |
| 7 | 64 | 88 | 5 | 106,500 |
| 8 | 80 | 112 | 5–6 | 111,700 |
| 9 | 96 | 136 | 6 | Not reached |

These steps come from the training logs across successive resumed runs, during
which we adjusted training settings. Within stage 8, performance reached a
plateau which we seemingly could not break.

## How to run this

You will need [uv](https://docs.astral.sh/uv/getting-started/installation/) and a
CUDA-capable GPU. Reaching stage 8 took about 8.4 days of elapsed time on an
RTX 5090. Run the following command from the root of this repo:

```bash
uv run --locked pure-rl-shortest-path train experiments/train.toml
```
