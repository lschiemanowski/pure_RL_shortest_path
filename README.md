# Pure-RL shortest paths

This repository trains a decoder-only transformer from scratch to find shortest
paths in synthetic graphs using only its sampled completions and graph-derived
reinforcement-learning feedback.

Install the local command in an environment containing PyTorch:

```bash
python -m pip install -e .
```

Start, resume, and evaluate TOML-declared experiments with:

```bash
pure-rl-shortest-path train experiments/baseline.toml
pure-rl-shortest-path resume runs/<run>/checkpoints/latest.pt
pure-rl-shortest-path evaluate \
  runs/<run>/checkpoints/latest.pt experiments/evaluation.toml
```

The run directory retains the source and resolved configuration, provenance,
append-only metrics and representative completions, and atomic checkpoints.
