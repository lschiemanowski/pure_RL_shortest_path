"""Learned values, generalized advantages, and PPO-Clip updates."""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real

import torch
from torch import nn

from .model import Transformer
from .rl import PackedRollouts


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _bounded_real(
    value: object, name: str, *, minimum: float, maximum: float, include_minimum: bool
) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    lower_ok = result >= minimum if include_minimum else result > minimum
    if not math.isfinite(result) or not lower_ok or result > maximum:
        bracket = "[" if include_minimum else "("
        raise ValueError(f"{name} must lie in {bracket}{minimum}, {maximum}]")
    return result


@dataclass(frozen=True)
class PPOConfig:
    update_epochs: int = 4
    microbatch_size: int = 8
    clip_epsilon: float = 0.2
    value_clip_epsilon: float | None = 0.2
    gamma: float = 1.0
    gae_lambda: float = 0.95
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.0
    kl_coefficient: float = 0.0
    valid_coefficient: float = 0.0
    gradient_clip: float = 1.0
    advantage_epsilon: float = 1e-6
    suppress_zero_reward_policy_updates: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.suppress_zero_reward_policy_updates, bool):
            raise TypeError("suppress_zero_reward_policy_updates must be a boolean")
        for name in ("update_epochs", "microbatch_size"):
            object.__setattr__(self, name, _positive_int(getattr(self, name), name))
        for name in ("gamma", "gae_lambda"):
            object.__setattr__(
                self,
                name,
                _bounded_real(
                    getattr(self, name),
                    name,
                    minimum=0.0,
                    maximum=1.0,
                    include_minimum=True,
                ),
            )
        clip_epsilon = _bounded_real(
            self.clip_epsilon,
            "clip_epsilon",
            minimum=0.0,
            maximum=1.0,
            include_minimum=False,
        )
        if clip_epsilon == 1.0:
            raise ValueError("clip_epsilon must be less than one")
        object.__setattr__(self, "clip_epsilon", clip_epsilon)
        value_clip = self.value_clip_epsilon
        if value_clip is not None:
            value_clip = _bounded_real(
                value_clip,
                "value_clip_epsilon",
                minimum=0.0,
                maximum=float("inf"),
                include_minimum=False,
            )
        object.__setattr__(self, "value_clip_epsilon", value_clip)
        for name in (
            "value_coefficient",
            "entropy_coefficient",
            "kl_coefficient",
            "valid_coefficient",
        ):
            object.__setattr__(
                self,
                name,
                _bounded_real(
                    getattr(self, name),
                    name,
                    minimum=0.0,
                    maximum=float("inf"),
                    include_minimum=True,
                ),
            )
        for name in ("gradient_clip", "advantage_epsilon"):
            object.__setattr__(
                self,
                name,
                _bounded_real(
                    getattr(self, name),
                    name,
                    minimum=0.0,
                    maximum=float("inf"),
                    include_minimum=False,
                ),
            )


class ValueHead(nn.Module):
    """Training-only scalar prediction from final normalized hidden states."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.projection = nn.Linear(_positive_int(hidden_size, "hidden_size"), 1)
        nn.init.normal_(self.projection.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.projection.bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.ndim != 3:
            raise ValueError("hidden_states must have shape [batch, length, hidden]")
        return self.projection(hidden_states).squeeze(-1).float()


@dataclass(frozen=True)
class PPOBatch:
    packed: PackedRollouts
    old_log_probabilities: torch.Tensor
    old_values: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor

    def slice(self, start: int, end: int) -> "PPOBatch":
        return PPOBatch(
            self.packed.slice(start, end),
            self.old_log_probabilities[start:end],
            self.old_values[start:end],
            self.advantages[start:end],
            self.returns[start:end],
        )


@dataclass(frozen=True)
class PPOMetrics:
    policy_loss: float
    value_loss: float
    entropy: float
    sampled_kl: float
    valid_loss: float
    valid_mass: float
    clip_fraction: float
    explained_variance: float
    total_loss: float
    gradient_norm: float


def generalized_advantages(
    terminal_rewards: torch.Tensor,
    values: torch.Tensor,
    action_mask: torch.Tensor,
    *,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute masked GAE with reward only on each trajectory's final action."""

    if terminal_rewards.ndim != 1 or values.ndim != 2:
        raise ValueError("terminal_rewards and values must be a vector and matrix")
    if (
        values.shape != action_mask.shape
        or values.shape[0] != terminal_rewards.shape[0]
    ):
        raise ValueError("values, action_mask, and rewards have incompatible shapes")
    gamma = _bounded_real(
        gamma, "gamma", minimum=0.0, maximum=1.0, include_minimum=True
    )
    gae_lambda = _bounded_real(
        gae_lambda, "gae_lambda", minimum=0.0, maximum=1.0, include_minimum=True
    )
    advantages = torch.zeros_like(values, dtype=torch.float32)
    for row in range(values.shape[0]):
        positions = action_mask[row].nonzero(as_tuple=False).flatten()
        following_advantage = values.new_zeros((), dtype=torch.float32)
        following_value = values.new_zeros((), dtype=torch.float32)
        for reverse_index in range(positions.numel() - 1, -1, -1):
            position = int(positions[reverse_index].item())
            reward = (
                terminal_rewards[row]
                if reverse_index == positions.numel() - 1
                else 0.0
            )
            delta = reward + gamma * following_value - values[row, position]
            following_advantage = delta + gamma * gae_lambda * following_advantage
            advantages[row, position] = following_advantage
            following_value = values[row, position]
    returns = advantages + values.float()
    return advantages, returns


def normalize_advantages(
    advantages: torch.Tensor, action_mask: torch.Tensor, epsilon: float
) -> torch.Tensor:
    """Normalize real actions, retaining raw constant advantages."""

    if advantages.shape != action_mask.shape:
        raise ValueError("advantages and action_mask must have equal shape")
    epsilon = _bounded_real(
        epsilon,
        "epsilon",
        minimum=0.0,
        maximum=float("inf"),
        include_minimum=False,
    )
    selected = advantages[action_mask]
    if selected.numel() == 0:
        raise ValueError("at least one sampled action is required")
    standard_deviation = selected.std(unbiased=False)
    if (
        not bool(torch.isfinite(standard_deviation))
        or float(standard_deviation) <= epsilon
    ):
        return advantages.clone()
    result = torch.zeros_like(advantages)
    result[action_mask] = (selected - selected.mean()) / (
        standard_deviation + epsilon
    )
    return result


def _gathered_log_probabilities(
    logits: torch.Tensor, labels: torch.Tensor
) -> torch.Tensor:
    return logits.float().log_softmax(dim=-1).gather(-1, labels[..., None]).squeeze(-1)


@torch.no_grad()
def prepare_ppo_batch(
    actor: Transformer,
    value_head: ValueHead,
    packed: PackedRollouts,
    config: PPOConfig,
    *,
    autocast_dtype: torch.dtype | None = None,
) -> PPOBatch:
    """Freeze rollout-time policy and value statistics and construct GAE targets."""

    actor.eval()
    value_head.eval()
    old_log_probabilities: list[torch.Tensor] = []
    old_values: list[torch.Tensor] = []
    amp_enabled = autocast_dtype is not None
    for start in range(0, packed.input_ids.shape[0], config.microbatch_size):
        chunk = packed.slice(start, start + config.microbatch_size)
        with torch.autocast(
            device_type=packed.input_ids.device.type,
            dtype=autocast_dtype,
            enabled=amp_enabled,
        ):
            logits, hidden_states = actor.forward_with_hidden_states(
                chunk.input_ids,
                attention_mask=chunk.attention_mask,
            )
            old_log_probabilities.append(
                _gathered_log_probabilities(logits, chunk.labels)
            )
            old_values.append(value_head(hidden_states))
    log_probabilities = torch.cat(old_log_probabilities)
    values = torch.cat(old_values)
    advantages, returns = generalized_advantages(
        packed.rewards,
        values,
        packed.action_mask,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
    )
    advantages = normalize_advantages(
        advantages, packed.action_mask, config.advantage_epsilon
    )
    return PPOBatch(packed, log_probabilities, values, advantages, returns)


@torch.no_grad()
def _reference_log_probabilities(
    reference: Transformer,
    batch: PPOBatch,
    config: PPOConfig,
    autocast_dtype: torch.dtype | None,
) -> torch.Tensor:
    chunks: list[torch.Tensor] = []
    amp_enabled = autocast_dtype is not None
    reference.eval()
    for start in range(0, batch.packed.input_ids.shape[0], config.microbatch_size):
        chunk = batch.packed.slice(start, start + config.microbatch_size)
        with torch.autocast(
            device_type=chunk.input_ids.device.type,
            dtype=autocast_dtype,
            enabled=amp_enabled,
        ):
            logits = reference(
                chunk.input_ids, attention_mask=chunk.attention_mask
            ).logits
        chunks.append(_gathered_log_probabilities(logits, chunk.labels))
    return torch.cat(chunks)


def ppo_update(
    actor: Transformer,
    value_head: ValueHead,
    reference: Transformer,
    optimizer: torch.optim.Optimizer,
    batch: PPOBatch,
    config: PPOConfig,
    *,
    autocast_dtype: torch.dtype | None = None,
) -> PPOMetrics:
    """Optimize a fixed on-policy actor-value batch with PPO-Clip."""

    if batch.packed.input_ids.shape[0] == 0:
        raise ValueError("PPO batch must be nonempty")
    reference_log_probabilities = _reference_log_probabilities(
        reference, batch, config, autocast_dtype
    )
    action_count = int(batch.packed.action_mask.sum().item())
    valid_position_count = int(
        batch.packed.valid_next_mask.any(dim=-1).sum().item()
    )
    if action_count == 0:
        raise ValueError("PPO batch must contain sampled actions")
    suppress_policy_update = (
        config.suppress_zero_reward_policy_updates
        and not bool(batch.packed.rewards.ne(0).any())
    )
    totals = {
        name: 0.0
        for name in (
            "policy_loss",
            "value_loss",
            "entropy",
            "sampled_kl",
            "valid_loss",
            "valid_mass",
            "clip_fraction",
            "gradient_norm",
        )
    }
    actor.train()
    value_head.train()
    amp_enabled = autocast_dtype is not None
    batch_size = batch.packed.input_ids.shape[0]
    for _ in range(config.update_epochs):
        optimizer.zero_grad(set_to_none=True)
        epoch = {name: 0.0 for name in totals if name != "gradient_norm"}
        for start in range(0, batch_size, config.microbatch_size):
            end = min(start + config.microbatch_size, batch_size)
            chunk = batch.slice(start, end)
            packed = chunk.packed
            with torch.autocast(
                device_type=packed.input_ids.device.type,
                dtype=autocast_dtype,
                enabled=amp_enabled,
            ):
                logits, hidden_states = actor.forward_with_hidden_states(
                    packed.input_ids,
                    attention_mask=packed.attention_mask,
                )
                current_log_probabilities = _gathered_log_probabilities(
                    logits, packed.labels
                )
                values = value_head(hidden_states)
                log_ratio = current_log_probabilities - chunk.old_log_probabilities
                ratio = log_ratio.exp()
                clipped_ratio = ratio.clamp(
                    1.0 - config.clip_epsilon, 1.0 + config.clip_epsilon
                )
                surrogate = torch.minimum(
                    ratio * chunk.advantages, clipped_ratio * chunk.advantages
                )
                if suppress_policy_update:
                    policy_loss = surrogate[packed.action_mask].sum() * 0.0
                else:
                    policy_loss = -surrogate[packed.action_mask].sum() / action_count

                value_error = (values - chunk.returns).square()
                if config.value_clip_epsilon is not None:
                    clipped_values = chunk.old_values + (
                        values - chunk.old_values
                    ).clamp(-config.value_clip_epsilon, config.value_clip_epsilon)
                    value_error = torch.maximum(
                        value_error, (clipped_values - chunk.returns).square()
                    )
                value_loss = 0.5 * value_error[packed.action_mask].sum() / action_count

                log_probabilities = logits.float().log_softmax(dim=-1)
                probabilities = log_probabilities.exp()
                token_entropy = -(probabilities * log_probabilities).sum(dim=-1)
                entropy = token_entropy[packed.action_mask].sum() / action_count

                reference_delta = (
                    reference_log_probabilities[start:end]
                    - current_log_probabilities
                )
                sampled_kl_tokens = reference_delta.exp() - reference_delta - 1.0
                sampled_kl = (
                    sampled_kl_tokens[packed.action_mask].sum() / action_count
                )
                clipped = (ratio - 1.0).abs() > config.clip_epsilon
                clip_fraction = (
                    clipped[packed.action_mask].float().sum() / action_count
                )

                valid_positions = packed.valid_next_mask.any(dim=-1)
                if bool(valid_positions.any()):
                    valid_logits = logits.float().masked_fill(
                        ~packed.valid_next_mask, -torch.inf
                    )
                    log_valid_mass = valid_logits.logsumexp(
                        dim=-1
                    ) - logits.float().logsumexp(dim=-1)
                    denominator = max(valid_position_count, 1)
                    valid_loss = -log_valid_mass[valid_positions].sum() / denominator
                    valid_mass = (
                        log_valid_mass[valid_positions].exp().sum() / denominator
                    )
                else:
                    valid_loss = logits.sum() * 0.0
                    valid_mass = logits.sum() * 0.0

                loss = (
                    policy_loss
                    + config.value_coefficient * value_loss
                    - config.entropy_coefficient * entropy
                    + config.kl_coefficient * sampled_kl
                    + config.valid_coefficient * valid_loss
                )
            loss.backward()
            for name, value in (
                ("policy_loss", policy_loss),
                ("value_loss", value_loss),
                ("entropy", entropy),
                ("sampled_kl", sampled_kl),
                ("valid_loss", valid_loss),
                ("valid_mass", valid_mass),
                ("clip_fraction", clip_fraction),
            ):
                epoch[name] += float(value.detach())
        parameters = [*actor.parameters(), *value_head.parameters()]
        gradient_norm = nn.utils.clip_grad_norm_(parameters, config.gradient_clip)
        optimizer.step()
        for name, value in epoch.items():
            totals[name] += value
        totals["gradient_norm"] += float(gradient_norm)

    scale = 1.0 / config.update_epochs
    metrics = {name: value * scale for name, value in totals.items()}
    selected_returns = batch.returns[batch.packed.action_mask]
    selected_values = batch.old_values[batch.packed.action_mask]
    return_variance = selected_returns.var(unbiased=False)
    explained_variance = (
        1.0
        - float(
            (selected_returns - selected_values).var(unbiased=False)
            / return_variance
        )
        if float(return_variance) > config.advantage_epsilon
        else 0.0
    )
    total_loss = (
        metrics["policy_loss"]
        + config.value_coefficient * metrics["value_loss"]
        - config.entropy_coefficient * metrics["entropy"]
        + config.kl_coefficient * metrics["sampled_kl"]
        + config.valid_coefficient * metrics["valid_loss"]
    )
    return PPOMetrics(
        policy_loss=metrics["policy_loss"],
        value_loss=metrics["value_loss"],
        entropy=metrics["entropy"],
        sampled_kl=metrics["sampled_kl"],
        valid_loss=metrics["valid_loss"],
        valid_mass=metrics["valid_mass"],
        clip_fraction=metrics["clip_fraction"],
        explained_variance=explained_variance,
        total_loss=total_loss,
        gradient_norm=metrics["gradient_norm"],
    )
