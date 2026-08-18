"""On-policy grouped rollouts, graph-derived signals, and GRPO updates."""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
from typing import Literal, Sequence

import torch
from torch import nn

from .model import Transformer, generate_tokens
from .task import (
    BEGIN_ANSWER,
    END_ANSWER,
    END_REASON,
    EOS,
    JUMP,
    PAD,
    GraphExample,
    OutcomeFacts,
    Vocabulary,
    canonical_edge,
    verify_completion,
)


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _nonnegative_real(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return result


@dataclass(frozen=True)
class RolloutSamplingConfig:
    """Sampling parameters for one on-policy training batch."""

    group_size: int
    max_new_tokens: int
    generation_batch_size: int | None = None
    temperature: float = 1.0
    top_p: float = 1.0

    def __post_init__(self) -> None:
        for name in ("group_size", "max_new_tokens"):
            object.__setattr__(self, name, _positive_int(getattr(self, name), name))
        batch_size = self.generation_batch_size
        if batch_size is not None:
            batch_size = _positive_int(batch_size, "generation_batch_size")
        temperature = _nonnegative_real(self.temperature, "temperature")
        top_p = _nonnegative_real(self.top_p, "top_p")
        if temperature != 1.0 or top_p != 1.0:
            raise ValueError(
                "training rollouts require temperature=1 and top_p=1 so samples "
                "come from the policy distribution used by GRPO"
            )
        object.__setattr__(self, "generation_batch_size", batch_size)
        object.__setattr__(self, "temperature", temperature)
        object.__setattr__(self, "top_p", top_p)


@dataclass(frozen=True)
class SampledCompletion:
    example: GraphExample
    completion: tuple[int, ...]
    group_id: int
    terminated_by_eos: bool


def sample_policy_tokens(
    logits: torch.Tensor, generator: torch.Generator | None = None
) -> torch.Tensor:
    """Draw one categorical action per row from the complete policy softmax."""

    if logits.ndim != 2 or logits.shape[0] == 0 or logits.shape[1] == 0:
        raise ValueError("logits must have shape [positive_batch, positive_vocabulary]")
    probabilities = logits.float().softmax(dim=-1)
    if generator is not None and probabilities.device.type != "cpu":
        return torch.multinomial(
            probabilities.cpu(), 1, generator=generator
        ).squeeze(1).to(logits.device)
    return torch.multinomial(probabilities, 1, generator=generator).squeeze(1)


@torch.no_grad()
def collect_grouped_completions(
    model: Transformer,
    examples: Sequence[GraphExample],
    config: RolloutSamplingConfig,
    *,
    device: torch.device,
    generator: torch.Generator | None = None,
    autocast_dtype: torch.dtype | None = None,
) -> list[SampledCompletion]:
    """Sample adjacent completion groups without task-specific token constraints."""

    expanded = [
        (group_id, example)
        for group_id, example in enumerate(examples)
        for _ in range(config.group_size)
    ]
    generated = generate_tokens(
        model,
        [example.prompt for _, example in expanded],
        max_new_tokens=config.max_new_tokens,
        batch_size=config.generation_batch_size,
        pad_token=PAD,
        eos_token=EOS,
        device=device,
        select_next=lambda logits: sample_policy_tokens(logits, generator),
        autocast_dtype=autocast_dtype,
    )
    return [
        SampledCompletion(
            example=example,
            completion=completion,
            group_id=group_id,
            terminated_by_eos=bool(completion and completion[-1] == EOS),
        )
        for (group_id, example), completion in zip(expanded, generated)
    ]


@dataclass(frozen=True)
class RewardComponents:
    base_reward: float
    reasoning_coverage: float
    coverage_coefficient: float
    sterile_repetition_rate_all: float
    sterile_repetition_rate_off_answer: float
    sterile_repetition_mode: Literal["all", "off_answer"]
    sterile_repetition_coefficient: float
    sterile_repetition_penalty: float
    total_reward: float


@dataclass(frozen=True)
class SterileRepetitionFacts:
    legal_transitions: int
    sterile_repetitions_all: int
    repetition_rate_all: float
    off_answer_legal_transitions: int
    sterile_repetitions_off_answer: int
    repetition_rate_off_answer: float


@dataclass(frozen=True)
class Rollout:
    sample: SampledCompletion
    outcome: OutcomeFacts
    reward: RewardComponents

    @property
    def example(self) -> GraphExample:
        return self.sample.example

    @property
    def completion(self) -> tuple[int, ...]:
        return self.sample.completion

    @property
    def group_id(self) -> int:
        return self.sample.group_id


def verifier_reward_and_coverage(
    outcome: OutcomeFacts, coverage_coefficient: float
) -> tuple[float, float, float, float]:
    """Return base reward, coverage, coefficient, and applied coverage bonus."""
    coefficient = _nonnegative_real(coverage_coefficient, "coverage_coefficient")
    if not outcome.format_ok:
        base_reward = 0.0
    elif not outcome.valid_path:
        base_reward = 0.05
    elif outcome.shortest:
        base_reward = 1.0
    else:
        assert outcome.answer_length is not None and outcome.answer_length > 0
        base_reward = 0.5 * outcome.shortest_distance / outcome.answer_length
    coverage = (
        outcome.reasoning.answer_edge_coverage
        if outcome.valid_path and outcome.reasoning is not None
        else 0.0
    )
    return base_reward, coverage, coefficient, coefficient * coverage


def sterile_repetition_facts(outcome: OutcomeFacts) -> SterileRepetitionFacts:
    """Measure reward-specific unproductive repetition in a reasoning trace."""

    if outcome.reasoning is None:
        return SterileRepetitionFacts(0, 0, 0.0, 0, 0, 0.0)
    answer_edges = (
        {
            canonical_edge(u, v)
            for u, v in zip(outcome.parsed.answer, outcome.parsed.answer[1:])
        }
        if outcome.valid_path
        else set()
    )
    discovered: set[tuple[int, int]] = set()
    previous_discovery: dict[tuple[int, int], frozenset[tuple[int, int]]] = {}
    legal = sterile_all = off_answer = sterile_off_answer = 0
    for directed in outcome.reasoning.legal_directed_transitions:
        edge = canonical_edge(*directed)
        prior = previous_discovery.get(directed)
        sterile = prior is not None and prior == frozenset(discovered)
        discovered.add(edge)
        previous_discovery[directed] = frozenset(discovered)
        legal += 1
        sterile_all += sterile
        if edge not in answer_edges:
            off_answer += 1
            sterile_off_answer += sterile
    return SterileRepetitionFacts(
        legal_transitions=legal,
        sterile_repetitions_all=sterile_all,
        repetition_rate_all=sterile_all / legal if legal else 0.0,
        off_answer_legal_transitions=off_answer,
        sterile_repetitions_off_answer=sterile_off_answer,
        repetition_rate_off_answer=(sterile_off_answer / off_answer if off_answer else 0.0),
    )


def reward_from_outcome(
    outcome: OutcomeFacts,
    coverage_coefficient: float = 0.0,
    sterile_repetition_coefficient: float = 0.0,
    sterile_repetition_mode: Literal["all", "off_answer"] = "off_answer",
) -> RewardComponents:
    """Compose base reward and separately configured reasoning shaping."""

    base_reward, coverage, coverage_value, coverage_bonus = (
        verifier_reward_and_coverage(outcome, coverage_coefficient)
    )
    sterile_coefficient = _nonnegative_real(
        sterile_repetition_coefficient, "sterile_repetition_coefficient"
    )
    if sterile_repetition_mode not in ("all", "off_answer"):
        raise ValueError("sterile_repetition_mode must be all or off_answer")
    repetition = sterile_repetition_facts(outcome)
    selected_repetition = (
        repetition.repetition_rate_all
        if sterile_repetition_mode == "all"
        else repetition.repetition_rate_off_answer
    )
    sterile_penalty = (
        sterile_coefficient * selected_repetition if outcome.valid_path else 0.0
    )
    total = base_reward + coverage_bonus - sterile_penalty
    return RewardComponents(
        base_reward=base_reward,
        reasoning_coverage=coverage,
        coverage_coefficient=coverage_value,
        sterile_repetition_rate_all=repetition.repetition_rate_all,
        sterile_repetition_rate_off_answer=repetition.repetition_rate_off_answer,
        sterile_repetition_mode=sterile_repetition_mode,
        sterile_repetition_coefficient=sterile_coefficient,
        sterile_repetition_penalty=sterile_penalty,
        total_reward=total,
    )


def evaluate_samples(
    samples: Sequence[SampledCompletion],
    vocabulary: Vocabulary,
    coverage_coefficient: float = 0.0,
    sterile_repetition_coefficient: float = 0.0,
    sterile_repetition_mode: Literal["all", "off_answer"] = "off_answer",
    minimum_reason_tokens: int = 1,
) -> list[Rollout]:
    """Verify sampled completions and attach their separately recorded rewards."""

    rollouts: list[Rollout] = []
    for sample in samples:
        outcome = verify_completion(
            sample.example,
            sample.completion,
            vocabulary,
            minimum_reason_tokens,
        )
        rollouts.append(
            Rollout(
                sample,
                outcome,
                reward_from_outcome(
                    outcome,
                    coverage_coefficient,
                    sterile_repetition_coefficient,
                    sterile_repetition_mode,
                ),
            )
        )
    return rollouts


def valid_next_token_sets(
    example: GraphExample,
    completion: Sequence[int],
    vocabulary: Vocabulary,
    minimum_reason_tokens: int = 1,
) -> tuple[frozenset[int], ...]:
    """Return every locally legal next-token set along a sampled completion.

    An invalid sampled action is rejected by this auxiliary state machine and
    leaves its state unchanged.  It never changes the actual completion.
    """

    minimum_reason_tokens = _positive_int(
        minimum_reason_tokens, "minimum_reason_tokens"
    )
    results: list[frozenset[int]] = []
    active_tokens = {
        vocabulary.node_token(label) for label in example.active_labels
    }
    state = "reasoning"
    current = example.source
    after_jump = False
    walk_edges = 0
    accepted_reasoning_tokens = 0
    visited_answer_nodes: set[int] = set()

    for action in completion:
        if state == "reasoning":
            if after_jump:
                valid = set(active_tokens)
            else:
                valid = {
                    vocabulary.node_token(label)
                    for label in example.adjacency[current]
                }
                if walk_edges > 0:
                    valid.add(JUMP)
                if accepted_reasoning_tokens >= minimum_reason_tokens:
                    valid.add(END_REASON)
        elif state == "begin_answer":
            valid = {BEGIN_ANSWER}
        elif state == "answer_source":
            valid = {vocabulary.node_token(example.source)}
        elif state == "answer_walk":
            if current == example.target:
                valid = {END_ANSWER}
            else:
                valid = {
                    vocabulary.node_token(label)
                    for label in example.adjacency[current]
                    if label not in visited_answer_nodes
                }
        elif state == "eos":
            valid = {EOS}
        else:
            valid = set()

        results.append(frozenset(valid))
        if action not in valid:
            continue

        if state == "reasoning":
            if after_jump:
                label = vocabulary.token_node(action)
                assert label is not None
                current = label
                after_jump = False
                walk_edges = 0
                accepted_reasoning_tokens += 1
            elif action == JUMP:
                after_jump = True
                accepted_reasoning_tokens += 1
            elif action == END_REASON:
                state = "begin_answer"
            else:
                label = vocabulary.token_node(action)
                assert label is not None
                current = label
                walk_edges += 1
                accepted_reasoning_tokens += 1
        elif state == "begin_answer":
            state = "answer_source"
        elif state == "answer_source":
            current = example.source
            visited_answer_nodes = {current}
            state = "answer_walk"
        elif state == "answer_walk":
            if action == END_ANSWER:
                state = "eos"
            else:
                label = vocabulary.token_node(action)
                assert label is not None
                current = label
                visited_answer_nodes.add(label)
        elif state == "eos":
            state = "done"

    return tuple(results)


@dataclass(frozen=True)
class PackedRollouts:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor
    action_mask: torch.Tensor
    valid_next_mask: torch.Tensor
    rewards: torch.Tensor
    group_ids: torch.Tensor

    def slice(self, start: int, end: int) -> "PackedRollouts":
        return PackedRollouts(
            self.input_ids[start:end],
            self.attention_mask[start:end],
            self.labels[start:end],
            self.action_mask[start:end],
            self.valid_next_mask[start:end],
            self.rewards[start:end],
            self.group_ids[start:end],
        )


def pack_rollouts(
    rollouts: Sequence[Rollout],
    vocabulary: Vocabulary,
    *,
    device: torch.device,
    minimum_reason_tokens: int = 1,
) -> PackedRollouts:
    """Pack prompts and sampled actions without creating target completions."""

    if not rollouts:
        raise ValueError("at least one rollout is required")
    maximum = max(
        len(rollout.example.prompt) + len(rollout.completion)
        for rollout in rollouts
    )
    input_ids = torch.full((len(rollouts), maximum - 1), PAD, dtype=torch.long)
    labels = torch.full_like(input_ids, PAD)
    attention_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    action_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    valid_next_mask = torch.zeros(
        (len(rollouts), maximum - 1, vocabulary.size), dtype=torch.bool
    )

    for row, rollout in enumerate(rollouts):
        sequence = (*rollout.example.prompt, *rollout.completion)
        length = len(sequence)
        input_ids[row, : length - 1] = torch.tensor(sequence[:-1])
        labels[row, : length - 1] = torch.tensor(sequence[1:])
        attention_mask[row, : length - 1] = True
        first_action = len(rollout.example.prompt) - 1
        action_mask[row, first_action : length - 1] = True
        valid_sets = valid_next_token_sets(
            rollout.example,
            rollout.completion,
            vocabulary,
            minimum_reason_tokens,
        )
        for completion_index, valid in enumerate(valid_sets):
            if valid:
                position = first_action + completion_index
                valid_next_mask[row, position, list(valid)] = True

    return PackedRollouts(
        input_ids.to(device),
        attention_mask.to(device),
        labels.to(device),
        action_mask.to(device),
        valid_next_mask.to(device),
        torch.tensor(
            [rollout.reward.total_reward for rollout in rollouts],
            dtype=torch.float32,
            device=device,
        ),
        torch.tensor(
            [rollout.group_id for rollout in rollouts],
            dtype=torch.long,
            device=device,
        ),
    )


@dataclass(frozen=True)
class GRPOConfig:
    group_size: int
    update_epochs: int = 1
    microbatch_size: int = 8
    clip_epsilon: float = 0.2
    kl_coefficient: float = 0.0
    valid_coefficient: float = 0.0
    gradient_clip: float = 1.0
    advantage_epsilon: float = 1e-6

    def __post_init__(self) -> None:
        for name in ("group_size", "update_epochs", "microbatch_size"):
            object.__setattr__(self, name, _positive_int(getattr(self, name), name))
        for name in (
            "clip_epsilon",
            "kl_coefficient",
            "valid_coefficient",
            "gradient_clip",
            "advantage_epsilon",
        ):
            object.__setattr__(
                self, name, _nonnegative_real(getattr(self, name), name)
            )
        if self.clip_epsilon >= 1.0:
            raise ValueError("clip_epsilon must be less than one")
        if self.gradient_clip == 0.0:
            raise ValueError("gradient_clip must be positive")
        if self.advantage_epsilon == 0.0:
            raise ValueError("advantage_epsilon must be positive")


@dataclass(frozen=True)
class UpdateMetrics:
    policy_loss: float
    sampled_kl: float
    valid_loss: float
    valid_mass: float
    total_loss: float
    zero_variance_group_fraction: float
    gradient_norm: float


def group_relative_advantages(
    rewards: torch.Tensor,
    group_ids: torch.Tensor,
    group_size: int,
    epsilon: float = 1e-6,
) -> tuple[torch.Tensor, float]:
    """Normalize rewards within adjacent fixed-size comparison groups."""

    if rewards.ndim != 1 or group_ids.shape != rewards.shape:
        raise ValueError("rewards and group_ids must be equally sized vectors")
    group_size = _positive_int(group_size, "group_size")
    epsilon = _nonnegative_real(epsilon, "epsilon")
    if epsilon == 0.0:
        raise ValueError("epsilon must be positive")
    if rewards.numel() == 0 or rewards.numel() % group_size:
        raise ValueError("rollout count must be a positive multiple of group_size")

    advantages = torch.zeros_like(rewards, dtype=torch.float32)
    seen: set[int] = set()
    zero_variance_groups = 0
    for start in range(0, rewards.numel(), group_size):
        group = group_ids[start : start + group_size]
        group_id = int(group[0].item())
        if not bool(group.eq(group_id).all()) or group_id in seen:
            raise ValueError("comparison groups must be adjacent and uniquely identified")
        seen.add(group_id)
        values = rewards[start : start + group_size].float()
        standard_deviation = values.std(unbiased=False)
        if float(standard_deviation.item()) <= epsilon:
            zero_variance_groups += 1
        else:
            advantages[start : start + group_size] = (
                values - values.mean()
            ) / (standard_deviation + epsilon)
    return advantages, zero_variance_groups / len(seen)


def _gathered_log_probabilities(
    logits: torch.Tensor, labels: torch.Tensor
) -> torch.Tensor:
    return (
        logits.float()
        .log_softmax(dim=-1)
        .gather(-1, labels.unsqueeze(-1))
        .squeeze(-1)
    )


@torch.no_grad()
def _score_sampled_actions(
    model: Transformer,
    packed: PackedRollouts,
    microbatch_size: int,
    autocast_dtype: torch.dtype | None,
) -> torch.Tensor:
    chunks: list[torch.Tensor] = []
    amp_enabled = autocast_dtype is not None
    for start in range(0, packed.input_ids.shape[0], microbatch_size):
        chunk = packed.slice(start, start + microbatch_size)
        with torch.autocast(
            device_type=packed.input_ids.device.type,
            dtype=autocast_dtype,
            enabled=amp_enabled,
        ):
            logits = model(
                chunk.input_ids, attention_mask=chunk.attention_mask
            ).logits
        chunks.append(_gathered_log_probabilities(logits, chunk.labels))
    return torch.cat(chunks, dim=0)


def grpo_update(
    model: Transformer,
    reference: Transformer,
    optimizer: torch.optim.Optimizer,
    packed: PackedRollouts,
    config: GRPOConfig,
    *,
    autocast_dtype: torch.dtype | None = None,
) -> UpdateMetrics:
    """Update the current policy from its fixed grouped rollout batch."""

    if packed.input_ids.shape[0] == 0:
        raise ValueError("packed rollout batch must be nonempty")
    model.eval()
    reference.eval()
    old_log_probabilities = _score_sampled_actions(
        model, packed, config.microbatch_size, autocast_dtype
    )
    reference_log_probabilities = _score_sampled_actions(
        reference, packed, config.microbatch_size, autocast_dtype
    )
    advantages, zero_variance_fraction = group_relative_advantages(
        packed.rewards,
        packed.group_ids,
        config.group_size,
        config.advantage_epsilon,
    )
    batch_size = packed.input_ids.shape[0]
    valid_position_count = int(packed.valid_next_mask.any(dim=-1).sum().item())
    amp_enabled = autocast_dtype is not None
    totals = {
        "policy_loss": 0.0,
        "sampled_kl": 0.0,
        "valid_loss": 0.0,
        "valid_mass": 0.0,
        "gradient_norm": 0.0,
    }

    model.train()
    for _ in range(config.update_epochs):
        optimizer.zero_grad(set_to_none=True)
        epoch = {key: 0.0 for key in totals if key != "gradient_norm"}
        for start in range(0, batch_size, config.microbatch_size):
            end = min(start + config.microbatch_size, batch_size)
            chunk = packed.slice(start, end)
            with torch.autocast(
                device_type=packed.input_ids.device.type,
                dtype=autocast_dtype,
                enabled=amp_enabled,
            ):
                logits = model(
                    chunk.input_ids, attention_mask=chunk.attention_mask
                ).logits
                current_log_probabilities = _gathered_log_probabilities(
                    logits, chunk.labels
                )
                token_counts = chunk.action_mask.sum(dim=1).clamp_min(1)

                log_ratio = (
                    current_log_probabilities - old_log_probabilities[start:end]
                )
                ratio = log_ratio.exp()
                clipped_ratio = ratio.clamp(
                    1.0 - config.clip_epsilon, 1.0 + config.clip_epsilon
                )
                advantage = advantages[start:end, None]
                token_objective = torch.minimum(
                    ratio * advantage, clipped_ratio * advantage
                )
                sequence_objective = (
                    (token_objective * chunk.action_mask).sum(dim=1) / token_counts
                )
                policy_loss = -sequence_objective.sum() / batch_size

                reference_delta = (
                    reference_log_probabilities[start:end]
                    - current_log_probabilities
                )
                sampled_kl_tokens = (
                    reference_delta.exp() - reference_delta - 1.0
                )
                sampled_kl = (
                    (
                        (sampled_kl_tokens * chunk.action_mask).sum(dim=1)
                        / token_counts
                    ).sum()
                    / batch_size
                )

                valid_positions = chunk.valid_next_mask.any(dim=-1)
                if bool(valid_positions.any()):
                    float_logits = logits.float()
                    valid_logits = float_logits.masked_fill(
                        ~chunk.valid_next_mask, -torch.inf
                    )
                    log_valid_mass = valid_logits.logsumexp(
                        dim=-1
                    ) - float_logits.logsumexp(dim=-1)
                    denominator = max(valid_position_count, 1)
                    valid_loss = -log_valid_mass[valid_positions].sum() / denominator
                    valid_mass = log_valid_mass[valid_positions].exp().sum() / denominator
                else:
                    valid_loss = logits.sum() * 0.0
                    valid_mass = logits.sum() * 0.0

                loss = (
                    policy_loss
                    + config.kl_coefficient * sampled_kl
                    + config.valid_coefficient * valid_loss
                )
            loss.backward()
            epoch["policy_loss"] += float(policy_loss.detach().item())
            epoch["sampled_kl"] += float(sampled_kl.detach().item())
            epoch["valid_loss"] += float(valid_loss.detach().item())
            epoch["valid_mass"] += float(valid_mass.detach().item())

        gradient_norm = nn.utils.clip_grad_norm_(
            model.parameters(), config.gradient_clip
        )
        optimizer.step()
        for key, value in epoch.items():
            totals[key] += value
        totals["gradient_norm"] += float(gradient_norm.item())

    scale = 1.0 / config.update_epochs
    policy_loss = totals["policy_loss"] * scale
    sampled_kl = totals["sampled_kl"] * scale
    valid_loss = totals["valid_loss"] * scale
    return UpdateMetrics(
        policy_loss=policy_loss,
        sampled_kl=sampled_kl,
        valid_loss=valid_loss,
        valid_mass=totals["valid_mass"] * scale,
        total_loss=(
            policy_loss
            + config.kl_coefficient * sampled_kl
            + config.valid_coefficient * valid_loss
        ),
        zero_variance_group_fraction=zero_variance_fraction,
        gradient_norm=totals["gradient_norm"] * scale,
    )
