"""Curriculum and experiment orchestration independent of policy optimization."""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
import random
from typing import Literal, Sequence

import torch

from .model import Transformer, generate_tokens
from .task import (
    EOS,
    PAD,
    GraphExample,
    GraphProblemConfig,
    OutcomeFacts,
    Vocabulary,
    generate_example,
    verify_completion,
)


def _plain_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    return value


def _positive_int(value: object, name: str) -> int:
    value = _plain_int(value, name)
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _positive_finite_real(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


DEFAULT_CURRICULUM = (
    GraphProblemConfig(8, 7, 1, 1),
    GraphProblemConfig(8, 9, 2, 2),
    GraphProblemConfig(12, 14, 2, 3),
    GraphProblemConfig(16, 20, 3, 3),
    GraphProblemConfig(24, 32, 3, 4),
    GraphProblemConfig(32, 44, 4, 4),
    GraphProblemConfig(48, 64, 4, 5),
    GraphProblemConfig(64, 88, 5, 5),
    GraphProblemConfig(80, 112, 5, 6),
    GraphProblemConfig(96, 136, 6, 6),
    GraphProblemConfig(112, 160, 6, 7),
    GraphProblemConfig(128, 184, 7, 7),
    GraphProblemConfig(160, 220, 7, 8),
    GraphProblemConfig(192, 256, 8, 8),
)


@dataclass(frozen=True)
class CurriculumConfig:
    """Ordered stages, asymmetric mixture, and validation advancement gate."""

    stages: tuple[GraphProblemConfig, ...]
    past_decay_scale: float = 2.0
    future_decay_scale: float = 0.25
    advancement_threshold: float = 0.9
    advancement_patience: int = 3

    def __post_init__(self) -> None:
        stages = tuple(self.stages)
        if not stages:
            raise ValueError("curriculum must contain at least one stage")
        if not all(isinstance(stage, GraphProblemConfig) for stage in stages):
            raise TypeError("every curriculum stage must be a GraphProblemConfig")
        for name in ("past_decay_scale", "future_decay_scale"):
            value = _positive_finite_real(getattr(self, name), name)
            object.__setattr__(self, name, value)
        if self.past_decay_scale <= self.future_decay_scale:
            raise ValueError(
                "past_decay_scale must exceed future_decay_scale for an asymmetric tail"
            )
        threshold = _positive_finite_real(
            self.advancement_threshold, "advancement_threshold"
        )
        if threshold > 1.0:
            raise ValueError("advancement_threshold must not exceed one")
        patience = _positive_int(self.advancement_patience, "advancement_patience")

        greatest_exponent = (len(stages) - 1) / self.future_decay_scale
        if greatest_exponent > -math.log(math.nextafter(0.0, 1.0)):
            raise ValueError(
                "future_decay_scale is too small to retain positive floating-point "
                "probability for every stage"
            )
        object.__setattr__(self, "stages", stages)
        object.__setattr__(self, "advancement_threshold", threshold)
        object.__setattr__(self, "advancement_patience", patience)


@dataclass(frozen=True)
class CurriculumState:
    frontier: int = 0
    advancement_streak: int = 0

    def validate_for(self, curriculum: CurriculumConfig) -> None:
        frontier = _plain_int(self.frontier, "frontier")
        if not 0 <= frontier < len(curriculum.stages):
            raise ValueError("frontier is outside the curriculum")
        streak = _plain_int(self.advancement_streak, "advancement_streak")
        if streak < 0:
            raise ValueError("advancement_streak must be nonnegative")


def frontier_stage_probabilities(
    curriculum: CurriculumConfig, frontier: int
) -> tuple[float, ...]:
    """Return the normalized asymmetric distribution around frontier."""

    state = CurriculumState(frontier=frontier)
    state.validate_for(curriculum)
    weights = []
    for stage_index in range(len(curriculum.stages)):
        if stage_index < frontier:
            exponent = -(frontier - stage_index) / curriculum.past_decay_scale
        elif stage_index > frontier:
            exponent = -(stage_index - frontier) / curriculum.future_decay_scale
        else:
            exponent = 0.0
        weights.append(math.exp(exponent))
    if any(weight <= 0.0 for weight in weights):
        raise ValueError(
            "decay scales must give every curriculum stage positive weight"
        )
    total = math.fsum(weights)
    return tuple(weight / total for weight in weights)


@dataclass(frozen=True)
class TrainingProblemBatch:
    examples: tuple[GraphExample, ...]
    stage_indices: tuple[int, ...]
    stage_probabilities: tuple[float, ...]
    stage_counts: tuple[int, ...]
    frontier: int


def generate_training_problem_batch(
    curriculum: CurriculumConfig,
    state: CurriculumState,
    vocabulary: Vocabulary,
    problem_count: int,
    rng: random.Random,
) -> TrainingProblemBatch:
    """Independently sample a stage and construct each training problem."""

    state.validate_for(curriculum)
    problem_count = _positive_int(problem_count, "problem_count")
    probabilities = frontier_stage_probabilities(curriculum, state.frontier)
    population = tuple(range(len(curriculum.stages)))
    stage_indices: list[int] = []
    examples: list[GraphExample] = []
    counts = [0] * len(curriculum.stages)
    for _ in range(problem_count):
        stage_index = rng.choices(population, weights=probabilities, k=1)[0]
        stage_indices.append(stage_index)
        counts[stage_index] += 1
        examples.append(
            generate_example(curriculum.stages[stage_index], vocabulary, rng)
        )
    return TrainingProblemBatch(
        examples=tuple(examples),
        stage_indices=tuple(stage_indices),
        stage_probabilities=probabilities,
        stage_counts=tuple(counts),
        frontier=state.frontier,
    )


@dataclass(frozen=True)
class FrontierValidation:
    training_step: int
    frontier: int
    example_count: int
    shortest_path_successes: int

    def __post_init__(self) -> None:
        step = _positive_int(self.training_step, "training_step")
        frontier = _plain_int(self.frontier, "frontier")
        count = _positive_int(self.example_count, "example_count")
        successes = _plain_int(
            self.shortest_path_successes, "shortest_path_successes"
        )
        if not 0 <= successes <= count:
            raise ValueError("shortest_path_successes must lie within the denominator")
        object.__setattr__(self, "training_step", step)
        object.__setattr__(self, "frontier", frontier)
        object.__setattr__(self, "example_count", count)

    @property
    def shortest_path_success(self) -> float:
        return self.shortest_path_successes / self.example_count


@dataclass(frozen=True)
class FrontierTransition:
    training_step: int
    completed_frontier: int
    new_frontier: int
    validation_shortest_path_success: float


@dataclass(frozen=True)
class CurriculumValidationDecision:
    validation: FrontierValidation
    threshold: float
    qualified: bool
    resulting_state: CurriculumState
    transition: FrontierTransition | None


def apply_frontier_validation(
    curriculum: CurriculumConfig,
    state: CurriculumState,
    validation: FrontierValidation,
) -> CurriculumValidationDecision:
    """Update the monotone frontier from one independent validation result."""

    state.validate_for(curriculum)
    if validation.frontier != state.frontier:
        raise ValueError("validation frontier does not match current curriculum state")
    qualified = validation.shortest_path_success >= curriculum.advancement_threshold
    streak = state.advancement_streak + 1 if qualified else 0
    transition = None
    frontier = state.frontier
    if (
        streak >= curriculum.advancement_patience
        and frontier < len(curriculum.stages) - 1
    ):
        transition = FrontierTransition(
            training_step=validation.training_step,
            completed_frontier=frontier,
            new_frontier=frontier + 1,
            validation_shortest_path_success=validation.shortest_path_success,
        )
        frontier += 1
        streak = 0
    resulting_state = CurriculumState(frontier, streak)
    return CurriculumValidationDecision(
        validation=validation,
        threshold=curriculum.advancement_threshold,
        qualified=qualified,
        resulting_state=resulting_state,
        transition=transition,
    )

@dataclass(frozen=True)
class EvaluationProblemSet:
    """An immutable fresh-seed or fixed-identity evaluation set."""

    problem_config: GraphProblemConfig
    examples: tuple[GraphExample, ...]
    generation_seed: int | None = None
    fixed_set_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.problem_config, GraphProblemConfig):
            raise TypeError("problem_config must be a GraphProblemConfig")
        examples = tuple(self.examples)
        if not examples:
            raise ValueError("evaluation set must contain at least one example")
        has_seed = self.generation_seed is not None
        has_fixed_id = self.fixed_set_id is not None
        if has_seed == has_fixed_id:
            raise ValueError(
                "evaluation set requires exactly one generation seed or "
                "fixed-set identity"
            )
        if has_seed:
            _plain_int(self.generation_seed, "generation_seed")
        if has_fixed_id and (
            not isinstance(self.fixed_set_id, str) or not self.fixed_set_id.strip()
        ):
            raise ValueError("fixed_set_id must be a nonempty string")
        for example in examples:
            if not isinstance(example, GraphExample):
                raise TypeError("every evaluation item must be a GraphExample")
            if (
                example.problem.vertex_count != self.problem_config.vertices
                or len(example.problem.edges) != self.problem_config.edges
                or not self.problem_config.min_distance
                <= example.shortest_distance
                <= self.problem_config.max_distance
            ):
                raise ValueError(
                    "evaluation example does not match its declared problem "
                    "configuration"
                )
        object.__setattr__(self, "examples", examples)

    @property
    def identity(self) -> str:
        if self.fixed_set_id is not None:
            return f"fixed:{self.fixed_set_id}"
        return f"seed:{self.generation_seed}"


def generate_evaluation_problem_set(
    problem_config: GraphProblemConfig,
    vocabulary: Vocabulary,
    example_count: int,
    generation_seed: int,
) -> EvaluationProblemSet:
    """Generate a reproducible fresh evaluation set from its independent seed."""

    example_count = _positive_int(example_count, "example_count")
    generation_seed = _plain_int(generation_seed, "generation_seed")
    rng = random.Random(generation_seed)
    examples = tuple(
        generate_example(problem_config, vocabulary, rng)
        for _ in range(example_count)
    )
    return EvaluationProblemSet(
        problem_config=problem_config,
        examples=examples,
        generation_seed=generation_seed,
    )


def fixed_evaluation_problem_set(
    fixed_set_id: str,
    problem_config: GraphProblemConfig,
    examples: Sequence[GraphExample],
) -> EvaluationProblemSet:
    """Attach an explicit stable identity to a supplied immutable evaluation set."""

    return EvaluationProblemSet(
        problem_config=problem_config,
        examples=tuple(examples),
        fixed_set_id=fixed_set_id,
    )


@dataclass(frozen=True)
class EvaluationSamplingConfig:
    """Declared greedy or stochastic completion generation for evaluation."""

    max_new_tokens: int
    batch_size: int | None = None
    mode: Literal["greedy", "stochastic"] = "greedy"
    temperature: float | None = None
    top_p: float | None = None

    def __post_init__(self) -> None:
        max_new_tokens = _positive_int(self.max_new_tokens, "max_new_tokens")
        batch_size = self.batch_size
        if batch_size is not None:
            batch_size = _positive_int(batch_size, "batch_size")
        if self.mode == "greedy":
            if self.temperature is not None or self.top_p is not None:
                raise ValueError(
                    "greedy evaluation does not accept temperature or top_p"
                )
        elif self.mode == "stochastic":
            if self.temperature is None or self.top_p is None:
                raise ValueError(
                    "stochastic evaluation requires temperature and top_p"
                )
            temperature = _positive_finite_real(self.temperature, "temperature")
            top_p = _positive_finite_real(self.top_p, "top_p")
            if top_p > 1.0:
                raise ValueError("top_p must not exceed one")
            object.__setattr__(self, "temperature", temperature)
            object.__setattr__(self, "top_p", top_p)
        else:
            raise ValueError("mode must be 'greedy' or 'stochastic'")
        object.__setattr__(self, "max_new_tokens", max_new_tokens)
        object.__setattr__(self, "batch_size", batch_size)


def select_evaluation_tokens(
    logits: torch.Tensor,
    sampling: EvaluationSamplingConfig,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Select one evaluation action per row under the declared sampling protocol."""

    if logits.ndim != 2 or logits.shape[0] == 0 or logits.shape[1] == 0:
        raise ValueError("logits must have shape [positive_batch, positive_vocabulary]")
    if sampling.mode == "greedy":
        return logits.argmax(dim=-1)

    assert sampling.temperature is not None and sampling.top_p is not None
    scaled = logits.float() / sampling.temperature
    if sampling.top_p < 1.0:
        sorted_logits, sorted_indices = scaled.sort(dim=-1, descending=True)
        sorted_probabilities = sorted_logits.softmax(dim=-1)
        probability_before = sorted_probabilities.cumsum(dim=-1) - sorted_probabilities
        keep = probability_before < sampling.top_p
        sorted_logits = sorted_logits.masked_fill(~keep, -torch.inf)
        scaled = torch.full_like(scaled, -torch.inf).scatter(
            1, sorted_indices, sorted_logits
        )
    probabilities = scaled.softmax(dim=-1)
    if generator is not None and probabilities.device.type != "cpu":
        return torch.multinomial(
            probabilities.cpu(), 1, generator=generator
        ).squeeze(1).to(logits.device)
    return torch.multinomial(probabilities, 1, generator=generator).squeeze(1)


@dataclass(frozen=True)
class GeneratedEvaluationCompletion:
    example: GraphExample
    completion: tuple[int, ...]
    terminated_by_eos: bool


@torch.no_grad()
def generate_evaluation_completions(
    model: Transformer,
    problem_set: EvaluationProblemSet,
    sampling: EvaluationSamplingConfig,
    *,
    device: torch.device,
    sampling_seed: int | None = None,
    autocast_dtype: torch.dtype | None = None,
) -> tuple[GeneratedEvaluationCompletion, ...]:
    """Generate one unconstrained completion for each independent evaluation item."""

    if sampling.mode == "greedy":
        if sampling_seed is not None:
            raise ValueError("greedy evaluation does not use a sampling seed")
        generator = None
    else:
        if sampling_seed is None:
            raise ValueError("stochastic evaluation requires a sampling seed")
        sampling_seed = _plain_int(sampling_seed, "sampling_seed")
        generator = torch.Generator().manual_seed(sampling_seed)

    generated = generate_tokens(
        model,
        [example.prompt for example in problem_set.examples],
        max_new_tokens=sampling.max_new_tokens,
        batch_size=sampling.batch_size,
        pad_token=PAD,
        eos_token=EOS,
        device=device,
        select_next=lambda logits: select_evaluation_tokens(
            logits, sampling, generator
        ),
        autocast_dtype=autocast_dtype,
    )
    return tuple(
        GeneratedEvaluationCompletion(
            example=example,
            completion=completion,
            terminated_by_eos=bool(completion and completion[-1] == EOS),
        )
        for example, completion in zip(problem_set.examples, generated)
    )


@dataclass(frozen=True)
class EvaluationProtocol:
    policy_checkpoint: str
    sampling: EvaluationSamplingConfig
    sampling_seed: int | None = None
    minimum_reason_tokens: int = 1
    intervention_id: str | None = None
    comparison_condition: str | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.policy_checkpoint, str)
            or not self.policy_checkpoint.strip()
        ):
            raise ValueError("policy_checkpoint must be a nonempty string")
        if not isinstance(self.sampling, EvaluationSamplingConfig):
            raise TypeError("sampling must be an EvaluationSamplingConfig")
        minimum_reason_tokens = _positive_int(
            self.minimum_reason_tokens, "minimum_reason_tokens"
        )
        object.__setattr__(
            self, "minimum_reason_tokens", minimum_reason_tokens
        )
        if self.sampling.mode == "greedy":
            if self.sampling_seed is not None:
                raise ValueError("greedy evaluation does not use a sampling seed")
        elif self.sampling_seed is None:
            raise ValueError("stochastic evaluation requires a sampling seed")
        else:
            _plain_int(self.sampling_seed, "sampling_seed")
        has_intervention = self.intervention_id is not None
        has_comparison = self.comparison_condition is not None
        if has_intervention != has_comparison:
            raise ValueError(
                "intervention evaluation requires both identity and comparison "
                "condition"
            )
        if has_intervention and (
            not isinstance(self.intervention_id, str)
            or not self.intervention_id.strip()
            or not isinstance(self.comparison_condition, str)
            or not self.comparison_condition.strip()
        ):
            raise ValueError("intervention identity and comparison must be nonempty")

    @property
    def kind(self) -> str:
        return "intervention" if self.intervention_id is not None else "ordinary"


@dataclass(frozen=True)
class EvaluationCaseResult:
    completion: GeneratedEvaluationCompletion
    outcome: OutcomeFacts


@dataclass(frozen=True)
class EvaluationMetrics:
    example_count: int
    format_successes: int
    valid_path_successes: int
    shortest_path_successes: int
    format_success_rate: float
    valid_path_success_rate: float
    shortest_path_success_rate: float
    mean_valid_path_length: float | None
    mean_valid_excess_length: float | None


@dataclass(frozen=True)
class EvaluationResult:
    protocol: EvaluationProtocol
    problem_set_identity: str
    problem_config: GraphProblemConfig
    cases: tuple[EvaluationCaseResult, ...]
    metrics: EvaluationMetrics


def aggregate_evaluation_outcomes(
    protocol: EvaluationProtocol,
    problem_set: EvaluationProblemSet,
    completions: Sequence[GeneratedEvaluationCompletion],
    vocabulary: Vocabulary,
) -> EvaluationResult:
    """Verify generated completions and retain exact primary counts and rates."""

    completions = tuple(completions)
    if len(completions) != len(problem_set.examples):
        raise ValueError("evaluation requires exactly one completion per problem")
    if any(
        completion.example != expected
        for completion, expected in zip(completions, problem_set.examples)
    ):
        raise ValueError("evaluation completions do not match the ordered problem set")

    cases = tuple(
        EvaluationCaseResult(
            completion=completion,
            outcome=verify_completion(
                completion.example,
                completion.completion,
                vocabulary,
                protocol.minimum_reason_tokens,
            ),
        )
        for completion in completions
    )
    count = len(cases)
    format_successes = sum(case.outcome.format_ok for case in cases)
    valid_path_successes = sum(case.outcome.valid_path for case in cases)
    shortest_path_successes = sum(case.outcome.shortest for case in cases)
    valid_lengths = [
        case.outcome.answer_length
        for case in cases
        if case.outcome.valid_path and case.outcome.answer_length is not None
    ]
    valid_excesses = [
        case.outcome.excess_length
        for case in cases
        if case.outcome.valid_path and case.outcome.excess_length is not None
    ]
    metrics = EvaluationMetrics(
        example_count=count,
        format_successes=format_successes,
        valid_path_successes=valid_path_successes,
        shortest_path_successes=shortest_path_successes,
        format_success_rate=format_successes / count,
        valid_path_success_rate=valid_path_successes / count,
        shortest_path_success_rate=shortest_path_successes / count,
        mean_valid_path_length=(
            math.fsum(valid_lengths) / len(valid_lengths) if valid_lengths else None
        ),
        mean_valid_excess_length=(
            math.fsum(valid_excesses) / len(valid_excesses)
            if valid_excesses
            else None
        ),
    )
    return EvaluationResult(
        protocol=protocol,
        problem_set_identity=problem_set.identity,
        problem_config=problem_set.problem_config,
        cases=cases,
        metrics=metrics,
    )


def run_evaluation(
    model: Transformer,
    protocol: EvaluationProtocol,
    problem_set: EvaluationProblemSet,
    vocabulary: Vocabulary,
    *,
    device: torch.device,
    autocast_dtype: torch.dtype | None = None,
) -> EvaluationResult:
    """Run ordinary evaluation without mutating training or curriculum state."""

    if protocol.kind != "ordinary":
        raise ValueError(
            "intervention completions must be produced explicitly before aggregation"
        )
    completions = generate_evaluation_completions(
        model,
        problem_set,
        protocol.sampling,
        device=device,
        sampling_seed=protocol.sampling_seed,
        autocast_dtype=autocast_dtype,
    )
    return aggregate_evaluation_outcomes(
        protocol, problem_set, completions, vocabulary
    )
