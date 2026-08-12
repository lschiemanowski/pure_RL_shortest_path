"""Curriculum and experiment orchestration independent of policy optimization."""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
import random

from .task import (
    GraphExample,
    GraphProblemConfig,
    Vocabulary,
    generate_example,
)


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
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
        past = _positive_finite_real(self.past_decay_scale, "past_decay_scale")
        future = _positive_finite_real(
            self.future_decay_scale, "future_decay_scale"
        )
        if past <= future:
            raise ValueError(
                "past_decay_scale must exceed future_decay_scale for an asymmetric tail"
            )
        threshold = _positive_finite_real(
            self.advancement_threshold, "advancement_threshold"
        )
        if threshold > 1.0:
            raise ValueError("advancement_threshold must not exceed one")
        patience = _positive_int(self.advancement_patience, "advancement_patience")

        greatest_exponent = (len(stages) - 1) / future
        if greatest_exponent > -math.log(math.nextafter(0.0, 1.0)):
            raise ValueError(
                "future_decay_scale is too small to retain positive floating-point "
                "probability for every stage"
            )
        object.__setattr__(self, "stages", stages)
        object.__setattr__(self, "past_decay_scale", past)
        object.__setattr__(self, "future_decay_scale", future)
        object.__setattr__(self, "advancement_threshold", threshold)
        object.__setattr__(self, "advancement_patience", patience)


@dataclass(frozen=True)
class CurriculumState:
    frontier: int = 0
    advancement_streak: int = 0

    def validate_for(self, curriculum: CurriculumConfig) -> None:
        if isinstance(self.frontier, bool) or not isinstance(self.frontier, int):
            raise TypeError("frontier must be an integer")
        if not 0 <= self.frontier < len(curriculum.stages):
            raise ValueError("frontier is outside the curriculum")
        if (
            isinstance(self.advancement_streak, bool)
            or not isinstance(self.advancement_streak, int)
        ):
            raise TypeError("advancement_streak must be an integer")
        if self.advancement_streak < 0:
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
        frontier = self.frontier
        if isinstance(frontier, bool) or not isinstance(frontier, int):
            raise TypeError("frontier must be an integer")
        count = _positive_int(self.example_count, "example_count")
        successes = self.shortest_path_successes
        if isinstance(successes, bool) or not isinstance(successes, int):
            raise TypeError("shortest_path_successes must be an integer")
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
