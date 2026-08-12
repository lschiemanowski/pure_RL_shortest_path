"""Resolved experiment declarations and reproducible run initialization."""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
import math
from numbers import Real
import os
from pathlib import Path
import platform
import random
import re
import shutil
import signal
import subprocess
import sys
import threading
import tomllib
from typing import Any, Literal, Mapping, Sequence
import uuid

import torch

from .experiment import (
    CurriculumConfig,
    CurriculumState,
    EvaluationProtocol,
    EvaluationResult,
    EvaluationSamplingConfig,
    FrontierValidation,
    TrainingProblemBatch,
    apply_frontier_validation,
    generate_evaluation_problem_set,
    generate_training_problem_batch,
    run_evaluation,
)
from .model import Transformer, TransformerConfig
from .rl import (
    GRPOConfig,
    Rollout,
    RolloutSamplingConfig,
    UpdateMetrics,
    collect_grouped_completions,
    evaluate_samples,
    grpo_update,
    pack_rollouts,
)
from .task import GraphProblemConfig, Vocabulary


CONFIG_SCHEMA_VERSION = 1
CHECKPOINT_SCHEMA_VERSION = 2
EVIDENCE_SCHEMA_VERSION = 2
SEED_STREAM_NAMES = (
    "model_initialization",
    "training_problems",
    "rollout_sampling",
    "curriculum_validation",
    "evaluation_problems",
    "evaluation_sampling",
)


class ConfigurationError(ValueError):
    """Raised when a TOML experiment declaration cannot be resolved."""


def _plain_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigurationError(f"{name} must be an integer")
    return value


def _positive_int(value: object, name: str) -> int:
    result = _plain_int(value, name)
    if result <= 0:
        raise ConfigurationError(f"{name} must be positive")
    return result


def _nonnegative_real(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ConfigurationError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ConfigurationError(f"{name} must be finite and nonnegative")
    return result


def _positive_real(value: object, name: str) -> float:
    result = _nonnegative_real(value, name)
    if result == 0.0:
        raise ConfigurationError(f"{name} must be positive")
    return result


def _nonempty_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"{name} must be a nonempty string")
    return value.strip()


def _table(
    mapping: Mapping[str, Any], name: str, *, required: bool = False
) -> dict[str, Any]:
    if name not in mapping:
        if required:
            raise ConfigurationError(f"missing required table [{name}]")
        return {}
    value = mapping[name]
    if not isinstance(value, dict):
        raise ConfigurationError(f"{name} must be a TOML table")
    return dict(value)


def _reject_unknown(
    mapping: Mapping[str, Any], allowed: set[str], location: str
) -> None:
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        rendered = ", ".join(f"{location}.{key}" for key in unknown)
        raise ConfigurationError(f"unknown configuration key(s): {rendered}")


@dataclass(frozen=True)
class RewardConfig:
    coverage_coefficient: float = 0.0
    sterile_repetition_coefficient: float = 0.0
    sterile_repetition_mode: Literal["all", "off_answer"] = "off_answer"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "coverage_coefficient",
            _nonnegative_real(
                self.coverage_coefficient, "reward.coverage_coefficient"
            ),
        )
        object.__setattr__(
            self,
            "sterile_repetition_coefficient",
            _nonnegative_real(
                self.sterile_repetition_coefficient,
                "reward.sterile_repetition_coefficient",
            ),
        )
        if self.sterile_repetition_mode not in ("all", "off_answer"):
            raise ConfigurationError(
                "reward.sterile_repetition_mode must be all or off_answer"
            )


@dataclass(frozen=True)
class OptimizerConfig:
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.95

    def __post_init__(self) -> None:
        learning_rate = _positive_real(
            self.learning_rate, "optimization.learning_rate"
        )
        weight_decay = _nonnegative_real(
            self.weight_decay, "optimization.weight_decay"
        )
        beta1 = _nonnegative_real(self.beta1, "optimization.beta1")
        beta2 = _nonnegative_real(self.beta2, "optimization.beta2")
        if beta1 >= 1.0 or beta2 >= 1.0:
            raise ConfigurationError("optimization betas must be less than one")
        object.__setattr__(self, "learning_rate", learning_rate)
        object.__setattr__(self, "weight_decay", weight_decay)
        object.__setattr__(self, "beta1", beta1)
        object.__setattr__(self, "beta2", beta2)


@dataclass(frozen=True)
class TrainingScheduleConfig:
    max_steps: int = 50_000
    problems_per_step: int = 4
    reference_update_every: int = 100

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "max_steps", _positive_int(self.max_steps, "training.max_steps")
        )
        object.__setattr__(
            self,
            "problems_per_step",
            _positive_int(self.problems_per_step, "training.problems_per_step"),
        )
        object.__setattr__(
            self,
            "reference_update_every",
            _positive_int(
                self.reference_update_every, "training.reference_update_every"
            ),
        )


@dataclass(frozen=True)
class EvaluationScheduleConfig:
    every_steps: int
    example_count: int
    sampling: EvaluationSamplingConfig

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "every_steps",
            _positive_int(self.every_steps, "evaluation.every_steps"),
        )
        object.__setattr__(
            self,
            "example_count",
            _positive_int(self.example_count, "evaluation.example_count"),
        )
        if not isinstance(self.sampling, EvaluationSamplingConfig):
            raise ConfigurationError(
                "evaluation sampling must be an EvaluationSamplingConfig"
            )


@dataclass(frozen=True)
class ArtifactConfig:
    output_root: Path
    log_every: int = 10
    checkpoint_every: int = 500
    representative_every: int = 100
    representative_count: int = 4
    representative_selection: Literal["first_completion_per_problem"] = (
        "first_completion_per_problem"
    )

    def __post_init__(self) -> None:
        output_root = Path(self.output_root)
        object.__setattr__(self, "output_root", output_root)
        object.__setattr__(
            self, "log_every", _positive_int(self.log_every, "artifacts.log_every")
        )
        object.__setattr__(
            self,
            "checkpoint_every",
            _positive_int(
                self.checkpoint_every, "artifacts.checkpoint_every"
            ),
        )
        object.__setattr__(
            self,
            "representative_every",
            _positive_int(
                self.representative_every, "artifacts.representative_every"
            ),
        )
        object.__setattr__(
            self,
            "representative_count",
            _positive_int(
                self.representative_count, "artifacts.representative_count"
            ),
        )
        if self.representative_selection != "first_completion_per_problem":
            raise ConfigurationError(
                "artifacts.representative_selection must be "
                "first_completion_per_problem"
            )


@dataclass(frozen=True)
class RuntimeConfig:
    device: str = "auto"
    dtype: Literal["bfloat16", "float16", "float32"] = "bfloat16"
    allow_dirty_source: bool = False

    def __post_init__(self) -> None:
        device = _nonempty_string(self.device, "runtime.device")
        if device != "auto":
            try:
                parsed_device = torch.device(device)
            except RuntimeError as error:
                raise ConfigurationError(
                    f"runtime.device is invalid: {device}"
                ) from error
            if parsed_device.type not in ("cpu", "cuda", "mps"):
                raise ConfigurationError(
                    "runtime.device must select auto, cpu, cuda, or mps"
                )
        if self.dtype not in ("bfloat16", "float16", "float32"):
            raise ConfigurationError(
                "runtime.dtype must be bfloat16, float16, or float32"
            )
        if not isinstance(self.allow_dirty_source, bool):
            raise ConfigurationError("runtime.allow_dirty_source must be boolean")
        object.__setattr__(self, "device", device)


@dataclass(frozen=True)
class RunConfiguration:
    """All scientific and operational settings after TOML resolution."""

    schema_version: int
    name: str
    master_seed: int
    vocabulary: Vocabulary
    model: TransformerConfig
    rollout: RolloutSamplingConfig
    reward: RewardConfig
    optimizer: OptimizerConfig
    grpo: GRPOConfig
    curriculum: CurriculumConfig
    training: TrainingScheduleConfig
    evaluation: EvaluationScheduleConfig
    artifacts: ArtifactConfig
    runtime: RuntimeConfig

    def __post_init__(self) -> None:
        version = _plain_int(self.schema_version, "schema_version")
        if version != CONFIG_SCHEMA_VERSION:
            raise ConfigurationError(
                f"unsupported schema_version {version}; expected "
                f"{CONFIG_SCHEMA_VERSION}"
            )
        name = _nonempty_string(self.name, "name")
        seed = _plain_int(self.master_seed, "seed")
        if not 0 <= seed < 2**63:
            raise ConfigurationError("seed must lie in [0, 2^63)")
        largest_graph = max(stage.vertices for stage in self.curriculum.stages)
        if self.vocabulary.node_label_count < largest_graph:
            raise ConfigurationError(
                "task.node_label_count must cover every curriculum stage"
            )
        if self.model.vocab_size != self.vocabulary.size:
            raise ConfigurationError(
                "model vocabulary size must be derived from the task vocabulary"
            )
        if self.grpo.group_size != self.rollout.group_size:
            raise ConfigurationError(
                "GRPO group size must equal the rollout comparison-group size"
            )
        maximum_prompt = max(2 * stage.edges + 7 for stage in self.curriculum.stages)
        required_context = maximum_prompt + max(
            self.rollout.max_new_tokens,
            self.evaluation.sampling.max_new_tokens,
        )
        if required_context > self.model.max_context_length:
            raise ConfigurationError(
                "model.max_context_length is smaller than the largest configured "
                "prompt plus completion budget"
            )
        object.__setattr__(self, "schema_version", version)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "master_seed", seed)


@dataclass(frozen=True)
class LoadedRunConfiguration:
    source_path: Path
    source_bytes: bytes
    resolved: RunConfiguration
    source_sha256: str
    resolved_sha256: str


def _plain_data(value: object) -> object:
    if is_dataclass(value) and not isinstance(value, type):
        return _plain_data(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _plain_data(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain_data(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return [_plain_data(item) for item in sorted(value)]
    return value


def resolved_configuration_record(config: RunConfiguration) -> dict[str, object]:
    record = _plain_data(config)
    assert isinstance(record, dict)
    return record


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        + "\n"
    ).encode("utf-8")


def _graph_stage(value: object, index: int) -> GraphProblemConfig:
    if not isinstance(value, dict):
        raise ConfigurationError(f"curriculum.stages[{index}] must be a table")
    _reject_unknown(
        value,
        {
            "vertices",
            "edges",
            "min_distance",
            "max_distance",
            "generation_attempts",
        },
        f"curriculum.stages[{index}]",
    )
    required = ("vertices", "edges", "min_distance", "max_distance")
    missing = [key for key in required if key not in value]
    if missing:
        raise ConfigurationError(
            f"curriculum.stages[{index}] is missing: {', '.join(missing)}"
        )
    return GraphProblemConfig(
        vertices=value["vertices"],
        edges=value["edges"],
        min_distance=value["min_distance"],
        max_distance=value["max_distance"],
        generation_attempts=value.get("generation_attempts", 100),
    )


def resolve_run_configuration(
    declaration: Mapping[str, Any], *, declaration_directory: Path
) -> RunConfiguration:
    """Resolve a parsed TOML declaration into the project's typed contracts."""

    _reject_unknown(
        declaration,
        {
            "schema_version",
            "name",
            "seed",
            "task",
            "model",
            "rollout",
            "reward",
            "optimization",
            "curriculum",
            "training",
            "evaluation",
            "artifacts",
            "runtime",
        },
        "root",
    )
    for key in ("schema_version", "name", "seed"):
        if key not in declaration:
            raise ConfigurationError(f"missing required key: {key}")

    task = _table(declaration, "task", required=True)
    _reject_unknown(task, {"node_label_count"}, "task")
    if "node_label_count" not in task:
        raise ConfigurationError("missing required key: task.node_label_count")
    vocabulary = Vocabulary(task["node_label_count"])

    model_data = _table(declaration, "model")
    _reject_unknown(
        model_data,
        {
            "max_context_length",
            "d_model",
            "layers",
            "heads",
            "mlp_dim",
            "rope_base",
            "norm_eps",
        },
        "model",
    )
    model = TransformerConfig(
        vocab_size=vocabulary.size,
        max_context_length=model_data.get("max_context_length", 768),
        d_model=model_data.get("d_model", 256),
        n_layers=model_data.get("layers", 6),
        n_heads=model_data.get("heads", 8),
        mlp_dim=model_data.get("mlp_dim", 768),
        rope_base=model_data.get("rope_base", 10_000.0),
        norm_eps=model_data.get("norm_eps", 1e-5),
    )

    rollout_data = _table(declaration, "rollout")
    _reject_unknown(
        rollout_data,
        {
            "group_size",
            "max_new_tokens",
            "generation_batch_size",
            "temperature",
            "top_p",
        },
        "rollout",
    )
    rollout = RolloutSamplingConfig(
        group_size=rollout_data.get("group_size", 8),
        max_new_tokens=rollout_data.get("max_new_tokens", 96),
        generation_batch_size=rollout_data.get("generation_batch_size"),
        temperature=rollout_data.get("temperature", 1.0),
        top_p=rollout_data.get("top_p", 1.0),
    )

    reward_data = _table(declaration, "reward")
    _reject_unknown(
        reward_data,
        {
            "coverage_coefficient",
            "sterile_repetition_coefficient",
            "sterile_repetition_mode",
        },
        "reward",
    )
    reward = RewardConfig(
        coverage_coefficient=reward_data.get("coverage_coefficient", 0.0),
        sterile_repetition_coefficient=reward_data.get(
            "sterile_repetition_coefficient", 0.0
        ),
        sterile_repetition_mode=reward_data.get(
            "sterile_repetition_mode", "off_answer"
        ),
    )

    optimization = _table(declaration, "optimization")
    _reject_unknown(
        optimization,
        {
            "learning_rate",
            "weight_decay",
            "beta1",
            "beta2",
            "update_epochs",
            "microbatch_size",
            "clip_epsilon",
            "kl_coefficient",
            "valid_coefficient",
            "gradient_clip",
            "advantage_epsilon",
        },
        "optimization",
    )
    optimizer = OptimizerConfig(
        learning_rate=optimization.get("learning_rate", 3e-4),
        weight_decay=optimization.get("weight_decay", 0.01),
        beta1=optimization.get("beta1", 0.9),
        beta2=optimization.get("beta2", 0.95),
    )
    grpo = GRPOConfig(
        group_size=rollout.group_size,
        update_epochs=optimization.get("update_epochs", 2),
        microbatch_size=optimization.get("microbatch_size", 8),
        clip_epsilon=optimization.get("clip_epsilon", 0.2),
        kl_coefficient=optimization.get("kl_coefficient", 0.001),
        valid_coefficient=optimization.get("valid_coefficient", 0.0),
        gradient_clip=optimization.get("gradient_clip", 1.0),
        advantage_epsilon=optimization.get("advantage_epsilon", 1e-6),
    )

    curriculum_data = _table(declaration, "curriculum", required=True)
    _reject_unknown(
        curriculum_data,
        {
            "past_decay_scale",
            "future_decay_scale",
            "advancement_threshold",
            "advancement_patience",
            "stages",
        },
        "curriculum",
    )
    stages_data = curriculum_data.get("stages")
    if not isinstance(stages_data, list) or not stages_data:
        raise ConfigurationError(
            "curriculum.stages must be a nonempty array of tables"
        )
    stages = tuple(
        _graph_stage(stage, index) for index, stage in enumerate(stages_data)
    )
    curriculum = CurriculumConfig(
        stages=stages,
        past_decay_scale=curriculum_data.get("past_decay_scale", 2.0),
        future_decay_scale=curriculum_data.get("future_decay_scale", 0.25),
        advancement_threshold=curriculum_data.get(
            "advancement_threshold", 0.9
        ),
        advancement_patience=curriculum_data.get("advancement_patience", 3),
    )

    training_data = _table(declaration, "training")
    _reject_unknown(
        training_data,
        {"max_steps", "problems_per_step", "reference_update_every"},
        "training",
    )
    training = TrainingScheduleConfig(
        max_steps=training_data.get("max_steps", 50_000),
        problems_per_step=training_data.get("problems_per_step", 4),
        reference_update_every=training_data.get("reference_update_every", 100),
    )

    evaluation_data = _table(declaration, "evaluation")
    _reject_unknown(
        evaluation_data,
        {
            "every_steps",
            "example_count",
            "max_new_tokens",
            "batch_size",
            "mode",
            "temperature",
            "top_p",
        },
        "evaluation",
    )
    evaluation_sampling = EvaluationSamplingConfig(
        max_new_tokens=evaluation_data.get(
            "max_new_tokens", rollout.max_new_tokens
        ),
        batch_size=evaluation_data.get("batch_size"),
        mode=evaluation_data.get("mode", "greedy"),
        temperature=evaluation_data.get("temperature"),
        top_p=evaluation_data.get("top_p"),
    )
    evaluation = EvaluationScheduleConfig(
        every_steps=evaluation_data.get("every_steps", 100),
        example_count=evaluation_data.get("example_count", 256),
        sampling=evaluation_sampling,
    )

    artifacts_data = _table(declaration, "artifacts", required=True)
    _reject_unknown(
        artifacts_data,
        {
            "output_root",
            "log_every",
            "checkpoint_every",
            "representative_every",
            "representative_count",
            "representative_selection",
        },
        "artifacts",
    )
    if "output_root" not in artifacts_data:
        raise ConfigurationError("missing required key: artifacts.output_root")
    output_text = _nonempty_string(
        artifacts_data["output_root"], "artifacts.output_root"
    )
    output_root = Path(output_text).expanduser()
    if not output_root.is_absolute():
        output_root = declaration_directory / output_root
    output_root = output_root.resolve()
    artifacts = ArtifactConfig(
        output_root=output_root,
        log_every=artifacts_data.get("log_every", 10),
        checkpoint_every=artifacts_data.get("checkpoint_every", 500),
        representative_every=artifacts_data.get("representative_every", 100),
        representative_count=artifacts_data.get("representative_count", 4),
        representative_selection=artifacts_data.get(
            "representative_selection", "first_completion_per_problem"
        ),
    )

    runtime_data = _table(declaration, "runtime")
    _reject_unknown(
        runtime_data,
        {"device", "dtype", "allow_dirty_source"},
        "runtime",
    )
    runtime = RuntimeConfig(
        device=runtime_data.get("device", "auto"),
        dtype=runtime_data.get("dtype", "bfloat16"),
        allow_dirty_source=runtime_data.get("allow_dirty_source", False),
    )

    return RunConfiguration(
        schema_version=declaration["schema_version"],
        name=declaration["name"],
        master_seed=declaration["seed"],
        vocabulary=vocabulary,
        model=model,
        rollout=rollout,
        reward=reward,
        optimizer=optimizer,
        grpo=grpo,
        curriculum=curriculum,
        training=training,
        evaluation=evaluation,
        artifacts=artifacts,
        runtime=runtime,
    )


def load_run_configuration(path: str | Path) -> LoadedRunConfiguration:
    """Read one TOML declaration and retain both source and resolved identities."""

    source_path = Path(path).expanduser().resolve()
    source_bytes = source_path.read_bytes()
    try:
        declaration = tomllib.loads(source_bytes.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise ConfigurationError(f"invalid TOML configuration: {error}") from error
    resolved = resolve_run_configuration(
        declaration, declaration_directory=source_path.parent
    )
    resolved_bytes = _canonical_json_bytes(resolved_configuration_record(resolved))
    return LoadedRunConfiguration(
        source_path=source_path,
        source_bytes=source_bytes,
        resolved=resolved,
        source_sha256=hashlib.sha256(source_bytes).hexdigest(),
        resolved_sha256=hashlib.sha256(resolved_bytes).hexdigest(),
    )


def derive_named_seeds(master_seed: int) -> dict[str, int]:
    """Derive stable, explicitly named random streams from one master seed."""

    seed = _plain_int(master_seed, "seed")
    if not 0 <= seed < 2**63:
        raise ConfigurationError("seed must lie in [0, 2^63)")
    result = {}
    for name in SEED_STREAM_NAMES:
        material = f"pure-rl-shortest-path:v1:{seed}:{name}".encode("ascii")
        derived = int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
        result[name] = derived % (2**63)
    if len(set(result.values())) != len(result):
        raise RuntimeError("named seed derivation unexpectedly collided")
    return result


@dataclass(frozen=True)
class SourceProvenance:
    repository_root: Path
    revision: str
    dirty: bool
    status: tuple[str, ...]
    patch: bytes
    patch_sha256: str | None


def _git(
    repository: Path,
    arguments: Sequence[str],
    *,
    expected_returncodes: tuple[int, ...] = (0,),
) -> subprocess.CompletedProcess[bytes]:
    result = subprocess.run(
        ("git", *arguments),
        cwd=repository,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode not in expected_returncodes:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"git {' '.join(arguments)} failed: {detail}")
    return result


def inspect_source_provenance(repository: str | Path) -> SourceProvenance:
    """Identify the Git revision and preserve a patch for every dirty source file."""

    requested = Path(repository).expanduser().resolve()
    root_text = _git(requested, ("rev-parse", "--show-toplevel")).stdout.decode()
    root = Path(root_text.strip()).resolve()
    revision = (
        _git(root, ("rev-parse", "HEAD")).stdout.decode("ascii").strip()
    )
    status_text = _git(
        root, ("status", "--porcelain=v1", "--untracked-files=all")
    ).stdout.decode("utf-8")
    status = tuple(line for line in status_text.splitlines() if line)
    patch_parts = [_git(root, ("diff", "--binary", "HEAD", "--")).stdout]
    untracked_output = _git(
        root, ("ls-files", "--others", "--exclude-standard", "-z")
    ).stdout
    for raw_path in untracked_output.split(b"\0"):
        if not raw_path:
            continue
        relative = raw_path.decode("utf-8")
        patch = _git(
            root,
            ("diff", "--no-index", "--binary", "--", "/dev/null", relative),
            expected_returncodes=(0, 1),
        ).stdout
        patch_parts.append(patch)
    patch_bytes = b"\n".join(part.rstrip(b"\n") for part in patch_parts if part)
    if patch_bytes:
        patch_bytes += b"\n"
    return SourceProvenance(
        repository_root=root,
        revision=revision,
        dirty=bool(status),
        status=status,
        patch=patch_bytes,
        patch_sha256=(
            hashlib.sha256(patch_bytes).hexdigest() if patch_bytes else None
        ),
    )


@dataclass(frozen=True)
class RunProvenance:
    run_id: str
    started_at_utc: str
    command: tuple[str, ...]
    configuration_source: str
    configuration_source_sha256: str
    resolved_configuration_sha256: str
    seeds: dict[str, int]
    source_revision: str
    source_dirty: bool
    source_status: tuple[str, ...]
    source_patch_sha256: str | None
    python_version: str
    torch_version: str
    cuda_version: str | None
    requested_device: str
    device: str
    dtype: str
    parent_run_id: str | None = None
    parent_checkpoint: str | None = None
    configuration_differences: tuple[str, ...] = ()


@dataclass(frozen=True)
class InitializedRun:
    directory: Path
    configuration: LoadedRunConfiguration
    provenance: RunProvenance


def _run_slug(name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", name.strip()).strip("-.")
    return slug or "experiment"


def resolve_runtime_device(specification: str) -> str:
    """Resolve and validate the concrete device recorded for a run."""

    if specification == "auto":
        if torch.cuda.is_available():
            return "cuda:0"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    device = torch.device(specification)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        index = 0 if device.index is None else device.index
        if not 0 <= index < torch.cuda.device_count():
            raise RuntimeError(f"CUDA device index is unavailable: {index}")
        return f"cuda:{index}"
    if device.type == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is unavailable")
        return "mps"
    if device.type == "cpu":
        return "cpu"
    raise RuntimeError(f"unsupported runtime device: {device.type}")


def initialize_run(
    configuration: LoadedRunConfiguration,
    *,
    source_repository: str | Path,
    command: Sequence[str] | None = None,
    started_at: datetime | None = None,
    run_id: str | None = None,
    parent_run_id: str | None = None,
    parent_checkpoint: str | None = None,
    configuration_differences: Sequence[str] = (),
) -> InitializedRun:
    """Create a fresh, self-identifying run directory before model construction."""

    resolved_bytes = _canonical_json_bytes(
        resolved_configuration_record(configuration.resolved)
    )
    resolved_sha256 = hashlib.sha256(resolved_bytes).hexdigest()
    if resolved_sha256 != configuration.resolved_sha256:
        raise ValueError("loaded configuration digest does not match its resolved data")
    source = inspect_source_provenance(source_repository)
    if source.dirty and not configuration.resolved.runtime.allow_dirty_source:
        summary = ", ".join(source.status[:5])
        if len(source.status) > 5:
            summary += ", ..."
        raise RuntimeError(
            "source tree is dirty; commit the changes or set "
            f"runtime.allow_dirty_source=true ({summary})"
        )

    timestamp = started_at or datetime.now(timezone.utc)
    if timestamp.tzinfo is None:
        raise ValueError("started_at must be timezone-aware")
    timestamp = timestamp.astimezone(timezone.utc)
    identity = run_id or str(uuid.uuid4())
    try:
        identity = str(uuid.UUID(identity))
    except ValueError as error:
        raise ValueError("run_id must be a UUID") from error
    short_time = timestamp.strftime("%Y%m%dT%H%M%SZ")
    directory_name = (
        f"{_run_slug(configuration.resolved.name)}_{short_time}_{identity[:8]}"
    )
    resolved_device = resolve_runtime_device(configuration.resolved.runtime.device)
    output_root = configuration.resolved.artifacts.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    directory = output_root / directory_name
    directory.mkdir(exist_ok=False)

    seeds = derive_named_seeds(configuration.resolved.master_seed)
    actual_command = tuple(command) if command is not None else tuple(sys.argv)
    provenance = RunProvenance(
        run_id=identity,
        started_at_utc=timestamp.isoformat().replace("+00:00", "Z"),
        command=actual_command,
        configuration_source=str(configuration.source_path),
        configuration_source_sha256=configuration.source_sha256,
        resolved_configuration_sha256=configuration.resolved_sha256,
        seeds=seeds,
        source_revision=source.revision,
        source_dirty=source.dirty,
        source_status=source.status,
        source_patch_sha256=source.patch_sha256,
        python_version=platform.python_version(),
        torch_version=str(torch.__version__),
        cuda_version=torch.version.cuda,
        requested_device=configuration.resolved.runtime.device,
        device=resolved_device,
        dtype=configuration.resolved.runtime.dtype,
        parent_run_id=parent_run_id,
        parent_checkpoint=parent_checkpoint,
        configuration_differences=tuple(configuration_differences),
    )

    (directory / "experiment.toml").write_bytes(configuration.source_bytes)
    (directory / "resolved-config.json").write_bytes(resolved_bytes)
    (directory / "provenance.json").write_bytes(
        _canonical_json_bytes(_plain_data(provenance))
    )
    if source.patch:
        (directory / "source.patch").write_bytes(source.patch)
    return InitializedRun(directory, configuration, provenance)


def run_configuration_from_record(record: Mapping[str, Any]) -> RunConfiguration:
    """Reconstruct and validate a resolved configuration stored as evidence."""

    vocabulary = Vocabulary(**record["vocabulary"])
    model = TransformerConfig(**record["model"])
    rollout = RolloutSamplingConfig(**record["rollout"])
    reward = RewardConfig(**record["reward"])
    optimizer = OptimizerConfig(**record["optimizer"])
    grpo = GRPOConfig(**record["grpo"])
    curriculum_data = record["curriculum"]
    curriculum = CurriculumConfig(
        stages=tuple(
            GraphProblemConfig(**stage) for stage in curriculum_data["stages"]
        ),
        past_decay_scale=curriculum_data["past_decay_scale"],
        future_decay_scale=curriculum_data["future_decay_scale"],
        advancement_threshold=curriculum_data["advancement_threshold"],
        advancement_patience=curriculum_data["advancement_patience"],
    )
    training = TrainingScheduleConfig(**record["training"])
    evaluation_data = record["evaluation"]
    evaluation = EvaluationScheduleConfig(
        every_steps=evaluation_data["every_steps"],
        example_count=evaluation_data["example_count"],
        sampling=EvaluationSamplingConfig(**evaluation_data["sampling"]),
    )
    artifacts = ArtifactConfig(
        output_root=Path(record["artifacts"]["output_root"]),
        log_every=record["artifacts"]["log_every"],
        checkpoint_every=record["artifacts"]["checkpoint_every"],
        representative_every=record["artifacts"]["representative_every"],
        representative_count=record["artifacts"]["representative_count"],
        representative_selection=record["artifacts"][
            "representative_selection"
        ],
    )
    runtime = RuntimeConfig(**record["runtime"])
    return RunConfiguration(
        schema_version=record["schema_version"],
        name=record["name"],
        master_seed=record["master_seed"],
        vocabulary=vocabulary,
        model=model,
        rollout=rollout,
        reward=reward,
        optimizer=optimizer,
        grpo=grpo,
        curriculum=curriculum,
        training=training,
        evaluation=evaluation,
        artifacts=artifacts,
        runtime=runtime,
    )


def configuration_sha256(config: RunConfiguration) -> str:
    return hashlib.sha256(
        _canonical_json_bytes(resolved_configuration_record(config))
    ).hexdigest()


def configuration_differences(
    left: RunConfiguration, right: RunConfiguration
) -> tuple[str, ...]:
    """Return stable field-level differences without judging comparability."""

    differences: list[str] = []

    def visit(path: str, first: object, second: object) -> None:
        if isinstance(first, dict) and isinstance(second, dict):
            for key in sorted(set(first) | set(second)):
                child = f"{path}.{key}" if path else key
                if key not in first or key not in second:
                    differences.append(child)
                else:
                    visit(child, first[key], second[key])
            return
        if isinstance(first, list) and isinstance(second, list):
            if len(first) != len(second):
                differences.append(f"{path}.length")
            for index, (left_item, right_item) in enumerate(zip(first, second)):
                visit(f"{path}[{index}]", left_item, right_item)
            return
        if first != second:
            differences.append(path)

    visit(
        "",
        resolved_configuration_record(left),
        resolved_configuration_record(right),
    )
    return tuple(differences)


@dataclass
class NamedRandomStreams:
    training_problems: random.Random
    rollout_sampling: torch.Generator
    curriculum_validation: random.Random
    evaluation_sampling: torch.Generator

    def state_dict(self) -> dict[str, object]:
        return {
            "training_problems": self.training_problems.getstate(),
            "rollout_sampling": self.rollout_sampling.get_state(),
            "curriculum_validation": self.curriculum_validation.getstate(),
            "evaluation_sampling": self.evaluation_sampling.get_state(),
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        self.training_problems.setstate(state["training_problems"])
        self.rollout_sampling.set_state(state["rollout_sampling"])
        self.curriculum_validation.setstate(state["curriculum_validation"])
        self.evaluation_sampling.set_state(state["evaluation_sampling"])


def create_random_streams(seeds: Mapping[str, int]) -> NamedRandomStreams:
    missing = sorted(
        {
            "training_problems",
            "rollout_sampling",
            "curriculum_validation",
            "evaluation_sampling",
        }
        - set(seeds)
    )
    if missing:
        raise ValueError(f"missing named random seeds: {', '.join(missing)}")
    return NamedRandomStreams(
        training_problems=random.Random(seeds["training_problems"]),
        rollout_sampling=torch.Generator().manual_seed(seeds["rollout_sampling"]),
        curriculum_validation=random.Random(seeds["curriculum_validation"]),
        evaluation_sampling=torch.Generator().manual_seed(
            seeds["evaluation_sampling"]
        ),
    )


def snapshot_reference(reference: Transformer, policy: Transformer) -> None:
    reference.load_state_dict(policy.state_dict())
    reference.eval()
    for parameter in reference.parameters():
        parameter.requires_grad_(False)


CheckpointPurpose = Literal[
    "periodic", "curriculum_transition", "interruption", "final"
]


@dataclass(frozen=True)
class LoadedCheckpoint:
    path: Path
    run_id: str
    purpose: CheckpointPurpose
    step: int
    curriculum_state: CurriculumState
    configuration: RunConfiguration
    configuration_sha256: str
    payload: Mapping[str, Any]


def _atomic_torch_save(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        torch.save(value, temporary)
        with temporary.open("rb+") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_copy(source: Path, destination: Path) -> None:
    temporary = destination.with_name(
        f".{destination.name}.{uuid.uuid4().hex}.tmp"
    )
    try:
        shutil.copyfile(source, temporary)
        with temporary.open("rb+") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def save_training_checkpoint(
    run_directory: Path,
    *,
    run_id: str,
    purpose: CheckpointPurpose,
    step: int,
    curriculum_state: CurriculumState,
    configuration: RunConfiguration,
    policy: Transformer,
    reference: Transformer,
    optimizer: torch.optim.Optimizer,
    random_streams: NamedRandomStreams,
) -> Path:
    """Atomically preserve all state needed for exact training continuation."""

    if purpose not in (
        "periodic",
        "curriculum_transition",
        "interruption",
        "final",
    ):
        raise ValueError(f"unknown checkpoint purpose: {purpose}")
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ValueError("checkpoint step must be a nonnegative integer")
    curriculum_state.validate_for(configuration.curriculum)
    record = resolved_configuration_record(configuration)
    digest = configuration_sha256(configuration)
    payload: dict[str, object] = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        "run_id": run_id,
        "purpose": purpose,
        "step": step,
        "curriculum_state": asdict(curriculum_state),
        "configuration": record,
        "configuration_sha256": digest,
        "policy": policy.state_dict(),
        "reference": reference.state_dict(),
        "optimizer": optimizer.state_dict(),
        "random_streams": random_streams.state_dict(),
        "torch_rng_state": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        payload["cuda_rng_state"] = torch.cuda.get_rng_state_all()
    if torch.backends.mps.is_available() and hasattr(torch.mps, "get_rng_state"):
        payload["mps_rng_state"] = torch.mps.get_rng_state()

    checkpoint_directory = Path(run_directory) / "checkpoints"
    filename = f"step-{step:09d}-{purpose}.pt"
    path = checkpoint_directory / filename
    _atomic_torch_save(payload, path)
    _atomic_copy(path, checkpoint_directory / "latest.pt")
    return path


def load_training_checkpoint(path: str | Path) -> LoadedCheckpoint:
    checkpoint_path = Path(path).expanduser().resolve()
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("checkpoint payload must be a mapping")
    if payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("unsupported checkpoint schema version")
    required = {
        "run_id",
        "purpose",
        "step",
        "curriculum_state",
        "configuration",
        "configuration_sha256",
        "policy",
        "reference",
        "optimizer",
        "random_streams",
        "torch_rng_state",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"checkpoint is missing: {', '.join(missing)}")
    configuration = run_configuration_from_record(payload["configuration"])
    digest = configuration_sha256(configuration)
    if digest != payload["configuration_sha256"]:
        raise ValueError("checkpoint configuration digest is invalid")
    purpose = payload["purpose"]
    if purpose not in (
        "periodic",
        "curriculum_transition",
        "interruption",
        "final",
    ):
        raise ValueError("checkpoint purpose is invalid")
    step = payload["step"]
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ValueError("checkpoint step is invalid")
    state = CurriculumState(**payload["curriculum_state"])
    state.validate_for(configuration.curriculum)
    return LoadedCheckpoint(
        path=checkpoint_path,
        run_id=_nonempty_string(payload["run_id"], "checkpoint.run_id"),
        purpose=purpose,
        step=step,
        curriculum_state=state,
        configuration=configuration,
        configuration_sha256=digest,
        payload=payload,
    )


def validate_resume_configuration(
    checkpoint: LoadedCheckpoint,
    requested: RunConfiguration,
    *,
    derived_run: bool,
) -> tuple[str, ...]:
    """Reject incompatible parameter semantics and identify permitted changes."""

    if requested.vocabulary != checkpoint.configuration.vocabulary:
        raise ValueError("requested task vocabulary is incompatible with checkpoint")
    if requested.model != checkpoint.configuration.model:
        raise ValueError("requested model architecture is incompatible with checkpoint")
    if requested.master_seed != checkpoint.configuration.master_seed:
        raise ValueError("requested master seed is incompatible with checkpoint")
    if requested.curriculum.stages != checkpoint.configuration.curriculum.stages:
        raise ValueError(
            "requested curriculum stages are incompatible with checkpoint"
        )
    differences = configuration_differences(checkpoint.configuration, requested)
    if differences and not derived_run:
        raise ValueError(
            "same-run resumption requires the recorded configuration; use a "
            "derived run for changed experimental settings"
        )
    return differences


def restore_training_checkpoint(
    checkpoint: LoadedCheckpoint,
    *,
    configuration: RunConfiguration,
    derived_run: bool,
    policy: Transformer,
    reference: Transformer,
    optimizer: torch.optim.Optimizer,
    random_streams: NamedRandomStreams,
) -> tuple[CurriculumState, tuple[str, ...]]:
    differences = validate_resume_configuration(
        checkpoint, configuration, derived_run=derived_run
    )
    policy.load_state_dict(checkpoint.payload["policy"])
    reference.load_state_dict(checkpoint.payload["reference"])
    reference.eval()
    for parameter in reference.parameters():
        parameter.requires_grad_(False)
    optimizer.load_state_dict(checkpoint.payload["optimizer"])
    random_streams.load_state_dict(checkpoint.payload["random_streams"])
    torch.set_rng_state(checkpoint.payload["torch_rng_state"])
    if torch.cuda.is_available() and "cuda_rng_state" in checkpoint.payload:
        torch.cuda.set_rng_state_all(checkpoint.payload["cuda_rng_state"])
    if (
        torch.backends.mps.is_available()
        and "mps_rng_state" in checkpoint.payload
        and hasattr(torch.mps, "set_rng_state")
    ):
        torch.mps.set_rng_state(checkpoint.payload["mps_rng_state"])
    return checkpoint.curriculum_state, differences


def _append_json_line(path: Path, record: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = _canonical_json_bytes(record)
    with path.open("ab") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


@dataclass(frozen=True)
class EvidenceWriter:
    run_directory: Path
    run_id: str
    configuration_sha256: str

    def event(
        self,
        kind: str,
        step: int,
        payload: Mapping[str, object],
    ) -> None:
        record = {
            "schema_version": EVIDENCE_SCHEMA_VERSION,
            "kind": kind,
            "run_id": self.run_id,
            "step": step,
            "configuration_sha256": self.configuration_sha256,
            "recorded_at_utc": datetime.now(timezone.utc)
            .isoformat()
            .replace("+00:00", "Z"),
            **payload,
        }
        _append_json_line(self.run_directory / "metrics.jsonl", record)

    def completion(self, step: int, payload: Mapping[str, object]) -> None:
        record = {
            "schema_version": EVIDENCE_SCHEMA_VERSION,
            "kind": "representative_completion",
            "run_id": self.run_id,
            "step": step,
            "configuration_sha256": self.configuration_sha256,
            **payload,
        }
        _append_json_line(self.run_directory / "completions.jsonl", record)


def _mean(values: Sequence[float]) -> float:
    return math.fsum(values) / len(values) if values else 0.0


def training_evidence_payload(
    batch: TrainingProblemBatch,
    rollouts: Sequence[Rollout],
    update: UpdateMetrics,
    configuration: RunConfiguration,
) -> dict[str, object]:
    count = len(rollouts)
    return {
        "frontier": batch.frontier,
        "problem_count": len(batch.examples),
        "rollout_count": count,
        "stage_probabilities": list(batch.stage_probabilities),
        "stage_counts": list(batch.stage_counts),
        "format_successes": sum(rollout.outcome.format_ok for rollout in rollouts),
        "valid_path_successes": sum(
            rollout.outcome.valid_path for rollout in rollouts
        ),
        "shortest_path_successes": sum(
            rollout.outcome.shortest for rollout in rollouts
        ),
        "mean_base_reward": _mean(
            [rollout.reward.base_reward for rollout in rollouts]
        ),
        "mean_reasoning_coverage": _mean(
            [rollout.reward.reasoning_coverage for rollout in rollouts]
        ),
        "mean_sterile_repetition_rate_all": _mean(
            [rollout.reward.sterile_repetition_rate_all for rollout in rollouts]
        ),
        "mean_sterile_repetition_rate_off_answer": _mean(
            [
                rollout.reward.sterile_repetition_rate_off_answer
                for rollout in rollouts
            ]
        ),
        "mean_sterile_repetition_penalty": _mean(
            [rollout.reward.sterile_repetition_penalty for rollout in rollouts]
        ),
        "mean_total_reward": _mean(
            [rollout.reward.total_reward for rollout in rollouts]
        ),
        "coverage_coefficient": configuration.reward.coverage_coefficient,
        "sterile_repetition_coefficient": (
            configuration.reward.sterile_repetition_coefficient
        ),
        "sterile_repetition_mode": configuration.reward.sterile_repetition_mode,
        "policy_loss": update.policy_loss,
        "sampled_kl": update.sampled_kl,
        "valid_next_loss": update.valid_loss,
        "valid_next_mass": update.valid_mass,
        "total_loss": update.total_loss,
        "kl_coefficient": configuration.grpo.kl_coefficient,
        "valid_coefficient": configuration.grpo.valid_coefficient,
        "zero_variance_group_fraction": update.zero_variance_group_fraction,
        "gradient_norm": update.gradient_norm,
        "sampling": _plain_data(configuration.rollout),
    }


def evaluation_evidence_payload(
    result: EvaluationResult,
    *,
    frontier: int | None,
) -> dict[str, object]:
    return {
        "frontier": frontier,
        "policy_checkpoint": result.protocol.policy_checkpoint,
        "problem_set_identity": result.problem_set_identity,
        "problem_config": _plain_data(result.problem_config),
        "sampling": _plain_data(result.protocol.sampling),
        "sampling_seed": result.protocol.sampling_seed,
        "protocol_kind": result.protocol.kind,
        "metrics": _plain_data(result.metrics),
    }


def representative_completion_payload(
    rollout: Rollout,
    *,
    vocabulary: Vocabulary,
    stage_index: int,
    selection_rule: str,
) -> dict[str, object]:
    example = rollout.example
    return {
        "selection_rule": selection_rule,
        "stage_index": stage_index,
        "graph": {
            "vertex_count": example.problem.vertex_count,
            "abstract_edges": _plain_data(example.problem.edges),
            "abstract_source": example.problem.source,
            "abstract_target": example.problem.target,
            "shortest_distance": example.problem.shortest_distance,
            "label_by_vertex": list(example.label_by_vertex),
            "visible_edges": _plain_data(example.edges),
            "serialized_edges": _plain_data(example.serialized_edges),
            "source": example.source,
            "target": example.target,
        },
        "prompt_tokens": list(example.prompt),
        "prompt_text": vocabulary.render(example.prompt),
        "completion_tokens": list(rollout.completion),
        "completion_text": vocabulary.render(rollout.completion),
        "terminated_by_eos": rollout.sample.terminated_by_eos,
        "parsed_and_verified": _plain_data(rollout.outcome),
        "reward": _plain_data(rollout.reward),
    }


def resolve_autocast_dtype(
    dtype: str, device: torch.device
) -> torch.dtype | None:
    if dtype == "float32":
        return None
    if device.type == "cuda":
        if dtype == "bfloat16":
            if not torch.cuda.is_bf16_supported():
                raise RuntimeError("bfloat16 was requested but is unavailable")
            return torch.bfloat16
        return torch.float16
    if device.type == "mps":
        if dtype == "float16":
            return torch.float16
        raise RuntimeError("bfloat16 autocast is not supported on MPS")
    return None


def _build_training_components(
    configuration: RunConfiguration,
    *,
    device: torch.device,
    seeds: Mapping[str, int],
) -> tuple[
    Transformer,
    Transformer,
    torch.optim.AdamW,
    NamedRandomStreams,
]:
    torch.manual_seed(seeds["model_initialization"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seeds["model_initialization"])
    policy = Transformer(configuration.model).to(device)
    reference = Transformer(configuration.model).to(device)
    snapshot_reference(reference, policy)
    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=configuration.optimizer.learning_rate,
        betas=(configuration.optimizer.beta1, configuration.optimizer.beta2),
        weight_decay=configuration.optimizer.weight_decay,
    )
    streams = create_random_streams(seeds)
    return policy, reference, optimizer, streams


def _apply_optimizer_configuration(
    optimizer: torch.optim.Optimizer, configuration: OptimizerConfig
) -> None:
    for group in optimizer.param_groups:
        group["lr"] = configuration.learning_rate
        group["weight_decay"] = configuration.weight_decay
        group["betas"] = (configuration.beta1, configuration.beta2)


@dataclass(frozen=True)
class TrainingRunResult:
    run_directory: Path
    run_id: str
    final_step: int
    final_checkpoint: Path
    curriculum_state: CurriculumState


def execute_training(
    *,
    run_directory: Path,
    run_id: str,
    configuration: RunConfiguration,
    seeds: Mapping[str, int],
    checkpoint: LoadedCheckpoint | None = None,
    derived_run: bool = False,
    max_steps_override: int | None = None,
) -> TrainingRunResult:
    """Run or resume the complete on-policy curriculum training protocol."""

    selected_device = resolve_runtime_device(configuration.runtime.device)
    device = torch.device(selected_device)
    autocast_dtype = resolve_autocast_dtype(configuration.runtime.dtype, device)
    policy, reference, optimizer, streams = _build_training_components(
        configuration, device=device, seeds=seeds
    )
    step = 0
    curriculum_state = CurriculumState()
    differences: tuple[str, ...] = ()
    if checkpoint is not None:
        curriculum_state, differences = restore_training_checkpoint(
            checkpoint,
            configuration=configuration,
            derived_run=derived_run,
            policy=policy,
            reference=reference,
            optimizer=optimizer,
            random_streams=streams,
        )
        _apply_optimizer_configuration(optimizer, configuration.optimizer)
        step = checkpoint.step

    maximum_step = (
        configuration.training.max_steps
        if max_steps_override is None
        else _positive_int(max_steps_override, "max_steps_override")
    )
    if maximum_step < step:
        raise ValueError("maximum training step precedes the checkpoint step")
    writer = EvidenceWriter(
        Path(run_directory), run_id, configuration_sha256(configuration)
    )
    writer.event(
        "run_started" if checkpoint is None else "run_resumed",
        step,
        {
            "maximum_step": maximum_step,
            "parent_checkpoint": str(checkpoint.path) if checkpoint else None,
            "configuration_differences": list(differences),
            "derived_run": derived_run,
        },
    )

    latest_checkpoint: Path | None = checkpoint.path if checkpoint else None
    interrupt_requested = False
    safe_to_checkpoint = True
    previous_sigint: object | None = None

    def defer_sigint(_signum: int, _frame: object) -> None:
        nonlocal interrupt_requested
        if interrupt_requested:
            raise KeyboardInterrupt
        interrupt_requested = True

    if threading.current_thread() is threading.main_thread():
        previous_sigint = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, defer_sigint)
    try:
        while step < maximum_step:
            safe_to_checkpoint = False
            batch = generate_training_problem_batch(
                configuration.curriculum,
                curriculum_state,
                configuration.vocabulary,
                configuration.training.problems_per_step,
                streams.training_problems,
            )
            samples = collect_grouped_completions(
                policy,
                batch.examples,
                configuration.rollout,
                device=device,
                generator=streams.rollout_sampling,
                autocast_dtype=autocast_dtype,
            )
            rollouts = evaluate_samples(
                samples,
                configuration.vocabulary,
                configuration.reward.coverage_coefficient,
                configuration.reward.sterile_repetition_coefficient,
                configuration.reward.sterile_repetition_mode,
            )
            packed = pack_rollouts(
                rollouts, configuration.vocabulary, device=device
            )
            update = grpo_update(
                policy,
                reference,
                optimizer,
                packed,
                configuration.grpo,
                autocast_dtype=autocast_dtype,
            )
            step += 1

            if step % configuration.training.reference_update_every == 0:
                snapshot_reference(reference, policy)
                writer.event("reference_updated", step, {})

            if step % configuration.artifacts.log_every == 0 or step == 1:
                writer.event(
                    "training",
                    step,
                    training_evidence_payload(
                        batch, rollouts, update, configuration
                    ),
                )

            if step % configuration.artifacts.representative_every == 0:
                selected = rollouts[:: configuration.rollout.group_size][
                    : configuration.artifacts.representative_count
                ]
                for rollout in selected:
                    writer.completion(
                        step,
                        representative_completion_payload(
                            rollout,
                            vocabulary=configuration.vocabulary,
                            stage_index=batch.stage_indices[rollout.group_id],
                            selection_rule=(
                                configuration.artifacts.representative_selection
                            ),
                        ),
                    )

            transitioned = False
            if step % configuration.evaluation.every_steps == 0:
                generation_seed = streams.curriculum_validation.randrange(2**63)
                problem_set = generate_evaluation_problem_set(
                    configuration.curriculum.stages[curriculum_state.frontier],
                    configuration.vocabulary,
                    configuration.evaluation.example_count,
                    generation_seed,
                )
                sampling_seed = None
                if configuration.evaluation.sampling.mode == "stochastic":
                    sampling_seed = int(
                        torch.randint(
                            0,
                            2**63 - 1,
                            (1,),
                            generator=streams.evaluation_sampling,
                        ).item()
                    )
                protocol = EvaluationProtocol(
                    policy_checkpoint=f"{run_id}:live-step-{step}",
                    sampling=configuration.evaluation.sampling,
                    sampling_seed=sampling_seed,
                )
                evaluation = run_evaluation(
                    policy,
                    protocol,
                    problem_set,
                    configuration.vocabulary,
                    device=device,
                    autocast_dtype=autocast_dtype,
                )
                writer.event(
                    "evaluation",
                    step,
                    evaluation_evidence_payload(
                        evaluation, frontier=curriculum_state.frontier
                    ),
                )
                decision = apply_frontier_validation(
                    configuration.curriculum,
                    curriculum_state,
                    FrontierValidation(
                        training_step=step,
                        frontier=curriculum_state.frontier,
                        example_count=evaluation.metrics.example_count,
                        shortest_path_successes=(
                            evaluation.metrics.shortest_path_successes
                        ),
                    ),
                )
                curriculum_state = decision.resulting_state
                writer.event(
                    "curriculum_validation",
                    step,
                    {
                        "frontier": decision.validation.frontier,
                        "example_count": decision.validation.example_count,
                        "shortest_path_successes": (
                            decision.validation.shortest_path_successes
                        ),
                        "threshold": decision.threshold,
                        "qualified": decision.qualified,
                        "resulting_frontier": curriculum_state.frontier,
                        "resulting_streak": curriculum_state.advancement_streak,
                        "transition": _plain_data(decision.transition),
                    },
                )
                if decision.transition is not None:
                    transitioned = True
                    latest_checkpoint = save_training_checkpoint(
                        Path(run_directory),
                        run_id=run_id,
                        purpose="curriculum_transition",
                        step=step,
                        curriculum_state=curriculum_state,
                        configuration=configuration,
                        policy=policy,
                        reference=reference,
                        optimizer=optimizer,
                        random_streams=streams,
                    )

            if (
                step % configuration.artifacts.checkpoint_every == 0
                and not transitioned
            ):
                latest_checkpoint = save_training_checkpoint(
                    Path(run_directory),
                    run_id=run_id,
                    purpose="periodic",
                    step=step,
                    curriculum_state=curriculum_state,
                    configuration=configuration,
                    policy=policy,
                    reference=reference,
                    optimizer=optimizer,
                    random_streams=streams,
                )
            safe_to_checkpoint = True
            if interrupt_requested:
                raise KeyboardInterrupt
    except KeyboardInterrupt:
        if not safe_to_checkpoint:
            writer.event(
                "run_interrupted_unsafe_boundary",
                step,
                {
                    "last_usable_checkpoint": (
                        str(latest_checkpoint) if latest_checkpoint else None
                    )
                },
            )
            raise
        latest_checkpoint = save_training_checkpoint(
            Path(run_directory),
            run_id=run_id,
            purpose="interruption",
            step=step,
            curriculum_state=curriculum_state,
            configuration=configuration,
            policy=policy,
            reference=reference,
            optimizer=optimizer,
            random_streams=streams,
        )
        writer.event(
            "run_interrupted", step, {"checkpoint": str(latest_checkpoint)}
        )
        raise
    finally:
        if previous_sigint is not None:
            signal.signal(signal.SIGINT, previous_sigint)

    final_checkpoint = save_training_checkpoint(
        Path(run_directory),
        run_id=run_id,
        purpose="final",
        step=step,
        curriculum_state=curriculum_state,
        configuration=configuration,
        policy=policy,
        reference=reference,
        optimizer=optimizer,
        random_streams=streams,
    )
    writer.event(
        "run_completed", step, {"checkpoint": str(final_checkpoint)}
    )
    return TrainingRunResult(
        Path(run_directory), run_id, step, final_checkpoint, curriculum_state
    )


def start_training_run(
    configuration_path: str | Path,
    *,
    source_repository: str | Path,
    command: Sequence[str] | None = None,
) -> TrainingRunResult:
    loaded = load_run_configuration(configuration_path)
    initialized = initialize_run(
        loaded, source_repository=source_repository, command=command
    )
    return execute_training(
        run_directory=initialized.directory,
        run_id=initialized.provenance.run_id,
        configuration=loaded.resolved,
        seeds=initialized.provenance.seeds,
    )


def _checkpoint_run_directory(checkpoint: LoadedCheckpoint) -> Path:
    if checkpoint.path.parent.name != "checkpoints":
        raise ValueError("checkpoint must be located in a run's checkpoints directory")
    run_directory = checkpoint.path.parent.parent
    if not (run_directory / "provenance.json").is_file():
        raise ValueError("checkpoint run directory has no provenance record")
    return run_directory


def resume_training_run(
    checkpoint_path: str | Path,
    *,
    source_repository: str | Path,
    configuration_path: str | Path | None = None,
    max_steps: int | None = None,
    command: Sequence[str] | None = None,
) -> TrainingRunResult:
    checkpoint = load_training_checkpoint(checkpoint_path)
    source_run_directory = _checkpoint_run_directory(checkpoint)
    source_provenance = json.loads(
        (source_run_directory / "provenance.json").read_text(encoding="utf-8")
    )
    if source_provenance.get("run_id") != checkpoint.run_id:
        raise ValueError("checkpoint and run provenance identities disagree")
    seeds = source_provenance.get("seeds")
    if not isinstance(seeds, dict):
        raise ValueError("run provenance has no named random seeds")

    if configuration_path is None:
        source = inspect_source_provenance(source_repository)
        if source.dirty and not checkpoint.configuration.runtime.allow_dirty_source:
            raise RuntimeError("source tree is dirty during same-run resumption")
        if source.revision != source_provenance.get("source_revision"):
            raise RuntimeError(
                "same-run resumption requires the originating source revision; "
                "provide a configuration to create a derived run"
            )
        writer = EvidenceWriter(
            source_run_directory,
            checkpoint.run_id,
            checkpoint.configuration_sha256,
        )
        writer.event(
            "resume_requested",
            checkpoint.step,
            {
                "checkpoint": str(checkpoint.path),
                "command": list(command or sys.argv),
                "max_steps_override": max_steps,
            },
        )
        return execute_training(
            run_directory=source_run_directory,
            run_id=checkpoint.run_id,
            configuration=checkpoint.configuration,
            seeds=seeds,
            checkpoint=checkpoint,
            max_steps_override=max_steps,
        )

    requested = load_run_configuration(configuration_path)
    differences = validate_resume_configuration(
        checkpoint, requested.resolved, derived_run=True
    )
    initialized = initialize_run(
        requested,
        source_repository=source_repository,
        command=command,
        parent_run_id=checkpoint.run_id,
        parent_checkpoint=str(checkpoint.path),
        configuration_differences=differences,
    )
    return execute_training(
        run_directory=initialized.directory,
        run_id=initialized.provenance.run_id,
        configuration=requested.resolved,
        seeds=seeds,
        checkpoint=checkpoint,
        derived_run=True,
        max_steps_override=max_steps,
    )


@dataclass(frozen=True)
class DeclaredEvaluationSet:
    name: str
    problem_config: GraphProblemConfig
    example_count: int
    generation_seed: int
    sampling_seed: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _nonempty_string(self.name, "set.name"))
        object.__setattr__(
            self,
            "example_count",
            _positive_int(self.example_count, "set.example_count"),
        )
        seed = _plain_int(self.generation_seed, "set.generation_seed")
        if not 0 <= seed < 2**63:
            raise ConfigurationError("set.generation_seed must lie in [0, 2^63)")
        if self.sampling_seed is not None:
            sampling_seed = _plain_int(
                self.sampling_seed, "set.sampling_seed"
            )
            if not 0 <= sampling_seed < 2**63:
                raise ConfigurationError(
                    "set.sampling_seed must lie in [0, 2^63)"
                )


@dataclass(frozen=True)
class StandaloneEvaluationConfig:
    schema_version: int
    name: str
    sampling: EvaluationSamplingConfig
    sets: tuple[DeclaredEvaluationSet, ...]
    output_root: Path
    runtime: RuntimeConfig

    def __post_init__(self) -> None:
        if self.schema_version != CONFIG_SCHEMA_VERSION:
            raise ConfigurationError("unsupported evaluation schema version")
        object.__setattr__(self, "name", _nonempty_string(self.name, "name"))
        sets = tuple(self.sets)
        if not sets:
            raise ConfigurationError("evaluation requires at least one set")
        if len({item.name for item in sets}) != len(sets):
            raise ConfigurationError("evaluation set names must be unique")
        for item in sets:
            if self.sampling.mode == "greedy" and item.sampling_seed is not None:
                raise ConfigurationError(
                    "greedy evaluation sets do not use sampling_seed"
                )
            if self.sampling.mode == "stochastic" and item.sampling_seed is None:
                raise ConfigurationError(
                    "stochastic evaluation sets require sampling_seed"
                )
        object.__setattr__(self, "sets", sets)
        object.__setattr__(self, "output_root", Path(self.output_root))


def load_standalone_evaluation_config(
    path: str | Path,
) -> StandaloneEvaluationConfig:
    source_path = Path(path).expanduser().resolve()
    try:
        declaration = tomllib.loads(source_path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise ConfigurationError(f"invalid evaluation TOML: {error}") from error
    _reject_unknown(
        declaration,
        {"schema_version", "name", "sampling", "sets", "artifacts", "runtime"},
        "root",
    )
    sampling_data = _table(declaration, "sampling", required=True)
    _reject_unknown(
        sampling_data,
        {"max_new_tokens", "batch_size", "mode", "temperature", "top_p"},
        "sampling",
    )
    if "max_new_tokens" not in sampling_data:
        raise ConfigurationError("missing required key: sampling.max_new_tokens")
    sampling = EvaluationSamplingConfig(**sampling_data)
    sets_data = declaration.get("sets")
    if not isinstance(sets_data, list):
        raise ConfigurationError("sets must be an array of tables")
    declared_sets = []
    allowed_set_keys = {
        "name",
        "vertices",
        "edges",
        "min_distance",
        "max_distance",
        "generation_attempts",
        "example_count",
        "generation_seed",
        "sampling_seed",
    }
    for index, item in enumerate(sets_data):
        if not isinstance(item, dict):
            raise ConfigurationError(f"sets[{index}] must be a table")
        _reject_unknown(item, allowed_set_keys, f"sets[{index}]")
        required = {
            "name",
            "vertices",
            "edges",
            "min_distance",
            "max_distance",
            "example_count",
            "generation_seed",
        }
        missing = sorted(required - set(item))
        if missing:
            raise ConfigurationError(
                f"sets[{index}] is missing: {', '.join(missing)}"
            )
        declared_sets.append(
            DeclaredEvaluationSet(
                name=item["name"],
                problem_config=GraphProblemConfig(
                    vertices=item["vertices"],
                    edges=item["edges"],
                    min_distance=item["min_distance"],
                    max_distance=item["max_distance"],
                    generation_attempts=item.get("generation_attempts", 100),
                ),
                example_count=item["example_count"],
                generation_seed=item["generation_seed"],
                sampling_seed=item.get("sampling_seed"),
            )
        )
    artifacts = _table(declaration, "artifacts", required=True)
    _reject_unknown(artifacts, {"output_root"}, "artifacts")
    if "output_root" not in artifacts:
        raise ConfigurationError("missing required key: artifacts.output_root")
    output_root = Path(
        _nonempty_string(artifacts["output_root"], "artifacts.output_root")
    ).expanduser()
    if not output_root.is_absolute():
        output_root = source_path.parent / output_root
    runtime_data = _table(declaration, "runtime")
    _reject_unknown(runtime_data, {"device", "dtype"}, "runtime")
    runtime = RuntimeConfig(
        device=runtime_data.get("device", "auto"),
        dtype=runtime_data.get("dtype", "bfloat16"),
        allow_dirty_source=False,
    )
    for key in ("schema_version", "name"):
        if key not in declaration:
            raise ConfigurationError(f"missing required key: {key}")
    return StandaloneEvaluationConfig(
        schema_version=declaration["schema_version"],
        name=declaration["name"],
        sampling=sampling,
        sets=tuple(declared_sets),
        output_root=output_root.resolve(),
        runtime=runtime,
    )


@dataclass(frozen=True)
class StandaloneEvaluationResult:
    directory: Path
    evaluation_id: str
    results: tuple[EvaluationResult, ...]


def execute_standalone_evaluation(
    checkpoint_path: str | Path,
    evaluation_configuration_path: str | Path,
) -> StandaloneEvaluationResult:
    checkpoint = load_training_checkpoint(checkpoint_path)
    declaration = load_standalone_evaluation_config(
        evaluation_configuration_path
    )
    for item in declaration.sets:
        if (
            item.problem_config.vertices
            > checkpoint.configuration.vocabulary.node_label_count
        ):
            raise ValueError(
                f"evaluation set {item.name} exceeds the checkpoint vocabulary"
            )
    device = torch.device(resolve_runtime_device(declaration.runtime.device))
    autocast_dtype = resolve_autocast_dtype(declaration.runtime.dtype, device)
    policy = Transformer(checkpoint.configuration.model).to(device)
    policy.load_state_dict(checkpoint.payload["policy"])
    policy.eval()

    evaluation_id = str(uuid.uuid4())
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    directory = declaration.output_root / (
        f"{_run_slug(declaration.name)}_{timestamp}_{evaluation_id[:8]}"
    )
    directory.mkdir(parents=True, exist_ok=False)
    results = []
    for item in declaration.sets:
        problem_set = generate_evaluation_problem_set(
            item.problem_config,
            checkpoint.configuration.vocabulary,
            item.example_count,
            item.generation_seed,
        )
        protocol = EvaluationProtocol(
            policy_checkpoint=str(checkpoint.path),
            sampling=declaration.sampling,
            sampling_seed=item.sampling_seed,
        )
        result = run_evaluation(
            policy,
            protocol,
            problem_set,
            checkpoint.configuration.vocabulary,
            device=device,
            autocast_dtype=autocast_dtype,
        )
        results.append(result)
        _append_json_line(
            directory / "metrics.jsonl",
            {
                "schema_version": EVIDENCE_SCHEMA_VERSION,
                "kind": "evaluation",
                "evaluation_id": evaluation_id,
                "evaluation_name": declaration.name,
                "set_name": item.name,
                "checkpoint_run_id": checkpoint.run_id,
                "checkpoint": str(checkpoint.path),
                "checkpoint_step": checkpoint.step,
                "checkpoint_configuration_sha256": (
                    checkpoint.configuration_sha256
                ),
                **evaluation_evidence_payload(result, frontier=None),
            },
        )
        for case in result.cases:
            _append_json_line(
                directory / "completions.jsonl",
                {
                    "schema_version": EVIDENCE_SCHEMA_VERSION,
                    "kind": "evaluation_completion",
                    "evaluation_id": evaluation_id,
                    "set_name": item.name,
                    "problem": _plain_data(case.completion.example),
                    "completion_tokens": list(case.completion.completion),
                    "completion_text": checkpoint.configuration.vocabulary.render(
                        case.completion.completion
                    ),
                    "terminated_by_eos": case.completion.terminated_by_eos,
                    "outcome": _plain_data(case.outcome),
                },
            )
    return StandaloneEvaluationResult(directory, evaluation_id, tuple(results))
