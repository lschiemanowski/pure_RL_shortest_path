"""Resolved experiment declarations and reproducible run initialization."""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from numbers import Real
from pathlib import Path
import platform
import re
import subprocess
import sys
import tomllib
from typing import Any, Literal, Mapping, Sequence
import uuid

import torch

from .experiment import CurriculumConfig, EvaluationSamplingConfig
from .model import TransformerConfig
from .rl import GRPOConfig, RolloutSamplingConfig
from .task import GraphProblemConfig, Vocabulary


CONFIG_SCHEMA_VERSION = 1
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

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "coverage_coefficient",
            _nonnegative_real(
                self.coverage_coefficient, "reward.coverage_coefficient"
            ),
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
    _reject_unknown(reward_data, {"coverage_coefficient"}, "reward")
    reward = RewardConfig(reward_data.get("coverage_coefficient", 0.0))

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
    )

    (directory / "experiment.toml").write_bytes(configuration.source_bytes)
    (directory / "resolved-config.json").write_bytes(resolved_bytes)
    (directory / "provenance.json").write_bytes(
        _canonical_json_bytes(_plain_data(provenance))
    )
    if source.patch:
        (directory / "source.patch").write_bytes(source.patch)
    return InitializedRun(directory, configuration, provenance)
