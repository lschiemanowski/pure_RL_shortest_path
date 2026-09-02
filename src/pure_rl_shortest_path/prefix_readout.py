"""Paired reasoning-prefix readouts from a frozen shortest-path policy."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import sys
import tomllib
from typing import Any, Mapping, Sequence
import uuid

import torch

from .experiment import (
    EvaluationProtocol,
    EvaluationResult,
    EvaluationSamplingConfig,
    generate_evaluation_problem_set,
    run_evaluation,
    select_evaluation_tokens,
)
from .model import Transformer, generate_tokens
from .run import (
    ConfigurationError,
    RuntimeConfig,
    inspect_source_provenance,
    load_training_checkpoint,
    resolve_autocast_dtype,
    resolve_runtime_device,
)
from .task import (
    BEGIN_ANSWER,
    END_REASON,
    EOS,
    JUMP,
    PAD,
    GraphExample,
    GraphProblemConfig,
    OutcomeFacts,
    Vocabulary,
    parse_completion,
    verify_completion,
)


PREFIX_READOUT_SCHEMA_VERSION = 1
EVIDENCE_SCHEMA_VERSION = 1


def _plain_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigurationError(f"{name} must be an integer")
    return value


def _positive_int(value: object, name: str) -> int:
    result = _plain_int(value, name)
    if result <= 0:
        raise ConfigurationError(f"{name} must be positive")
    return result


def _seed(value: object, name: str) -> int:
    result = _plain_int(value, name)
    if not 0 <= result < 2**63:
        raise ConfigurationError(f"{name} must lie in [0, 2^63)")
    return result


def _nonempty_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"{name} must be a nonempty string")
    return value.strip()


def _strict_table(
    value: object, name: str, allowed_keys: frozenset[str]
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigurationError(f"{name} must be a TOML table")
    unexpected = sorted(set(value) - allowed_keys)
    if unexpected:
        joined = ", ".join(f"{name}.{key}" for key in unexpected)
        raise ConfigurationError(f"unexpected configuration key(s): {joined}")
    return dict(value)


def _required(table: Mapping[str, Any], key: str, location: str) -> Any:
    if key not in table:
        raise ConfigurationError(f"missing required key: {location}.{key}")
    return table[key]


@dataclass(frozen=True)
class DeclaredPrefixReadoutProblem:
    problem_config: GraphProblemConfig
    example_count: int
    generation_seed: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "example_count", _positive_int(self.example_count, "set.example_count")
        )
        object.__setattr__(
            self,
            "generation_seed",
            _seed(self.generation_seed, "set.generation_seed"),
        )


@dataclass(frozen=True)
class PrefixReadoutConfig:
    """Resolved declaration for one paired prefix-readout study."""

    schema_version: int
    name: str
    checkpoint_sha256: str
    baseline_sampling: EvaluationSamplingConfig
    readout_sampling: EvaluationSamplingConfig
    problem: DeclaredPrefixReadoutProblem
    output_root: Path
    runtime: RuntimeConfig

    def __post_init__(self) -> None:
        if self.schema_version != PREFIX_READOUT_SCHEMA_VERSION:
            raise ConfigurationError(
                f"unsupported schema_version {self.schema_version}; expected "
                f"{PREFIX_READOUT_SCHEMA_VERSION}"
            )
        object.__setattr__(self, "name", _nonempty_string(self.name, "name"))
        digest = _nonempty_string(
            self.checkpoint_sha256, "checkpoint_sha256"
        ).lower()
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ConfigurationError("checkpoint_sha256 must be 64 hexadecimal digits")
        if self.baseline_sampling.mode != "greedy":
            raise ConfigurationError("baseline_sampling must be greedy")
        if self.readout_sampling.mode != "greedy":
            raise ConfigurationError("readout_sampling must be greedy")
        object.__setattr__(self, "checkpoint_sha256", digest)
        object.__setattr__(self, "output_root", Path(self.output_root).resolve())


def _sampling_config(value: object, name: str) -> EvaluationSamplingConfig:
    table = _strict_table(
        value,
        name,
        frozenset({"max_new_tokens", "batch_size", "mode", "temperature", "top_p"}),
    )
    return EvaluationSamplingConfig(
        max_new_tokens=_required(table, "max_new_tokens", name),
        batch_size=table.get("batch_size"),
        mode=table.get("mode", "greedy"),
        temperature=table.get("temperature"),
        top_p=table.get("top_p"),
    )


def load_prefix_readout_config(path: str | Path) -> PrefixReadoutConfig:
    """Load a strict TOML declaration and resolve its output path."""

    source_path = Path(path).expanduser().resolve()
    try:
        declaration = tomllib.loads(source_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise ConfigurationError(f"invalid prefix-readout TOML: {error}") from error
    root = _strict_table(
        declaration,
        "root",
        frozenset(
            {
                "schema_version",
                "name",
                "checkpoint_sha256",
                "baseline_sampling",
                "readout_sampling",
                "set",
                "artifacts",
                "runtime",
            }
        ),
    )
    set_table = _strict_table(
        _required(root, "set", "root"),
        "set",
        frozenset(
            {
                "vertices",
                "edges",
                "min_distance",
                "max_distance",
                "generation_attempts",
                "example_count",
                "generation_seed",
            }
        ),
    )
    artifacts = _strict_table(
        _required(root, "artifacts", "root"),
        "artifacts",
        frozenset({"output_root"}),
    )
    runtime_table = _strict_table(
        _required(root, "runtime", "root"),
        "runtime",
        frozenset({"device", "dtype", "allow_dirty_source"}),
    )
    output_root = Path(
        _nonempty_string(
            _required(artifacts, "output_root", "artifacts"),
            "artifacts.output_root",
        )
    ).expanduser()
    if not output_root.is_absolute():
        output_root = source_path.parent / output_root
    problem_config = GraphProblemConfig(
        vertices=_required(set_table, "vertices", "set"),
        edges=_required(set_table, "edges", "set"),
        min_distance=_required(set_table, "min_distance", "set"),
        max_distance=_required(set_table, "max_distance", "set"),
        generation_attempts=set_table.get("generation_attempts", 100),
    )
    return PrefixReadoutConfig(
        schema_version=_required(root, "schema_version", "root"),
        name=_required(root, "name", "root"),
        checkpoint_sha256=_required(root, "checkpoint_sha256", "root"),
        baseline_sampling=_sampling_config(
            _required(root, "baseline_sampling", "root"), "baseline_sampling"
        ),
        readout_sampling=_sampling_config(
            _required(root, "readout_sampling", "root"), "readout_sampling"
        ),
        problem=DeclaredPrefixReadoutProblem(
            problem_config=problem_config,
            example_count=_required(set_table, "example_count", "set"),
            generation_seed=_required(set_table, "generation_seed", "set"),
        ),
        output_root=output_root.resolve(),
        runtime=RuntimeConfig(
            device=runtime_table.get("device", "auto"),
            dtype=runtime_table.get("dtype", "bfloat16"),
            allow_dirty_source=runtime_table.get("allow_dirty_source", False),
        ),
    )


@dataclass(frozen=True)
class ReasoningPrefixBoundary:
    ordinal: int
    reasoning_token_count: int
    walk_index: int
    follows_restart: bool
    prefix_tokens: tuple[int, ...]


def eligible_reasoning_prefixes(
    completion: Sequence[int],
    vocabulary: Vocabulary,
    minimum_reason_tokens: int,
) -> tuple[ReasoningPrefixBoundary, ...]:
    """Return every valid node-ending boundary from a parsed baseline trace."""

    if minimum_reason_tokens < 1:
        raise ValueError("minimum_reason_tokens must be positive")
    parsed = parse_completion(completion, vocabulary, minimum_reason_tokens)
    if not parsed.format_ok:
        return ()
    boundaries: list[ReasoningPrefixBoundary] = []
    walk_index = 0
    follows_restart = False
    for index, token in enumerate(parsed.reasoning_tokens):
        if token == JUMP:
            walk_index += 1
            follows_restart = True
            continue
        if vocabulary.token_node(token) is None:
            raise AssertionError("parsed reasoning unexpectedly contains a non-node token")
        token_count = index + 1
        if token_count >= minimum_reason_tokens:
            boundaries.append(
                ReasoningPrefixBoundary(
                    ordinal=len(boundaries),
                    reasoning_token_count=token_count,
                    walk_index=walk_index,
                    follows_restart=follows_restart,
                    prefix_tokens=tuple(parsed.reasoning_tokens[:token_count]),
                )
            )
        follows_restart = False
    return tuple(boundaries)


def validate_checkpoint_digest(path: str | Path, expected_sha256: str) -> str:
    """Hash checkpoint bytes and reject identity mismatches before deserialization."""

    checkpoint_path = Path(path).expanduser().resolve()
    expected = expected_sha256.lower()
    if not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ValueError("expected checkpoint digest must be 64 hexadecimal digits")
    digest = hashlib.sha256()
    with checkpoint_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual != expected:
        raise ValueError(
            f"checkpoint digest mismatch: expected {expected}, observed {actual}"
        )
    return actual


def create_artifact_directory(output_root: str | Path, study_id: str) -> Path:
    """Create one evidence directory without allowing an existing path."""

    root = Path(output_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    directory = root / _nonempty_string(study_id, "study_id")
    directory.mkdir(exist_ok=False)
    return directory


@dataclass(frozen=True)
class PrefixReadoutRequest:
    problem_index: int
    example: GraphExample
    baseline_completion: tuple[int, ...]
    boundary: ReasoningPrefixBoundary
    prompt_tokens: tuple[int, ...]
    generation_budget: int


@dataclass(frozen=True)
class PrefixReadoutResult:
    request: PrefixReadoutRequest
    generated_suffix: tuple[int, ...]
    reconstructed_completion: tuple[int, ...]
    terminated_by_eos: bool
    outcome: OutcomeFacts


def _build_readout_requests(
    baseline: EvaluationResult,
    vocabulary: Vocabulary,
    minimum_reason_tokens: int,
    maximum_context_length: int,
    maximum_new_tokens: int,
) -> tuple[PrefixReadoutRequest, ...]:
    requests = []
    forced = (END_REASON, BEGIN_ANSWER)
    for problem_index, case in enumerate(baseline.cases):
        completion = case.completion.completion
        for boundary in eligible_reasoning_prefixes(
            completion, vocabulary, minimum_reason_tokens
        ):
            prompt = case.completion.example.prompt + boundary.prefix_tokens + forced
            available = maximum_context_length - len(prompt)
            requests.append(
                PrefixReadoutRequest(
                    problem_index=problem_index,
                    example=case.completion.example,
                    baseline_completion=completion,
                    boundary=boundary,
                    prompt_tokens=prompt,
                    generation_budget=max(0, min(maximum_new_tokens, available)),
                )
            )
    return tuple(requests)


def _run_readouts(
    model: Transformer,
    requests: Sequence[PrefixReadoutRequest],
    sampling: EvaluationSamplingConfig,
    vocabulary: Vocabulary,
    minimum_reason_tokens: int,
    *,
    device: torch.device,
    autocast_dtype: torch.dtype | None,
) -> tuple[PrefixReadoutResult, ...]:
    results: list[PrefixReadoutResult] = []
    by_budget: dict[int, list[PrefixReadoutRequest]] = {}
    for request in requests:
        by_budget.setdefault(request.generation_budget, []).append(request)
    for budget in sorted(by_budget):
        group = by_budget[budget]
        if budget == 0:
            generated = ((),) * len(group)
        else:
            generated = generate_tokens(
                model,
                [request.prompt_tokens for request in group],
                max_new_tokens=budget,
                batch_size=sampling.batch_size,
                pad_token=PAD,
                eos_token=EOS,
                device=device,
                select_next=lambda logits: select_evaluation_tokens(logits, sampling),
                autocast_dtype=autocast_dtype,
            )
        for request, suffix in zip(group, generated):
            reconstructed = (
                request.boundary.prefix_tokens
                + (END_REASON, BEGIN_ANSWER)
                + tuple(suffix)
            )
            results.append(
                PrefixReadoutResult(
                    request=request,
                    generated_suffix=tuple(suffix),
                    reconstructed_completion=reconstructed,
                    terminated_by_eos=bool(suffix and suffix[-1] == EOS),
                    outcome=verify_completion(
                        request.example,
                        reconstructed,
                        vocabulary,
                        minimum_reason_tokens,
                    ),
                )
            )
    return tuple(
        sorted(
            results,
            key=lambda result: (
                result.request.problem_index,
                result.request.boundary.ordinal,
            ),
        )
    )


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


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            _plain_data(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )
        + "\n"
    ).encode("utf-8")


def _atomic_write_json(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_bytes(_canonical_json_bytes(value))
        with temporary.open("rb+") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_json_lines(path: Path, records: Sequence[object]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as handle:
            for record in records:
                handle.write(_canonical_json_bytes(record))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "-", value).strip("-.") or "study"


def _baseline_record(
    study_id: str,
    problem_index: int,
    result: EvaluationResult,
    vocabulary: Vocabulary,
) -> dict[str, object]:
    case = result.cases[problem_index]
    boundaries = eligible_reasoning_prefixes(
        case.completion.completion,
        vocabulary,
        result.protocol.minimum_reason_tokens,
    )
    return {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "kind": "ordinary_baseline",
        "study_id": study_id,
        "problem_set_identity": result.problem_set_identity,
        "problem_index": problem_index,
        "problem": case.completion.example,
        "prompt_tokens": case.completion.example.prompt,
        "completion_tokens": case.completion.completion,
        "completion_text": vocabulary.render(case.completion.completion),
        "terminated_by_eos": case.completion.terminated_by_eos,
        "outcome": case.outcome,
        "eligible_boundary_count": len(boundaries),
    }


def _readout_record(
    study_id: str,
    problem_set_identity: str,
    result: PrefixReadoutResult,
    vocabulary: Vocabulary,
) -> dict[str, object]:
    request = result.request
    return {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "kind": "reasoning_prefix_intervention",
        "study_id": study_id,
        "problem_set_identity": problem_set_identity,
        "problem_index": request.problem_index,
        "baseline_completion_tokens": request.baseline_completion,
        "boundary": request.boundary,
        "retained_prefix_text": vocabulary.render(request.boundary.prefix_tokens),
        "forced_tokens": (END_REASON, BEGIN_ANSWER),
        "forced_text": vocabulary.render((END_REASON, BEGIN_ANSWER)),
        "generation_prompt_tokens": request.prompt_tokens,
        "generation_budget": request.generation_budget,
        "generated_suffix_tokens": result.generated_suffix,
        "generated_suffix_text": vocabulary.render(result.generated_suffix),
        "reconstructed_completion_tokens": result.reconstructed_completion,
        "reconstructed_completion_text": vocabulary.render(
            result.reconstructed_completion
        ),
        "termination_condition": (
            "eos"
            if result.terminated_by_eos
            else "no_context_capacity"
            if request.generation_budget == 0
            else "token_budget"
        ),
        "outcome": result.outcome,
    }


def _intervention_summary(
    baseline: EvaluationResult,
    readouts: Sequence[PrefixReadoutResult],
) -> dict[str, object]:
    eligible_problem_indexes = {item.request.problem_index for item in readouts}
    shortest_problem_indexes = {
        item.request.problem_index for item in readouts if item.outcome.shortest
    }
    grouped: dict[int, list[PrefixReadoutResult]] = {}
    for item in readouts:
        grouped.setdefault(item.request.boundary.reasoning_token_count, []).append(item)
    by_reasoning_token_count = []
    for token_count, group in sorted(grouped.items()):
        count = len(group)
        format_successes = sum(item.outcome.format_ok for item in group)
        valid_successes = sum(item.outcome.valid_path for item in group)
        shortest_successes = sum(item.outcome.shortest for item in group)
        by_reasoning_token_count.append(
            {
                "reasoning_token_count": token_count,
                "readout_count": count,
                "format_successes": format_successes,
                "valid_path_successes": valid_successes,
                "shortest_path_successes": shortest_successes,
                "format_success_rate": format_successes / count,
                "valid_path_success_rate": valid_successes / count,
                "shortest_path_success_rate": shortest_successes / count,
            }
        )

    paired_problems = []
    for problem_index, baseline_case in enumerate(baseline.cases):
        problem_readouts = [
            item for item in readouts if item.request.problem_index == problem_index
        ]
        first_shortest = next(
            (item for item in problem_readouts if item.outcome.shortest), None
        )
        paired_problems.append(
            {
                "problem_index": problem_index,
                "baseline_shortest": baseline_case.outcome.shortest,
                "eligible_boundary_count": len(problem_readouts),
                "first_shortest_boundary": (
                    None if first_shortest is None else first_shortest.request.boundary
                ),
            }
        )

    count = len(readouts)
    format_successes = sum(item.outcome.format_ok for item in readouts)
    valid_successes = sum(item.outcome.valid_path for item in readouts)
    shortest_successes = sum(item.outcome.shortest for item in readouts)
    return {
        "readout_count": count,
        "format_successes": format_successes,
        "valid_path_successes": valid_successes,
        "shortest_path_successes": shortest_successes,
        "format_success_rate": format_successes / count if count else None,
        "valid_path_success_rate": valid_successes / count if count else None,
        "shortest_path_success_rate": shortest_successes / count if count else None,
        "problems_with_eligible_boundaries": len(eligible_problem_indexes),
        "problems_with_any_shortest_readout": len(shortest_problem_indexes),
        "problems_without_eligible_boundaries": (
            len(baseline.cases) - len(eligible_problem_indexes)
        ),
        "by_reasoning_token_count": by_reasoning_token_count,
        "paired_problems": paired_problems,
    }


@dataclass(frozen=True)
class PrefixReadoutStudyResult:
    directory: Path
    study_id: str
    baseline: EvaluationResult
    readouts: tuple[PrefixReadoutResult, ...]


def execute_prefix_readout(
    checkpoint_path: str | Path,
    configuration_path: str | Path,
    *,
    source_repository: str | Path = ".",
    command: Sequence[str] | None = None,
) -> PrefixReadoutStudyResult:
    """Run a declared ordinary baseline and its paired forced-prefix readouts."""

    config_path = Path(configuration_path).expanduser().resolve()
    config_bytes = config_path.read_bytes()
    config = load_prefix_readout_config(config_path)
    checkpoint_digest = validate_checkpoint_digest(
        checkpoint_path, config.checkpoint_sha256
    )
    checkpoint = load_training_checkpoint(checkpoint_path)
    if (
        config.problem.problem_config.vertices
        > checkpoint.configuration.vocabulary.node_label_count
    ):
        raise ValueError("declared problem set exceeds the checkpoint vocabulary")
    source = inspect_source_provenance(source_repository)
    if source.dirty and not config.runtime.allow_dirty_source:
        summary = ", ".join(source.status[:5])
        if len(source.status) > 5:
            summary += ", ..."
        raise RuntimeError(
            "source tree is dirty; commit changes or set "
            f"runtime.allow_dirty_source=true ({summary})"
        )

    device = torch.device(resolve_runtime_device(config.runtime.device))
    autocast_dtype = resolve_autocast_dtype(config.runtime.dtype, device)
    policy = Transformer(checkpoint.configuration.model).to(device)
    policy.load_state_dict(checkpoint.payload["policy"])
    policy.eval()
    for parameter in policy.parameters():
        parameter.requires_grad_(False)

    problem_set = generate_evaluation_problem_set(
        config.problem.problem_config,
        checkpoint.configuration.vocabulary,
        config.problem.example_count,
        config.problem.generation_seed,
    )
    protocol = EvaluationProtocol(
        policy_checkpoint=str(checkpoint.path),
        sampling=config.baseline_sampling,
        minimum_reason_tokens=checkpoint.configuration.minimum_reason_tokens,
    )
    baseline = run_evaluation(
        policy,
        protocol,
        problem_set,
        checkpoint.configuration.vocabulary,
        device=device,
        autocast_dtype=autocast_dtype,
    )
    requests = _build_readout_requests(
        baseline,
        checkpoint.configuration.vocabulary,
        checkpoint.configuration.minimum_reason_tokens,
        checkpoint.configuration.model.max_context_length,
        config.readout_sampling.max_new_tokens,
    )
    readouts = _run_readouts(
        policy,
        requests,
        config.readout_sampling,
        checkpoint.configuration.vocabulary,
        checkpoint.configuration.minimum_reason_tokens,
        device=device,
        autocast_dtype=autocast_dtype,
    )

    study_id = str(uuid.uuid4())
    timestamp = datetime.now(timezone.utc)
    directory_name = (
        f"{_slug(config.name)}_{timestamp.strftime('%Y%m%dT%H%M%SZ')}_{study_id[:8]}"
    )
    directory = create_artifact_directory(config.output_root, directory_name)
    config_digest = hashlib.sha256(config_bytes).hexdigest()
    resolved_config = _plain_data(config)
    assert isinstance(resolved_config, dict)
    _atomic_write_json(directory / "resolved-config.json", resolved_config)
    _atomic_write_json(
        directory / "provenance.json",
        {
            "schema_version": EVIDENCE_SCHEMA_VERSION,
            "study_id": study_id,
            "study_name": config.name,
            "started_at_utc": timestamp.isoformat().replace("+00:00", "Z"),
            "command": tuple(command) if command is not None else tuple(sys.argv),
            "configuration_source": str(config_path),
            "configuration_source_sha256": config_digest,
            "resolved_configuration_sha256": hashlib.sha256(
                _canonical_json_bytes(resolved_config)
            ).hexdigest(),
            "checkpoint": str(checkpoint.path),
            "checkpoint_sha256": checkpoint_digest,
            "checkpoint_run_id": checkpoint.run_id,
            "checkpoint_step": checkpoint.step,
            "checkpoint_purpose": checkpoint.purpose,
            "checkpoint_configuration_sha256": checkpoint.configuration_sha256,
            "problem_set_identity": problem_set.identity,
            "source_revision": source.revision,
            "source_dirty": source.dirty,
            "source_status": source.status,
            "source_patch_sha256": source.patch_sha256,
            "python_version": platform.python_version(),
            "torch_version": str(torch.__version__),
            "cuda_version": torch.version.cuda,
            "requested_device": config.runtime.device,
            "device": str(device),
            "dtype": config.runtime.dtype,
            "policy_mode": "evaluation",
            "policy_parameters_trainable": False,
            "optimizer_loaded": False,
            "reference_policy_loaded": False,
            "curriculum_state_applied": False,
        },
    )
    if source.patch:
        (directory / "source.patch").write_bytes(source.patch)
    _write_json_lines(
        directory / "baselines.jsonl",
        tuple(
            _baseline_record(
                study_id, index, baseline, checkpoint.configuration.vocabulary
            )
            for index in range(len(baseline.cases))
        ),
    )
    _write_json_lines(
        directory / "readouts.jsonl",
        tuple(
            _readout_record(
                study_id,
                baseline.problem_set_identity,
                readout,
                checkpoint.configuration.vocabulary,
            )
            for readout in readouts
        ),
    )
    _atomic_write_json(
        directory / "summary.json",
        {
            "schema_version": EVIDENCE_SCHEMA_VERSION,
            "study_id": study_id,
            "study_name": config.name,
            "problem_set_identity": baseline.problem_set_identity,
            "ordinary_baseline": baseline.metrics,
            "intervention": _intervention_summary(baseline, readouts),
            "interpretation": {
                "ordinary_and_intervened_results_are_separate": True,
                "interventions_are_not_unmodified_policy_performance": True,
            },
        },
    )
    return PrefixReadoutStudyResult(directory, study_id, baseline, readouts)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a TOML-declared reasoning-prefix intervention study."
    )
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("configuration", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    result = execute_prefix_readout(arguments.checkpoint, arguments.configuration)
    print(result.directory)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
