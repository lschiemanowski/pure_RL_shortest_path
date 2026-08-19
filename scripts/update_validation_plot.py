#!/usr/bin/env python3
"""Render scheduled validation success rates from a training run's evidence."""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from io import BytesIO
import json
import os
from pathlib import Path
import tempfile


os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "pure-rl-matplotlib")
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

plt.rcParams["svg.fonttype"] = "none"


@dataclass(frozen=True)
class ValidationPoint:
    step: int
    frontier: int | None
    example_count: int
    format_rate: float
    valid_rate: float
    shortest_rate: float
    stage_label: str = ""


@dataclass(frozen=True)
class CurriculumTransition:
    step: int
    completed_frontier: int
    new_frontier: int


def _rate(metrics: dict[str, object], rate_key: str, count_key: str) -> float:
    explicit = metrics.get(rate_key)
    if isinstance(explicit, int | float):
        return float(explicit)
    count = metrics.get(count_key)
    denominator = metrics.get("example_count")
    if not isinstance(count, int | float) or not isinstance(denominator, int):
        raise ValueError(f"evaluation metrics do not contain {rate_key}")
    if denominator <= 0:
        raise ValueError("evaluation example_count must be positive")
    return float(count) / denominator


def _stage_label(record: dict[str, object]) -> str:
    config = record.get("problem_config")
    if not isinstance(config, dict):
        return ""
    vertices = config.get("vertices")
    edges = config.get("edges")
    minimum = config.get("min_distance")
    maximum = config.get("max_distance")
    if not all(isinstance(value, int) for value in (vertices, edges, minimum, maximum)):
        return ""
    distance = str(minimum) if minimum == maximum else f"{minimum}–{maximum}"
    return f"n{vertices}, e{edges}, d{distance}"


def _json_records(metrics_path: Path) -> Iterable[dict[str, object]]:
    if not metrics_path.is_file():
        return
    for line in metrics_path.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict):
            yield record


def load_validation_points(metrics_path: Path) -> tuple[ValidationPoint, ...]:
    """Read complete evaluation records while tolerating a partial appended line."""

    points: list[ValidationPoint] = []
    for record in _json_records(metrics_path):
        if record.get("kind") != "evaluation":
            continue
        metrics = record.get("metrics")
        if not isinstance(metrics, dict):
            continue
        step = record.get("step")
        example_count = metrics.get("example_count")
        frontier = record.get("frontier")
        if not isinstance(step, int) or not isinstance(example_count, int):
            continue
        if frontier is not None and not isinstance(frontier, int):
            frontier = None
        points.append(
            ValidationPoint(
                step=step,
                frontier=frontier,
                example_count=example_count,
                format_rate=_rate(
                    metrics, "format_success_rate", "format_successes"
                ),
                valid_rate=_rate(
                    metrics, "valid_path_success_rate", "valid_path_successes"
                ),
                shortest_rate=_rate(
                    metrics,
                    "shortest_path_success_rate",
                    "shortest_path_successes",
                ),
                stage_label=_stage_label(record),
            )
        )
    return tuple(sorted(points, key=lambda point: point.step))


def load_curriculum_transitions(
    metrics_path: Path,
) -> tuple[CurriculumTransition, ...]:
    transitions: list[CurriculumTransition] = []
    for record in _json_records(metrics_path):
        transition = record.get("transition")
        if record.get("kind") != "curriculum_validation" or not isinstance(
            transition, dict
        ):
            continue
        step = transition.get("training_step")
        completed = transition.get("completed_frontier")
        new = transition.get("new_frontier")
        if all(isinstance(value, int) for value in (step, completed, new)):
            transitions.append(CurriculumTransition(step, completed, new))
    return tuple(sorted(transitions, key=lambda item: item.step))


def select_latest_run(runs_root: Path, experiment_name: str) -> Path:
    candidates = [
        path
        for path in runs_root.glob(f"{experiment_name}_*")
        if path.is_dir() and (path / "metrics.jsonl").exists()
    ]
    if not candidates:
        raise FileNotFoundError(
            f"no run matching {experiment_name!r} exists below {runs_root}"
        )
    return max(candidates, key=lambda path: path.stat().st_mtime_ns)


def run_lineage(run_directory: Path) -> tuple[Path, ...]:
    """Return available parent runs followed by the selected derived run."""

    newest_to_oldest: list[Path] = []
    seen: set[Path] = set()
    current = run_directory.resolve()
    while current not in seen:
        seen.add(current)
        newest_to_oldest.append(current)
        provenance_path = current / "provenance.json"
        if not provenance_path.is_file():
            break
        try:
            provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            break
        parent_checkpoint = provenance.get("parent_checkpoint")
        if not isinstance(parent_checkpoint, str):
            break
        parent = Path(parent_checkpoint).resolve().parent.parent
        if not (parent / "metrics.jsonl").is_file():
            break
        current = parent
    return tuple(reversed(newest_to_oldest))


def load_lineage_validation_points(
    run_directories: Iterable[Path],
) -> tuple[ValidationPoint, ...]:
    by_step: dict[int, ValidationPoint] = {}
    for run_directory in run_directories:
        for point in load_validation_points(run_directory / "metrics.jsonl"):
            by_step[point.step] = point
    return tuple(by_step[step] for step in sorted(by_step))


def load_lineage_transitions(
    run_directories: Iterable[Path],
) -> tuple[CurriculumTransition, ...]:
    by_step: dict[int, CurriculumTransition] = {}
    for run_directory in run_directories:
        for transition in load_curriculum_transitions(
            run_directory / "metrics.jsonl"
        ):
            by_step[transition.step] = transition
    return tuple(by_step[step] for step in sorted(by_step))


def _figure(
    points: tuple[ValidationPoint, ...],
    transitions: tuple[CurriculumTransition, ...],
    *,
    run_name: str,
    updated_at: datetime,
) -> plt.Figure:
    fig, axis = plt.subplots(figsize=(14, 7.5))
    fig.subplots_adjust(left=0.08, right=0.80, top=0.84, bottom=0.11)
    stages = sorted(
        {point.frontier for point in points if point.frontier is not None}
    )
    color_map = plt.get_cmap("tab10")
    colors = {stage: color_map(stage % 10) for stage in stages}
    metrics = (
        ("Shortest path", "shortest_rate", "-", "o", 2.8),
        ("Valid path", "valid_rate", "--", "x", 2.1),
        ("Format", "format_rate", ":", "^", 1.9),
    )

    for stage in stages:
        stage_points = [point for point in points if point.frontier == stage]
        steps = [point.step for point in stage_points]
        color = colors[stage]
        for _, attribute, line_style, marker, width in metrics:
            values = [100.0 * getattr(point, attribute) for point in stage_points]
            axis.plot(
                steps,
                values,
                color=color,
                linewidth=width,
                linestyle=line_style,
                solid_capstyle="round",
            )
            axis.scatter(
                steps,
                values,
                s=30 if marker != "x" else 35,
                marker=marker,
                color=color,
                alpha=0.30,
                linewidths=0.9 if marker == "x" else 0,
                zorder=3,
            )

    for transition in transitions:
        color = colors.get(transition.new_frontier, "#555555")
        axis.axvline(
            transition.step,
            color=color,
            linewidth=1.2,
            alpha=0.38,
        )
        axis.annotate(
            f"Stage {transition.completed_frontier} → {transition.new_frontier}",
            xy=(transition.step, 100),
            xytext=(8, -8),
            textcoords="offset points",
            ha="left",
            va="top",
            fontsize=9,
            color=color,
        )

    stage_handles: list[Line2D] = []
    for stage in stages:
        label = next(
            (point.stage_label for point in points if point.frontier == stage), ""
        )
        suffix = f": {label}" if label else ""
        stage_handles.append(
            Line2D(
                [0],
                [0],
                color=colors[stage],
                linewidth=3,
                label=f"Stage {stage}{suffix}",
            )
        )
    metric_handles = [
        Line2D(
            [0],
            [0],
            color="black",
            linewidth=width,
            linestyle=line_style,
            marker=marker,
            markersize=4,
            label=label,
        )
        for label, _, line_style, marker, width in metrics
    ]
    if stage_handles:
        stage_legend = axis.legend(
            handles=stage_handles,
            title="Curriculum stage",
            loc="upper left",
            framealpha=0.94,
            fontsize=9,
        )
        axis.add_artist(stage_legend)
    axis.legend(
        handles=metric_handles,
        title="Validation metric",
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        framealpha=0.94,
        fontsize=9,
    )

    axis.set_title("Deep GRPO validation performance", fontsize=15, pad=30)
    axis.text(
        0.5,
        1.008,
        (
            "Exact 256-example validations every 100 steps; "
            "metric lines are segmented at curriculum transitions"
        ),
        transform=axis.transAxes,
        ha="center",
        va="bottom",
        fontsize=9.5,
        color="#555555",
    )
    axis.set_xlabel("Global training step")
    axis.set_ylabel("Validation performance (%)")
    axis.set_ylim(-2, 102)
    axis.set_yticks(range(0, 101, 10))
    axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{int(value):,}"))
    axis.grid(axis="y", color="#c7c7c7", alpha=0.5, linewidth=0.7)
    axis.grid(axis="x", color="#dddddd", alpha=0.25, linewidth=0.5)
    axis.spines[["top", "right"]].set_visible(False)
    if not points:
        axis.text(
            0.5,
            0.5,
            "Waiting for the first scheduled validation record",
            transform=axis.transAxes,
            ha="center",
            va="center",
            fontsize=11,
            color="#666666",
        )
    return fig


def _render(
    points: tuple[ValidationPoint, ...],
    transitions: tuple[CurriculumTransition, ...],
    *,
    run_name: str,
    updated_at: datetime,
    output_format: str,
) -> bytes:
    figure = _figure(
        points,
        transitions,
        run_name=run_name,
        updated_at=updated_at,
    )
    output = BytesIO()
    figure.savefig(output, format=output_format, dpi=180, facecolor="white")
    plt.close(figure)
    return output.getvalue()


def render_png(
    points: tuple[ValidationPoint, ...],
    *,
    run_name: str,
    updated_at: datetime,
    transitions: tuple[CurriculumTransition, ...] = (),
) -> bytes:
    return _render(
        points,
        transitions,
        run_name=run_name,
        updated_at=updated_at,
        output_format="png",
    )


def render_svg(
    points: tuple[ValidationPoint, ...],
    *,
    run_name: str,
    updated_at: datetime,
    transitions: tuple[CurriculumTransition, ...] = (),
) -> bytes:
    return _render(
        points,
        transitions,
        run_name=run_name,
        updated_at=updated_at,
        output_format="svg",
    )


def write_bytes_atomically(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--png-output", type=Path)
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    run_directory = select_latest_run(
        arguments.runs_root, arguments.experiment_name
    )
    lineage = run_lineage(run_directory)
    points = load_lineage_validation_points(lineage)
    transitions = load_lineage_transitions(lineage)
    updated_at = datetime.now(timezone.utc)
    write_bytes_atomically(
        arguments.output,
        render_svg(
            points,
            transitions=transitions,
            run_name=run_directory.name,
            updated_at=updated_at,
        ),
    )
    if arguments.png_output is not None:
        write_bytes_atomically(
            arguments.png_output,
            render_png(
                points,
                transitions=transitions,
                run_name=run_directory.name,
                updated_at=updated_at,
            ),
        )
    print(
        json.dumps(
            {
                "evaluation_count": len(points),
                "lineage_run_count": len(lineage),
                "output": str(arguments.output),
                "png_output": (
                    str(arguments.png_output)
                    if arguments.png_output is not None
                    else None
                ),
                "run_directory": str(run_directory),
                "transition_count": len(transitions),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
