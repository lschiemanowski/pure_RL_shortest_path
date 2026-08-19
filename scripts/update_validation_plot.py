#!/usr/bin/env python3
"""Render scheduled validation success rates from a training run's evidence."""

from __future__ import annotations

import argparse
from io import BytesIO
from dataclasses import dataclass
from datetime import datetime, timezone
from html import escape
import json
import os
from pathlib import Path
import tempfile
from collections.abc import Callable, Iterable

from PIL import Image, ImageDraw, ImageFont


@dataclass(frozen=True)
class ValidationPoint:
    step: int
    frontier: int | None
    example_count: int
    format_rate: float
    valid_rate: float
    shortest_rate: float


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


def load_validation_points(metrics_path: Path) -> tuple[ValidationPoint, ...]:
    """Read complete evaluation records while tolerating a partial appended line."""

    points: list[ValidationPoint] = []
    if not metrics_path.is_file():
        return ()
    for line in metrics_path.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
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
            )
        )
    return tuple(sorted(points, key=lambda point: point.step))


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


def _polyline(
    points: Iterable[ValidationPoint],
    attribute: str,
    *,
    x_position: Callable[[int], float],
    y_position: Callable[[float], float],
) -> str:
    coordinates = " ".join(
        f"{x_position(point.step):.1f},{y_position(getattr(point, attribute)):.1f}"
        for point in points
    )
    return coordinates


def render_svg(
    points: tuple[ValidationPoint, ...],
    *,
    run_name: str,
    updated_at: datetime,
) -> str:
    width, height = 1200, 700
    left, right, top, bottom = 100, 48, 130, 90
    plot_width = width - left - right
    plot_height = height - top - bottom
    maximum_step = max((point.step for point in points), default=100)
    maximum_step = max(100, maximum_step)

    def x_position(step: int) -> float:
        return left + plot_width * step / maximum_step

    def y_position(rate: float) -> float:
        return top + plot_height * (1.0 - max(0.0, min(1.0, rate)))

    y_ticks = range(0, 101, 20)
    x_ticks = sorted({round(maximum_step * index / 5) for index in range(6)})
    series = (
        ("Format", "format_rate", "series-format", "circle"),
        ("Valid path", "valid_rate", "series-valid", "square"),
        ("Shortest path", "shortest_rate", "series-shortest", "diamond"),
    )
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="700" '
        'viewBox="0 0 1200 700" role="img" '
        'aria-labelledby="plot-title plot-description">',
        "<style>",
        ":root{color-scheme:light dark;--bg:#ffffff;--fg:#18212b;"
        "--muted:#667085;--grid:#d0d5dd;--frame:#98a2b3;"
        "--format:#2563eb;--valid:#d97706;--shortest:#15803d}",
        "@media(prefers-color-scheme:dark){:root{--bg:#111827;--fg:#f3f4f6;"
        "--muted:#cbd5e1;--grid:#374151;--frame:#6b7280;"
        "--format:#60a5fa;--valid:#fbbf24;--shortest:#4ade80}}",
        "text{font-family:ui-sans-serif,system-ui,sans-serif;fill:var(--fg)}",
        ".title{font-size:28px;font-weight:600}.subtitle{font-size:15px;fill:var(--muted)}",
        ".axis{font-size:14px;fill:var(--muted)}.legend{font-size:15px}",
        ".grid{stroke:var(--grid);stroke-width:1}.frame{fill:none;stroke:var(--frame)}",
        ".line{fill:none;stroke-width:3}.series-format{stroke:var(--format);fill:var(--format)}",
        ".series-valid{stroke:var(--valid);fill:var(--valid)}",
        ".series-shortest{stroke:var(--shortest);fill:var(--shortest)}",
        ".point{stroke:var(--bg);stroke-width:2}.empty{font-size:18px;fill:var(--muted)}",
        "</style>",
        '<rect width="1200" height="700" fill="var(--bg)"/>',
        '<title id="plot-title">Scheduled validation success rates</title>',
        '<desc id="plot-description">Format, valid-path, and shortest-path '
        "success percentages by training step.</desc>",
        '<text class="title" x="100" y="42">Scheduled validation success rates</text>',
        f'<text class="subtitle" x="100" y="70">{escape(run_name)} · updated '
        f'{escape(updated_at.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"))}</text>',
    ]
    for value in y_ticks:
        y = y_position(value / 100)
        parts.extend(
            (
                f'<line class="grid" x1="{left}" y1="{y:.1f}" '
                f'x2="{width - right}" y2="{y:.1f}"/>',
                f'<text class="axis" x="{left - 14}" y="{y + 5:.1f}" '
                f'text-anchor="end">{value}%</text>',
            )
        )
    for value in x_ticks:
        x = x_position(value)
        parts.append(
            f'<text class="axis" x="{x:.1f}" y="{height - bottom + 30}" '
            f'text-anchor="middle">{value}</text>'
        )
    parts.extend(
        (
            f'<rect class="frame" x="{left}" y="{top}" width="{plot_width}" '
            f'height="{plot_height}"/>',
            f'<text class="axis" x="{left + plot_width / 2:.1f}" y="{height - 25}" '
            'text-anchor="middle">Training step</text>',
            f'<text class="axis" x="24" y="{top + plot_height / 2:.1f}" '
            f'text-anchor="middle" transform="rotate(-90 24 {top + plot_height / 2:.1f})">'
            "Validation success</text>",
        )
    )
    legend_x = (100, 255, 440)
    for x, (label, _, css_class, _) in zip(legend_x, series, strict=True):
        parts.extend(
            (
                f'<line class="line {css_class}" x1="{x}" y1="100" '
                f'x2="{x + 28}" y2="100"/>',
                f'<text class="legend" x="{x + 38}" y="105">{label}</text>',
            )
        )
    if not points:
        parts.append(
            f'<text class="empty" x="{left + plot_width / 2:.1f}" '
            f'y="{top + plot_height / 2:.1f}" text-anchor="middle">'
            "Waiting for the first scheduled validation record</text>"
        )
    for _, attribute, css_class, marker in series:
        coordinates = _polyline(
            points,
            attribute,
            x_position=x_position,
            y_position=y_position,
        )
        if len(points) > 1:
            parts.append(
                f'<polyline class="line {css_class}" points="{coordinates}"/>'
            )
        for point in points:
            x = x_position(point.step)
            y = y_position(getattr(point, attribute))
            if marker == "circle":
                shape = f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6"/>'
            elif marker == "square":
                shape = f'<rect x="{x - 6:.1f}" y="{y - 6:.1f}" width="12" height="12"/>'
            else:
                shape = (
                    f'<path d="M {x:.1f} {y - 7:.1f} L {x + 7:.1f} {y:.1f} '
                    f'L {x:.1f} {y + 7:.1f} L {x - 7:.1f} {y:.1f} Z"/>'
                )
            parts.append(f'<g class="point {css_class}">{shape}</g>')
    if points:
        last = points[-1]
        parts.append(
            f'<text class="subtitle" x="{width - right}" y="{height - 25}" '
            f'text-anchor="end">latest: step {last.step}, frontier '
            f'{last.frontier if last.frontier is not None else "unknown"}, '
            f'n={last.example_count}</text>'
        )
    parts.append("</svg>\n")
    return "".join(parts)


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    path = Path("/usr/share/fonts/truetype/dejavu") / name
    try:
        return ImageFont.truetype(str(path), size=size)
    except OSError:
        return ImageFont.load_default()


def render_png(
    points: tuple[ValidationPoint, ...],
    *,
    run_name: str,
    updated_at: datetime,
) -> bytes:
    """Render the same validation history as a viewer-compatible PNG."""

    width, height = 1200, 700
    left, right, top, bottom = 100, 48, 130, 90
    plot_width = width - left - right
    plot_height = height - top - bottom
    maximum_step = max(100, max((point.step for point in points), default=100))
    colors = {
        "background": "#ffffff",
        "foreground": "#18212b",
        "muted": "#667085",
        "grid": "#d0d5dd",
        "frame": "#98a2b3",
        "format": "#2563eb",
        "valid": "#d97706",
        "shortest": "#15803d",
    }
    image = Image.new("RGB", (width, height), colors["background"])
    draw = ImageDraw.Draw(image)
    title_font = _font(28, bold=True)
    body_font = _font(15)
    axis_font = _font(14)
    empty_font = _font(18)

    def x_position(step: int) -> float:
        return left + plot_width * step / maximum_step

    def y_position(rate: float) -> float:
        return top + plot_height * (1.0 - max(0.0, min(1.0, rate)))

    draw.text(
        (left, 18),
        "Scheduled validation success rates",
        fill=colors["foreground"],
        font=title_font,
    )
    subtitle = (
        f"{run_name} · updated "
        f"{updated_at.astimezone(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
    )
    draw.text((left, 57), subtitle, fill=colors["muted"], font=body_font)

    for value in range(0, 101, 20):
        y = y_position(value / 100)
        draw.line((left, y, width - right, y), fill=colors["grid"], width=1)
        draw.text(
            (left - 14, y),
            f"{value}%",
            fill=colors["muted"],
            font=axis_font,
            anchor="rm",
        )
    x_ticks = sorted({round(maximum_step * index / 5) for index in range(6)})
    for value in x_ticks:
        draw.text(
            (x_position(value), height - bottom + 30),
            str(value),
            fill=colors["muted"],
            font=axis_font,
            anchor="mm",
        )
    draw.rectangle(
        (left, top, width - right, height - bottom),
        outline=colors["frame"],
        width=1,
    )
    draw.text(
        (left + plot_width / 2, height - 25),
        "Training step",
        fill=colors["muted"],
        font=axis_font,
        anchor="mm",
    )
    vertical_label = Image.new("RGBA", (220, 30), (0, 0, 0, 0))
    vertical_draw = ImageDraw.Draw(vertical_label)
    vertical_draw.text(
        (110, 15),
        "Validation success",
        fill=colors["muted"],
        font=axis_font,
        anchor="mm",
    )
    vertical_label = vertical_label.rotate(90, expand=True)
    image.paste(
        vertical_label,
        (10, round(top + plot_height / 2 - vertical_label.height / 2)),
        vertical_label,
    )

    series = (
        ("Format", "format_rate", colors["format"], "circle"),
        ("Valid path", "valid_rate", colors["valid"], "square"),
        ("Shortest path", "shortest_rate", colors["shortest"], "diamond"),
    )
    for x, (label, _, color, _) in zip((100, 255, 440), series, strict=True):
        draw.line((x, 100, x + 28, 100), fill=color, width=3)
        draw.text((x + 38, 100), label, fill=colors["foreground"], font=body_font, anchor="lm")

    if not points:
        draw.text(
            (left + plot_width / 2, top + plot_height / 2),
            "Waiting for the first scheduled validation record",
            fill=colors["muted"],
            font=empty_font,
            anchor="mm",
        )
    for _, attribute, color, marker in series:
        coordinates = [
            (x_position(point.step), y_position(getattr(point, attribute)))
            for point in points
        ]
        if len(coordinates) > 1:
            draw.line(coordinates, fill=color, width=3, joint="curve")
        for x, y in coordinates:
            if marker == "circle":
                draw.ellipse((x - 6, y - 6, x + 6, y + 6), fill=color, outline=colors["background"], width=2)
            elif marker == "square":
                draw.rectangle((x - 6, y - 6, x + 6, y + 6), fill=color, outline=colors["background"], width=2)
            else:
                draw.polygon(
                    ((x, y - 7), (x + 7, y), (x, y + 7), (x - 7, y)),
                    fill=color,
                    outline=colors["background"],
                )
    if points:
        last = points[-1]
        latest = (
            f"latest: step {last.step}, frontier "
            f"{last.frontier if last.frontier is not None else 'unknown'}, "
            f"n={last.example_count}"
        )
        draw.text(
            (width - right, height - 25),
            latest,
            fill=colors["muted"],
            font=axis_font,
            anchor="rm",
        )
    output = BytesIO()
    image.save(output, format="PNG", optimize=True)
    return output.getvalue()


def write_atomically(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


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
    svg = render_svg(
        points,
        run_name=run_directory.name,
        updated_at=datetime.now(timezone.utc),
    )
    write_atomically(arguments.output, svg)
    if arguments.png_output is not None:
        png = render_png(
            points,
            run_name=run_directory.name,
            updated_at=datetime.now(timezone.utc),
        )
        write_bytes_atomically(arguments.png_output, png)
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
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
