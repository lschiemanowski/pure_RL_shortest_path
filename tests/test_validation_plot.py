from __future__ import annotations

from datetime import datetime, timezone
from io import BytesIO
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

from PIL import Image


SCRIPT = Path(__file__).parents[1] / "scripts" / "update_validation_plot.py"
SPEC = importlib.util.spec_from_file_location("update_validation_plot", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
plot = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = plot
SPEC.loader.exec_module(plot)


class ValidationPlotTests(unittest.TestCase):
    def test_loads_nested_evaluation_metrics_and_skips_partial_line(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            metrics = Path(temporary) / "metrics.jsonl"
            records = [
                {"kind": "training", "step": 100},
                {
                    "kind": "evaluation",
                    "step": 100,
                    "frontier": 0,
                    "metrics": {
                        "example_count": 256,
                        "format_successes": 128,
                        "valid_path_successes": 64,
                        "shortest_path_successes": 32,
                    },
                },
            ]
            metrics.write_text(
                "\n".join(json.dumps(record) for record in records)
                + "\n{partial",
                encoding="utf-8",
            )

            points = plot.load_validation_points(metrics)

            self.assertEqual(len(points), 1)
            self.assertEqual(points[0].step, 100)
            self.assertEqual(points[0].frontier, 0)
            self.assertEqual(points[0].format_rate, 0.5)
            self.assertEqual(points[0].valid_rate, 0.25)
            self.assertEqual(points[0].shortest_rate, 0.125)

    def test_selects_latest_run_and_renders_all_three_series(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "experiment_20260819T000000Z_old"
            second = root / "experiment_20260819T010000Z_new"
            first.mkdir()
            second.mkdir()
            (first / "metrics.jsonl").write_text("", encoding="utf-8")
            (second / "metrics.jsonl").write_text("", encoding="utf-8")
            first.touch()
            second.touch()

            selected = plot.select_latest_run(root, "experiment")
            svg = plot.render_svg(
                (
                    plot.ValidationPoint(100, 0, 256, 0.5, 0.25, 0.125),
                    plot.ValidationPoint(200, 0, 256, 0.75, 0.5, 0.25),
                ),
                run_name=selected.name,
                updated_at=datetime(2026, 8, 19, tzinfo=timezone.utc),
            )

            self.assertEqual(selected, second)
            svg_text = svg.decode("utf-8")
            self.assertIn("Deep GRPO validation performance", svg_text)
            self.assertIn("Format", svg_text)
            self.assertIn("Valid path", svg_text)
            self.assertIn("Shortest path", svg_text)

            png = plot.render_png(
                (
                    plot.ValidationPoint(100, 0, 256, 0.5, 0.25, 0.125),
                    plot.ValidationPoint(200, 0, 256, 0.75, 0.5, 0.25),
                ),
                run_name=selected.name,
                updated_at=datetime(2026, 8, 19, tzinfo=timezone.utc),
            )
            with Image.open(BytesIO(png)) as image:
                self.assertEqual(image.format, "PNG")
                self.assertEqual(image.size, (2520, 1350))

    def test_loads_transition_and_segments_metric_lines_by_stage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            metrics = Path(temporary) / "metrics.jsonl"
            metrics.write_text(
                json.dumps(
                    {
                        "kind": "curriculum_validation",
                        "step": 200,
                        "transition": {
                            "training_step": 200,
                            "completed_frontier": 0,
                            "new_frontier": 1,
                        },
                    }
                ),
                encoding="utf-8",
            )
            transitions = plot.load_curriculum_transitions(metrics)
            self.assertEqual(
                transitions, (plot.CurriculumTransition(200, 0, 1),)
            )

            points = (
                plot.ValidationPoint(100, 0, 256, 0.5, 0.25, 0.125),
                plot.ValidationPoint(200, 0, 256, 0.8, 0.75, 0.7),
                plot.ValidationPoint(300, 1, 256, 0.4, 0.1, 0.05),
                plot.ValidationPoint(400, 1, 256, 0.6, 0.3, 0.2),
            )
            figure = plot._figure(
                points,
                transitions,
                run_name="test-run",
                updated_at=datetime(2026, 8, 19, tzinfo=timezone.utc),
            )
            line_steps = [tuple(line.get_xdata()) for line in figure.axes[0].lines]
            plot.plt.close(figure)

            self.assertIn((100, 200), line_steps)
            self.assertIn((300, 400), line_steps)
            self.assertNotIn((200, 300), line_steps)

    def test_smooths_each_stage_over_five_faint_raw_observations(self) -> None:
        points = (
            plot.ValidationPoint(100, 0, 256, 0.5, 0.3, 0.1),
            plot.ValidationPoint(200, 0, 256, 0.7, 0.5, 0.3),
            plot.ValidationPoint(300, 0, 256, 0.9, 0.7, 0.5),
        )
        figure = plot._figure(
            points,
            (),
            run_name="test-run",
            updated_at=datetime(2026, 8, 19, tzinfo=timezone.utc),
        )
        axis = figure.axes[0]
        shortest_line = next(
            line for line in axis.lines if line.get_linestyle() == "-"
        )
        raw_alphas = [collection.get_alpha() for collection in axis.collections]
        plot.plt.close(figure)

        self.assertEqual(tuple(shortest_line.get_xdata()), (100, 200, 300))
        self.assertEqual(tuple(shortest_line.get_ydata()), (10.0, 20.0, 30.0))
        self.assertEqual(len(raw_alphas), 3)
        self.assertTrue(
            all(alpha is not None and alpha <= 0.16 for alpha in raw_alphas)
        )

    def test_uses_thousand_step_ticks_and_labels_only_ten_thousand_steps(self) -> None:
        figure = plot._figure(
            (
                plot.ValidationPoint(0, 0, 256, 0.5, 0.25, 0.125),
                plot.ValidationPoint(25_000, 0, 256, 0.75, 0.5, 0.25),
            ),
            (),
            run_name="test-run",
            updated_at=datetime(2026, 8, 19, tzinfo=timezone.utc),
        )
        axis = figure.axes[0]
        major_locations = axis.xaxis.get_major_locator().tick_values(0, 25_000)
        minor_locations = axis.xaxis.get_minor_locator().tick_values(0, 25_000)
        major_formatter = axis.xaxis.get_major_formatter()
        plot.plt.close(figure)

        self.assertTrue(
            all(
                right - left == 10_000
                for left, right in zip(major_locations, major_locations[1:])
            )
        )
        self.assertTrue(
            all(
                right - left == 1_000
                for left, right in zip(minor_locations, minor_locations[1:])
            )
        )
        self.assertEqual(major_formatter(10_000, 0), "10,000")

    def test_follows_parent_checkpoint_lineage_and_keeps_parent_validation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            parent = root / "experiment_parent"
            child = root / "experiment_child"
            (parent / "checkpoints").mkdir(parents=True)
            child.mkdir()
            checkpoint = parent / "checkpoints" / "interruption.pt"
            checkpoint.write_bytes(b"checkpoint")
            (parent / "metrics.jsonl").write_text(
                json.dumps(
                    {
                        "kind": "evaluation",
                        "step": 100,
                        "frontier": 0,
                        "metrics": {
                            "example_count": 256,
                            "format_success_rate": 0.1,
                            "valid_path_success_rate": 0.05,
                            "shortest_path_success_rate": 0.025,
                        },
                    }
                ),
                encoding="utf-8",
            )
            (child / "metrics.jsonl").write_text(
                json.dumps(
                    {
                        "kind": "evaluation",
                        "step": 200,
                        "frontier": 0,
                        "metrics": {
                            "example_count": 256,
                            "format_success_rate": 0.2,
                            "valid_path_success_rate": 0.1,
                            "shortest_path_success_rate": 0.05,
                        },
                    }
                ),
                encoding="utf-8",
            )
            (child / "provenance.json").write_text(
                json.dumps({"parent_checkpoint": str(checkpoint)}),
                encoding="utf-8",
            )

            lineage = plot.run_lineage(child)
            points = plot.load_lineage_validation_points(lineage)

            self.assertEqual(lineage, (parent.resolve(), child.resolve()))
            self.assertEqual([point.step for point in points], [100, 200])


if __name__ == "__main__":
    unittest.main()
