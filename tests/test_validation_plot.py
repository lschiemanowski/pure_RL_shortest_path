from __future__ import annotations

from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


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
            self.assertIn("Format", svg)
            self.assertIn("Valid path", svg)
            self.assertIn("Shortest path", svg)
            self.assertIn('class="line series-format"', svg)
            self.assertIn('class="line series-valid"', svg)
            self.assertIn('class="line series-shortest"', svg)


if __name__ == "__main__":
    unittest.main()
