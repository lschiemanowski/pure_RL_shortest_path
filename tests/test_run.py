from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

from pure_rl_shortest_path.run import (
    CONFIG_SCHEMA_VERSION,
    ConfigurationError,
    LoadedRunConfiguration,
    derive_named_seeds,
    initialize_run,
    load_run_configuration,
)


BASE_CONFIG = """\
schema_version = 1
name = "small-test"
seed = 17

[task]
node_label_count = 12

[model]
max_context_length = 128
d_model = 32
layers = 2
heads = 4
mlp_dim = 64

[rollout]
group_size = 4
max_new_tokens = 32
temperature = 1.0
top_p = 1.0

[reward]
coverage_coefficient = 0.15

[optimization]
learning_rate = 3e-4
valid_coefficient = 1.0

[training]
max_steps = 100
problems_per_step = 3
reference_update_every = 10

[evaluation]
every_steps = 20
example_count = 8
max_new_tokens = 32
mode = "greedy"

[curriculum]
past_decay_scale = 2.0
future_decay_scale = 0.25
advancement_threshold = 0.8
advancement_patience = 2

[[curriculum.stages]]
vertices = 4
edges = 3
min_distance = 1
max_distance = 2

[[curriculum.stages]]
vertices = 8
edges = 10
min_distance = 2
max_distance = 4

[artifacts]
output_root = "../runs"
log_every = 5
checkpoint_every = 20
representative_every = 20

[runtime]
device = "cpu"
dtype = "float32"
allow_dirty_source = false
"""


def run_git(repository: Path, *arguments: str) -> None:
    subprocess.run(
        ("git", *arguments),
        cwd=repository,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


class RunConfigurationTests(unittest.TestCase):
    def write_config(self, directory: Path, text: str = BASE_CONFIG) -> Path:
        path = directory / "experiment.toml"
        path.write_text(text, encoding="utf-8")
        return path

    def test_toml_resolves_to_existing_typed_contracts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = self.write_config(root)
            loaded = load_run_configuration(path)

            self.assertEqual(loaded.resolved.schema_version, CONFIG_SCHEMA_VERSION)
            self.assertEqual(loaded.resolved.vocabulary.node_label_count, 12)
            self.assertEqual(loaded.resolved.model.vocab_size, 22)
            self.assertEqual(loaded.resolved.model.n_layers, 2)
            self.assertEqual(loaded.resolved.rollout.temperature, 1.0)
            self.assertEqual(loaded.resolved.grpo.group_size, 4)
            self.assertEqual(loaded.resolved.grpo.valid_coefficient, 1.0)
            self.assertEqual(len(loaded.resolved.curriculum.stages), 2)
            self.assertEqual(loaded.resolved.evaluation.sampling.mode, "greedy")
            self.assertEqual(
                loaded.resolved.artifacts.output_root,
                (root.parent / "runs").resolve(),
            )
            self.assertEqual(
                loaded.source_sha256, hashlib.sha256(BASE_CONFIG.encode()).hexdigest()
            )

            repeated = load_run_configuration(path)
            self.assertEqual(loaded.resolved_sha256, repeated.resolved_sha256)

    def test_unknown_keys_and_missing_required_fields_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            unknown = BASE_CONFIG.replace(
                "[reward]\n", "[reward]\nsecret_shaping = 2.0\n"
            )
            with self.assertRaisesRegex(
                ConfigurationError, "reward.secret_shaping"
            ):
                load_run_configuration(self.write_config(root, unknown))

            missing = BASE_CONFIG.replace("node_label_count = 12\n", "")
            with self.assertRaisesRegex(ConfigurationError, "node_label_count"):
                load_run_configuration(self.write_config(root, missing))

    def test_cross_contract_conflicts_are_rejected_before_a_run(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            too_few_labels = BASE_CONFIG.replace(
                "node_label_count = 12", "node_label_count = 6"
            )
            with self.assertRaisesRegex(ValueError, "cover every curriculum stage"):
                load_run_configuration(self.write_config(root, too_few_labels))

            truncated_training = BASE_CONFIG.replace("top_p = 1.0", "top_p = 0.9")
            with self.assertRaisesRegex(ValueError, "temperature=1 and top_p=1"):
                load_run_configuration(self.write_config(root, truncated_training))

            short_context = BASE_CONFIG.replace(
                "max_context_length = 128", "max_context_length = 40"
            )
            with self.assertRaisesRegex(ValueError, "completion budget"):
                load_run_configuration(self.write_config(root, short_context))

    def test_named_seed_derivation_is_stable_distinct_and_named(self) -> None:
        first = derive_named_seeds(17)
        second = derive_named_seeds(17)
        different = derive_named_seeds(18)

        self.assertEqual(first, second)
        self.assertEqual(len(first), 6)
        self.assertEqual(len(set(first.values())), len(first))
        self.assertNotEqual(first, different)
        self.assertIn("training_problems", first)
        self.assertIn("evaluation_problems", first)


class RunInitializationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.repository = self.root / "repository"
        self.repository.mkdir()
        run_git(self.repository, "init", "-q")
        run_git(self.repository, "config", "user.name", "Test Researcher")
        run_git(self.repository, "config", "user.email", "test@example.com")
        self.config_path = self.repository / "experiment.toml"
        self.config_path.write_text(BASE_CONFIG, encoding="utf-8")
        (self.repository / "source.py").write_text("VALUE = 1\n", encoding="utf-8")
        run_git(self.repository, "add", "experiment.toml", "source.py")
        run_git(self.repository, "commit", "-q", "-m", "initial")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def initialize(self, loaded: LoadedRunConfiguration | None = None):
        return initialize_run(
            loaded or load_run_configuration(self.config_path),
            source_repository=self.repository,
            command=("python", "-m", "pure_rl_shortest_path", "train"),
            started_at=datetime(2026, 8, 12, 8, 30, tzinfo=timezone.utc),
            run_id="12345678-1234-4234-9234-123456789abc",
        )

    def test_clean_initialization_records_exact_config_and_provenance(self) -> None:
        loaded = load_run_configuration(self.config_path)
        initialized = self.initialize(loaded)

        self.assertEqual(
            initialized.directory.name,
            "small-test_20260812T083000Z_12345678",
        )
        self.assertEqual(
            (initialized.directory / "experiment.toml").read_bytes(),
            loaded.source_bytes,
        )
        provenance = json.loads(
            (initialized.directory / "provenance.json").read_text()
        )
        self.assertEqual(
            provenance["resolved_configuration_sha256"], loaded.resolved_sha256
        )
        self.assertEqual(
            provenance["command"],
            ["python", "-m", "pure_rl_shortest_path", "train"],
        )
        self.assertFalse(provenance["source_dirty"])
        self.assertEqual(len(provenance["source_revision"]), 40)
        self.assertFalse((initialized.directory / "source.patch").exists())

        with self.assertRaises(FileExistsError):
            self.initialize(loaded)

    def test_dirty_source_is_rejected_by_default(self) -> None:
        loaded = load_run_configuration(self.config_path)
        (self.repository / "source.py").write_text("VALUE = 2\n", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "source tree is dirty"):
            self.initialize(loaded)
        self.assertFalse((self.root / "runs").exists())

    def test_explicit_dirty_run_preserves_patch_and_status(self) -> None:
        allowed_text = BASE_CONFIG.replace(
            "allow_dirty_source = false", "allow_dirty_source = true"
        )
        self.config_path.write_text(allowed_text, encoding="utf-8")
        run_git(self.repository, "add", "experiment.toml")
        run_git(self.repository, "commit", "-q", "-m", "allow dirty source")
        allowed = load_run_configuration(self.config_path)
        (self.repository / "source.py").write_text("VALUE = 2\n", encoding="utf-8")

        initialized = self.initialize(allowed)
        patch = (initialized.directory / "source.patch").read_text()
        provenance = json.loads(
            (initialized.directory / "provenance.json").read_text()
        )

        self.assertIn("-VALUE = 1", patch)
        self.assertIn("+VALUE = 2", patch)
        self.assertTrue(provenance["source_dirty"])
        self.assertTrue(provenance["source_status"])
        self.assertEqual(
            provenance["source_patch_sha256"],
            hashlib.sha256(patch.encode()).hexdigest(),
        )


if __name__ == "__main__":
    unittest.main()
