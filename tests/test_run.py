from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

import torch

from pure_rl_shortest_path.run import (
    CONFIG_SCHEMA_VERSION,
    ConfigurationError,
    LoadedRunConfiguration,
    configuration_differences,
    create_random_streams,
    derive_named_seeds,
    execute_standalone_evaluation,
    execute_training,
    initialize_run,
    load_standalone_evaluation_config,
    load_training_checkpoint,
    load_run_configuration,
    validate_resume_configuration,
)


BASE_CONFIG = """\
schema_version = 1
name = "small-test"
seed = 17

[task]
node_label_count = 12
minimum_reason_tokens = 4

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
sterile_repetition_coefficient = 0.0
sterile_repetition_mode = "off_answer"

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


def fast_config() -> str:
    return (
        BASE_CONFIG.replace("max_context_length = 128", "max_context_length = 64")
        .replace("layers = 2", "layers = 1")
        .replace("group_size = 4", "group_size = 2")
        .replace("max_new_tokens = 32", "max_new_tokens = 12")
        .replace(
            "learning_rate = 3e-4\n",
            "learning_rate = 3e-4\nupdate_epochs = 1\nmicrobatch_size = 2\n",
        )
        .replace("max_steps = 100", "max_steps = 1")
        .replace("problems_per_step = 3", "problems_per_step = 1")
        .replace("reference_update_every = 10", "reference_update_every = 1")
        .replace("every_steps = 20", "every_steps = 1")
        .replace("example_count = 8", "example_count = 2")
        .replace("log_every = 5", "log_every = 1")
        .replace("checkpoint_every = 20", "checkpoint_every = 1")
        .replace("representative_every = 20", "representative_every = 1")
    )


def ppo_fast_config() -> str:
    return (
        fast_config()
        .replace('name = "small-test"', 'name = "small-ppo-test"')
        .replace("seed = 17\n", 'seed = 17\nalgorithm = "ppo"\n')
        .replace("group_size = 2", "group_size = 1")
        .replace(
            "[training]\n",
            "[ppo]\n"
            "update_epochs = 1\n"
            "microbatch_size = 1\n"
            "gamma = 0.99\n"
            "gae_lambda = 0.95\n\n"
            "[training]\n",
        )
    )


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
            self.assertEqual(loaded.resolved.minimum_reason_tokens, 4)
            self.assertEqual(loaded.resolved.model.vocab_size, 22)
            self.assertEqual(loaded.resolved.model.n_layers, 2)
            self.assertEqual(loaded.resolved.rollout.temperature, 1.0)
            self.assertEqual(
                loaded.resolved.reward.sterile_repetition_mode, "off_answer"
            )
            self.assertEqual(
                loaded.resolved.reward.sterile_repetition_coefficient, 0.0
            )
            self.assertEqual(loaded.resolved.grpo.group_size, 4)
            self.assertEqual(loaded.resolved.grpo.valid_coefficient, 1.0)
            self.assertEqual(loaded.resolved.algorithm, "grpo")
            self.assertEqual(loaded.resolved.ppo.gae_lambda, 0.95)
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

            invalid_mode = BASE_CONFIG.replace(
                'sterile_repetition_mode = "off_answer"',
                'sterile_repetition_mode = "oracle"',
            )
            with self.assertRaisesRegex(ValueError, "must be all or off_answer"):
                load_run_configuration(self.write_config(root, invalid_mode))

            negative_coefficient = BASE_CONFIG.replace(
                "sterile_repetition_coefficient = 0.0",
                "sterile_repetition_coefficient = -0.1",
            )
            with self.assertRaisesRegex(ValueError, "finite and nonnegative"):
                load_run_configuration(
                    self.write_config(root, negative_coefficient)
                )

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

    def test_rng_seed_override_is_recorded_and_drives_named_seeds(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repository = root / "repository"
            repository.mkdir()
            run_git(repository, "init", "-q")
            run_git(repository, "config", "user.name", "Test Researcher")
            run_git(repository, "config", "user.email", "test@example.com")
            config_path = repository / "experiment.toml"
            config_path.write_text(BASE_CONFIG, encoding="utf-8")
            run_git(repository, "add", "experiment.toml")
            run_git(repository, "commit", "-q", "-m", "initial")

            initialized = initialize_run(
                load_run_configuration(config_path),
                source_repository=repository,
                rng_seed_override=99,
            )

            self.assertEqual(initialized.provenance.rng_seed_override, 99)
            self.assertEqual(initialized.provenance.seeds, derive_named_seeds(99))


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


class CheckpointEvidenceAndLoopTests(unittest.TestCase):
    def load_fast(self, root: Path):
        path = root / "fast.toml"
        path.write_text(fast_config(), encoding="utf-8")
        return load_run_configuration(path)

    def test_split_resume_matches_uninterrupted_training_exactly(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            loaded = self.load_fast(root)
            seeds = derive_named_seeds(loaded.resolved.master_seed)
            uninterrupted = root / "uninterrupted"
            split = root / "split"
            uninterrupted.mkdir()
            split.mkdir()

            full_result = execute_training(
                run_directory=uninterrupted,
                run_id="aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
                configuration=loaded.resolved,
                seeds=seeds,
                max_steps_override=2,
            )
            first_result = execute_training(
                run_directory=split,
                run_id="bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb",
                configuration=loaded.resolved,
                seeds=seeds,
            )
            checkpoint = load_training_checkpoint(first_result.final_checkpoint)
            resumed_result = execute_training(
                run_directory=split,
                run_id="bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb",
                configuration=loaded.resolved,
                seeds=seeds,
                checkpoint=checkpoint,
                max_steps_override=2,
            )

            full = load_training_checkpoint(full_result.final_checkpoint)
            resumed = load_training_checkpoint(resumed_result.final_checkpoint)
            self.assertEqual(full.payload["schema_version"], 2)
            self.assertEqual(full.step, 2)
            self.assertEqual(resumed.step, 2)
            self.assertEqual(full.curriculum_state, resumed.curriculum_state)
            for name, tensor in full.payload["policy"].items():
                self.assertTrue(torch.equal(tensor, resumed.payload["policy"][name]))
            self.assertTrue((split / "checkpoints" / "latest.pt").is_file())

            metrics = [
                json.loads(line)
                for line in (split / "metrics.jsonl").read_text().splitlines()
            ]
            kinds = {record["kind"] for record in metrics}
            self.assertTrue(
                all(record["schema_version"] == 2 for record in metrics)
            )
            self.assertIn("training", kinds)
            self.assertIn("evaluation", kinds)
            self.assertIn("curriculum_validation", kinds)
            self.assertIn("run_resumed", kinds)
            self.assertIn("run_completed", kinds)
            completions = (split / "completions.jsonl").read_text().splitlines()
            self.assertEqual(len(completions), 2)
            representative = json.loads(completions[0])
            self.assertIn("prompt_tokens", representative)
            self.assertIn("parsed_and_verified", representative)
            self.assertIn("reward", representative)
            self.assertIn(
                "sterile_repetition_rate_all", representative["reward"]
            )
            training_record = next(
                record for record in metrics if record["kind"] == "training"
            )
            self.assertIn(
                "mean_sterile_repetition_rate_all", training_record
            )
            self.assertEqual(
                training_record["sterile_repetition_mode"], "off_answer"
            )
            diagnostics = training_record["gradient_diagnostics"]
            self.assertEqual(diagnostics["measurement"], "full_batch_pre_clip")
            self.assertEqual(len(diagnostics["epochs"]), 1)
            self.assertIn("policy_norm", diagnostics["epochs"][0])
            self.assertIn("policy_valid_cosine", diagnostics["epochs"][0])

    def test_derived_resume_can_replace_checkpoint_random_streams(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            loaded = self.load_fast(root)
            original = root / "original"
            forked = root / "forked"
            original.mkdir()
            forked.mkdir()
            original_seeds = derive_named_seeds(loaded.resolved.master_seed)
            forked_seeds = derive_named_seeds(99)

            first = execute_training(
                run_directory=original,
                run_id="dddddddd-dddd-4ddd-8ddd-dddddddddddd",
                configuration=loaded.resolved,
                seeds=original_seeds,
            )
            checkpoint = load_training_checkpoint(first.final_checkpoint)
            result = execute_training(
                run_directory=forked,
                run_id="eeeeeeee-eeee-4eee-8eee-eeeeeeeeeeee",
                configuration=loaded.resolved,
                seeds=forked_seeds,
                checkpoint=checkpoint,
                derived_run=True,
                max_steps_override=checkpoint.step,
                restore_random_state=False,
            )

            resumed = load_training_checkpoint(result.final_checkpoint)
            expected = create_random_streams(forked_seeds).state_dict()
            actual = resumed.payload["random_streams"]
            self.assertEqual(
                actual["training_problems"], expected["training_problems"]
            )
            self.assertTrue(
                torch.equal(actual["rollout_sampling"], expected["rollout_sampling"])
            )
            self.assertNotEqual(
                actual["training_problems"],
                checkpoint.payload["random_streams"]["training_problems"],
            )
            events = [
                json.loads(line)
                for line in (forked / "metrics.jsonl").read_text().splitlines()
            ]
            resumed_event = next(
                event for event in events if event["kind"] == "run_resumed"
            )
            self.assertFalse(resumed_event["random_state_restored"])

    def test_resume_compatibility_separates_hard_and_derived_changes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            loaded = self.load_fast(root)
            run_directory = root / "run"
            run_directory.mkdir()
            result = execute_training(
                run_directory=run_directory,
                run_id="cccccccc-cccc-4ccc-8ccc-cccccccccccc",
                configuration=loaded.resolved,
                seeds=derive_named_seeds(17),
            )
            checkpoint = load_training_checkpoint(result.final_checkpoint)

            changed_reward = replace(
                loaded.resolved,
                reward=replace(
                    loaded.resolved.reward, coverage_coefficient=0.25
                ),
            )
            with self.assertRaisesRegex(ValueError, "derived run"):
                validate_resume_configuration(
                    checkpoint, changed_reward, derived_run=False
                )
            differences = validate_resume_configuration(
                checkpoint, changed_reward, derived_run=True
            )
            self.assertIn("reward.coverage_coefficient", differences)

            changed_model = replace(
                loaded.resolved,
                model=replace(loaded.resolved.model, n_layers=2),
            )
            with self.assertRaisesRegex(ValueError, "model architecture"):
                validate_resume_configuration(
                    checkpoint, changed_model, derived_run=True
                )
            changed_protocol = replace(
                loaded.resolved, minimum_reason_tokens=3
            )
            with self.assertRaisesRegex(ValueError, "completion protocol"):
                validate_resume_configuration(
                    checkpoint, changed_protocol, derived_run=True
                )
            changed_algorithm = replace(loaded.resolved, algorithm="ppo")
            with self.assertRaisesRegex(ValueError, "optimization algorithm"):
                validate_resume_configuration(
                    checkpoint, changed_algorithm, derived_run=True
                )
            self.assertIn(
                "reward.coverage_coefficient",
                configuration_differences(loaded.resolved, changed_reward),
            )

    def test_ppo_checkpoint_evidence_and_resume_include_learned_values(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "ppo.toml"
            path.write_text(ppo_fast_config(), encoding="utf-8")
            loaded = load_run_configuration(path)
            run_directory = root / "run"
            run_directory.mkdir()
            seeds = derive_named_seeds(loaded.resolved.master_seed)

            first = execute_training(
                run_directory=run_directory,
                run_id="eeeeeeee-eeee-4eee-8eee-eeeeeeeeeeee",
                configuration=loaded.resolved,
                seeds=seeds,
            )
            checkpoint = load_training_checkpoint(first.final_checkpoint)
            self.assertEqual(checkpoint.configuration.algorithm, "ppo")
            self.assertIn("value_head", checkpoint.payload)

            training_record = next(
                json.loads(line)
                for line in (run_directory / "metrics.jsonl").read_text().splitlines()
                if json.loads(line)["kind"] == "training"
            )
            self.assertEqual(training_record["algorithm"], "ppo")
            for field in (
                "policy_loss",
                "value_loss",
                "entropy",
                "clip_fraction",
                "explained_variance",
            ):
                self.assertIn(field, training_record)

            resumed = execute_training(
                run_directory=run_directory,
                run_id="eeeeeeee-eeee-4eee-8eee-eeeeeeeeeeee",
                configuration=loaded.resolved,
                seeds=seeds,
                checkpoint=checkpoint,
                max_steps_override=2,
            )
            resumed_checkpoint = load_training_checkpoint(
                resumed.final_checkpoint
            )
            self.assertEqual(resumed_checkpoint.step, 2)
            self.assertIn("value_head", resumed_checkpoint.payload)

    def test_checkpoint_accepts_a_valid_configuration_record_from_before_defaults_expand(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "ppo.toml"
            path.write_text(ppo_fast_config(), encoding="utf-8")
            loaded = load_run_configuration(path)
            run_directory = root / "run"
            run_directory.mkdir()
            result = execute_training(
                run_directory=run_directory,
                run_id="ffffffff-ffff-4fff-8fff-ffffffffffff",
                configuration=loaded.resolved,
                seeds=derive_named_seeds(loaded.resolved.master_seed),
            )

            payload = torch.load(
                result.final_checkpoint, map_location="cpu", weights_only=False
            )
            del payload["configuration"]["ppo"][
                "suppress_zero_reward_policy_updates"
            ]
            encoded = (
                json.dumps(
                    payload["configuration"],
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                )
                + "\n"
            ).encode("utf-8")
            payload["configuration_sha256"] = hashlib.sha256(encoded).hexdigest()
            legacy_checkpoint = root / "legacy.pt"
            torch.save(payload, legacy_checkpoint)

            checkpoint = load_training_checkpoint(legacy_checkpoint)
            self.assertFalse(
                checkpoint.configuration.ppo.suppress_zero_reward_policy_updates
            )
            self.assertEqual(
                checkpoint.configuration_sha256,
                payload["configuration_sha256"],
            )

    def test_standalone_evaluation_is_declared_and_nonmutating(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            loaded = self.load_fast(root)
            run_directory = root / "run"
            run_directory.mkdir()
            training = execute_training(
                run_directory=run_directory,
                run_id="dddddddd-dddd-4ddd-8ddd-dddddddddddd",
                configuration=loaded.resolved,
                seeds=derive_named_seeds(17),
            )
            before = load_training_checkpoint(training.final_checkpoint)
            evaluation_path = root / "evaluation.toml"
            evaluation_path.write_text(
                """\
schema_version = 1
name = "tiny-greedy"

[sampling]
max_new_tokens = 8
batch_size = 2
mode = "greedy"

[[sets]]
name = "tiny"
vertices = 4
edges = 3
min_distance = 1
max_distance = 2
example_count = 2
generation_seed = 123

[artifacts]
output_root = "evaluations"

[runtime]
device = "cpu"
dtype = "float32"
""",
                encoding="utf-8",
            )
            declaration = load_standalone_evaluation_config(evaluation_path)
            self.assertEqual(declaration.sets[0].generation_seed, 123)
            evaluated = execute_standalone_evaluation(
                training.final_checkpoint, evaluation_path
            )
            self.assertEqual(len(evaluated.results), 1)
            self.assertEqual(evaluated.results[0].metrics.example_count, 2)
            self.assertEqual(
                len(
                    (evaluated.directory / "completions.jsonl")
                    .read_text()
                    .splitlines()
                ),
                2,
            )
            after = load_training_checkpoint(training.final_checkpoint)
            for name, tensor in before.payload["policy"].items():
                self.assertTrue(torch.equal(tensor, after.payload["policy"][name]))


if __name__ == "__main__":
    unittest.main()
