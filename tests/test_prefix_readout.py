from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile
import unittest

from pure_rl_shortest_path.prefix_readout import (
    PrefixReadoutConfig,
    create_artifact_directory,
    eligible_reasoning_prefixes,
    load_prefix_readout_config,
    validate_checkpoint_digest,
)
from pure_rl_shortest_path.task import (
    BEGIN_ANSWER,
    END_ANSWER,
    END_REASON,
    EOS,
    JUMP,
    Vocabulary,
)


class ReasoningPrefixBoundaryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.vocabulary = Vocabulary(12)
        node = self.vocabulary.node_token
        self.completion = (
            node(1),
            node(2),
            JUMP,
            node(3),
            node(4),
            END_REASON,
            BEGIN_ANSWER,
            node(1),
            node(4),
            END_ANSWER,
            EOS,
        )

    def test_boundaries_end_at_nodes_and_respect_minimum_reasoning_length(self) -> None:
        boundaries = eligible_reasoning_prefixes(
            self.completion, self.vocabulary, minimum_reason_tokens=3
        )

        self.assertEqual(
            [boundary.reasoning_token_count for boundary in boundaries], [4, 5]
        )
        self.assertEqual([boundary.walk_index for boundary in boundaries], [1, 1])
        self.assertEqual(
            [boundary.follows_restart for boundary in boundaries], [True, False]
        )
        self.assertEqual(boundaries[0].prefix_tokens, self.completion[:4])

    def test_invalid_baseline_has_no_silent_partial_prefixes(self) -> None:
        malformed = self.completion[:-1]
        self.assertEqual(
            eligible_reasoning_prefixes(
                malformed, self.vocabulary, minimum_reason_tokens=1
            ),
            (),
        )


class PrefixReadoutConfigurationTests(unittest.TestCase):
    def test_loads_strict_toml_and_resolves_relative_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "study.toml"
            path.write_text(
                """
schema_version = 1
name = "small-prefix-study"
checkpoint_sha256 = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"

[baseline_sampling]
max_new_tokens = 32
batch_size = 4

[readout_sampling]
max_new_tokens = 8
batch_size = 16

[set]
vertices = 4
edges = 4
min_distance = 2
max_distance = 2
example_count = 3
generation_seed = 17

[artifacts]
output_root = "../evidence"

[runtime]
device = "cpu"
dtype = "float32"
allow_dirty_source = true
""".strip()
                + "\n",
                encoding="utf-8",
            )

            loaded = load_prefix_readout_config(path)

            self.assertIsInstance(loaded, PrefixReadoutConfig)
            self.assertEqual(loaded.name, "small-prefix-study")
            self.assertEqual(loaded.problem.example_count, 3)
            self.assertEqual(loaded.output_root, (root / "../evidence").resolve())

    def test_unknown_configuration_field_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "study.toml"
            path.write_text(
                "schema_version = 1\nname = 'x'\ncheckpoint_sha256 = '"
                + "a" * 64
                + "'\nunexpected = true\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "unexpected"):
                load_prefix_readout_config(path)


class PrefixReadoutProvenanceTests(unittest.TestCase):
    def test_checkpoint_digest_must_match_before_loading(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "checkpoint.pt"
            checkpoint.write_bytes(b"frozen policy")
            expected = hashlib.sha256(b"frozen policy").hexdigest()
            self.assertEqual(validate_checkpoint_digest(checkpoint, expected), expected)
            with self.assertRaisesRegex(ValueError, "digest"):
                validate_checkpoint_digest(checkpoint, "0" * 64)

    def test_artifact_directory_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            created = create_artifact_directory(root, "study-id")
            self.assertTrue(created.is_dir())
            with self.assertRaises(FileExistsError):
                create_artifact_directory(root, "study-id")


if __name__ == "__main__":
    unittest.main()
