from __future__ import annotations

import unittest

import torch

from pure_rl_shortest_path.model import Transformer, TransformerConfig, count_parameters


def small_config(**overrides: object) -> TransformerConfig:
    values: dict[str, object] = {
        "vocab_size": 32,
        "max_context_length": 16,
        "d_model": 24,
        "n_layers": 2,
        "n_heads": 3,
        "mlp_dim": 48,
    }
    values.update(overrides)
    return TransformerConfig(**values)  # type: ignore[arg-type]


class TransformerConstructionTests(unittest.TestCase):
    def test_configuration_rejects_invalid_dimensions(self) -> None:
        invalid = (
            {"vocab_size": 0},
            {"max_context_length": 0},
            {"d_model": 25, "n_heads": 3},
            {"d_model": 15, "n_heads": 3},
            {"n_layers": 0},
            {"mlp_dim": 0},
            {"rope_base": 0.0},
            {"norm_eps": 0.0},
        )
        for overrides in invalid:
            with self.subTest(overrides=overrides), self.assertRaises(
                (TypeError, ValueError)
            ):
                small_config(**overrides)

    def test_construction_is_reproducible_from_torch_random_state(self) -> None:
        config = small_config()
        torch.manual_seed(71)
        first = Transformer(config)
        torch.manual_seed(71)
        second = Transformer(config)
        self.assertTrue(
            all(
                torch.equal(first_value, second_value)
                for first_value, second_value in zip(
                    first.state_dict().values(), second.state_dict().values()
                )
            )
        )

    def test_model_uses_tied_full_vocabulary_output_weights(self) -> None:
        model = Transformer(small_config())
        self.assertIs(model.embedding.weight, model.lm_head.weight)
        output = model(torch.tensor([[1, 2, 3]], dtype=torch.long))
        self.assertEqual(output.logits.shape, (1, 3, model.config.vocab_size))
        self.assertIsNone(output.past_key_values)
        self.assertGreater(count_parameters(model), 0)


class PrefixScoringTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(13)
        self.model = Transformer(small_config()).eval()

    def test_scores_are_causal(self) -> None:
        first = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
        second = torch.tensor([[1, 2, 3, 17, 19]], dtype=torch.long)
        with torch.no_grad():
            first_logits = self.model(first).logits
            second_logits = self.model(second).logits
        torch.testing.assert_close(first_logits[:, :3], second_logits[:, :3])

    def test_single_token_cached_scoring_matches_full_prefix(self) -> None:
        tokens = torch.tensor([[1, 5, 7, 9, 11]], dtype=torch.long)
        with torch.no_grad():
            expected = self.model(tokens).logits
            cache = None
            pieces = []
            for index in range(tokens.shape[1]):
                result = self.model(
                    tokens[:, index : index + 1],
                    attention_mask=torch.ones((1, index + 1), dtype=torch.bool),
                    past_key_values=cache,
                    use_cache=True,
                )
                pieces.append(result.logits)
                cache = result.past_key_values
            actual = torch.cat(pieces, dim=1)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    def test_multi_token_cached_scoring_matches_full_prefix(self) -> None:
        tokens = torch.tensor([[2, 4, 6, 8, 10]], dtype=torch.long)
        with torch.no_grad():
            expected = self.model(tokens).logits
            first = self.model(tokens[:, :3], use_cache=True)
            second = self.model(
                tokens[:, 3:],
                attention_mask=torch.ones((1, 5), dtype=torch.bool),
                past_key_values=first.past_key_values,
                use_cache=True,
            )
            actual = torch.cat((first.logits, second.logits), dim=1)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    def test_padding_mask_preserves_effective_prefix_scores(self) -> None:
        padded = torch.tensor([[29, 30, 3, 7]], dtype=torch.long)
        mask = torch.tensor([[False, False, True, True]])
        compact = torch.tensor([[3, 7]], dtype=torch.long)
        with torch.no_grad():
            padded_logits = self.model(padded, attention_mask=mask).logits[:, -1]
            compact_logits = self.model(compact).logits[:, -1]
        torch.testing.assert_close(padded_logits, compact_logits, rtol=1e-5, atol=1e-6)

    def test_context_and_state_shapes_are_validated(self) -> None:
        with self.assertRaises(ValueError):
            self.model(torch.ones((1, 17), dtype=torch.long))
        with self.assertRaises(ValueError):
            self.model(
                torch.ones((1, 2), dtype=torch.long),
                attention_mask=torch.ones((1, 1), dtype=torch.bool),
            )
        with self.assertRaises(ValueError):
            self.model(
                torch.ones((1, 1), dtype=torch.long), past_key_values=(), use_cache=True
            )


if __name__ == "__main__":
    unittest.main()
