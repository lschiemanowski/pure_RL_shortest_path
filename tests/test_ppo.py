from __future__ import annotations

from copy import deepcopy
import math
import unittest

import torch

from pure_rl_shortest_path.model import Transformer, TransformerConfig
from pure_rl_shortest_path.ppo import (
    PPOConfig,
    ValueHead,
    generalized_advantages,
    normalize_advantages,
    ppo_update,
    prepare_ppo_batch,
)
from pure_rl_shortest_path.rl import PackedRollouts


def tiny_actor() -> Transformer:
    return Transformer(
        TransformerConfig(
            vocab_size=8,
            max_context_length=8,
            d_model=8,
            n_layers=1,
            n_heads=2,
            mlp_dim=16,
        )
    )


class GeneralizedAdvantageTests(unittest.TestCase):
    def test_terminal_reward_is_traced_back_over_real_actions(self) -> None:
        advantages, returns = generalized_advantages(
            torch.tensor([1.0]),
            torch.tensor([[0.5, 0.25, 0.0]]),
            torch.tensor([[True, True, False]]),
            gamma=0.9,
            gae_lambda=0.8,
        )

        torch.testing.assert_close(
            advantages, torch.tensor([[0.265, 0.75, 0.0]])
        )
        torch.testing.assert_close(returns, torch.tensor([[0.765, 1.0, 0.0]]))

    def test_constant_advantages_are_not_erased(self) -> None:
        advantages = torch.tensor([[2.0, 2.0, 0.0]])
        mask = torch.tensor([[True, True, False]])
        normalized = normalize_advantages(advantages, mask, 1e-6)
        torch.testing.assert_close(normalized, advantages)


class PPOUpdateTests(unittest.TestCase):
    def test_configuration_rejects_invalid_hyperparameters(self) -> None:
        with self.assertRaises(ValueError):
            PPOConfig(clip_epsilon=1.0)
        with self.assertRaises(ValueError):
            PPOConfig(gae_lambda=1.1)
        with self.assertRaises(ValueError):
            PPOConfig(microbatch_size=0)

    def test_hidden_state_path_preserves_actor_logits(self) -> None:
        torch.manual_seed(5)
        actor = tiny_actor().eval()
        tokens = torch.tensor([[1, 2, 3, 4]])
        with torch.no_grad():
            ordinary = actor(tokens).logits
            logits, hidden_states = actor.forward_with_hidden_states(tokens)
        torch.testing.assert_close(logits, ordinary)
        self.assertEqual(hidden_states.shape, (1, 4, actor.config.d_model))

    def test_fixed_batch_update_changes_parameters_and_reports_finite_metrics(
        self,
    ) -> None:
        torch.manual_seed(11)
        actor = tiny_actor()
        reference = deepcopy(actor).eval()
        value_head = ValueHead(actor.config.d_model)
        packed = PackedRollouts(
            input_ids=torch.tensor([[1, 2, 3, 4], [2, 3, 4, 5]]),
            attention_mask=torch.ones((2, 4), dtype=torch.bool),
            labels=torch.tensor([[2, 3, 4, 5], [3, 4, 5, 6]]),
            action_mask=torch.tensor(
                [[False, False, True, True], [False, True, True, True]]
            ),
            valid_next_mask=torch.zeros((2, 4, 8), dtype=torch.bool),
            rewards=torch.tensor([1.0, 0.4]),
            group_ids=torch.tensor([0, 1]),
        )
        config = PPOConfig(
            update_epochs=2,
            microbatch_size=1,
            entropy_coefficient=0.01,
        )
        batch = prepare_ppo_batch(actor, value_head, packed, config)
        parameters = [*actor.parameters(), *value_head.parameters()]
        before = [parameter.detach().clone() for parameter in parameters]
        optimizer = torch.optim.AdamW(parameters, lr=1e-3)

        metrics = ppo_update(
            actor, value_head, reference, optimizer, batch, config
        )

        self.assertTrue(
            all(math.isfinite(value) for value in metrics.__dict__.values())
        )
        self.assertTrue(
            any(
                not torch.equal(previous, parameter.detach())
                for previous, parameter in zip(before, parameters)
            )
        )


if __name__ == "__main__":
    unittest.main()
