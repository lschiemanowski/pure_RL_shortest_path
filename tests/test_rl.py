from __future__ import annotations

from copy import deepcopy
import random
import unittest

import torch
from torch import nn

from pure_rl_shortest_path.model import ModelOutput, Transformer, TransformerConfig
from pure_rl_shortest_path.rl import (
    GRPOConfig,
    RolloutSamplingConfig,
    SampledCompletion,
    collect_grouped_completions,
    evaluate_samples,
    grpo_update,
    group_relative_advantages,
    pack_rollouts,
    reward_from_outcome,
    valid_next_token_sets,
)
from pure_rl_shortest_path.task import (
    BEGIN_ANSWER,
    END_ANSWER,
    END_REASON,
    EOS,
    JUMP,
    PAD,
    AbstractGraphProblem,
    Vocabulary,
    adjacency_from_edges,
    label_and_serialize_problem,
    verify_completion,
)


def make_example() -> tuple[object, Vocabulary]:
    vocabulary = Vocabulary(12)
    problem = AbstractGraphProblem(
        vertex_count=5,
        edges=((0, 1), (1, 4), (0, 2), (2, 3), (3, 4)),
        source=0,
        target=4,
        shortest_distance=2,
    )
    return label_and_serialize_problem(problem, vocabulary, random.Random(7)), vocabulary


def labeled_path(example: object, abstract_path: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(example.label_by_vertex[vertex] for vertex in abstract_path)


def completion_for(
    vocabulary: Vocabulary,
    reasoning_path: tuple[int, ...],
    answer_path: tuple[int, ...],
) -> tuple[int, ...]:
    return (
        *(vocabulary.node_token(node) for node in reasoning_path[1:]),
        END_REASON,
        BEGIN_ANSWER,
        *(vocabulary.node_token(node) for node in answer_path),
        END_ANSWER,
        EOS,
    )


class EosPolicy(nn.Module):
    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.config = type(
            "Config", (), {"max_context_length": 128, "vocab_size": vocab_size}
        )()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: object | None = None,
        use_cache: bool = False,
    ) -> ModelOutput:
        logits = torch.full(
            (*input_ids.shape, self.config.vocab_size),
            -torch.inf,
            device=input_ids.device,
        )
        logits[..., EOS] = 0.0
        return ModelOutput(logits, () if use_cache else None)


class RolloutSamplingTests(unittest.TestCase):
    def test_training_sampling_requires_the_literal_policy_distribution(self) -> None:
        RolloutSamplingConfig(2, 8, temperature=1.0, top_p=1.0)
        with self.assertRaises(ValueError):
            RolloutSamplingConfig(2, 8, temperature=0.7)
        with self.assertRaises(ValueError):
            RolloutSamplingConfig(2, 8, top_p=0.9)

    def test_sampling_preserves_adjacent_groups_and_stops_at_eos(self) -> None:
        example, vocabulary = make_example()
        samples = collect_grouped_completions(
            EosPolicy(vocabulary.size),
            [example, example],
            RolloutSamplingConfig(group_size=3, max_new_tokens=5),
            device=torch.device("cpu"),
            generator=torch.Generator().manual_seed(4),
        )
        self.assertEqual([sample.group_id for sample in samples], [0, 0, 0, 1, 1, 1])
        self.assertTrue(all(sample.completion == (EOS,) for sample in samples))
        self.assertTrue(all(sample.terminated_by_eos for sample in samples))


class RewardTests(unittest.TestCase):
    def setUp(self) -> None:
        self.example, self.vocabulary = make_example()
        self.shortest = labeled_path(self.example, (0, 1, 4))
        self.long = labeled_path(self.example, (0, 2, 3, 4))

    def facts(self, completion: tuple[int, ...]):
        return verify_completion(self.example, completion, self.vocabulary)

    def test_reward_cases_and_coverage_are_separately_recorded(self) -> None:
        malformed = reward_from_outcome(self.facts((EOS,)), 0.4)
        invalid_completion = completion_for(
            self.vocabulary,
            self.shortest,
            (self.example.source,),
        )
        invalid = reward_from_outcome(self.facts(invalid_completion), 0.4)
        shortest_completion = completion_for(
            self.vocabulary, self.shortest, self.shortest
        )
        shortest = reward_from_outcome(self.facts(shortest_completion), 0.4)
        long_completion = completion_for(self.vocabulary, self.long, self.long)
        longer = reward_from_outcome(self.facts(long_completion), 0.4)

        self.assertEqual(malformed.total_reward, 0.0)
        self.assertEqual(invalid.base_reward, 0.05)
        self.assertEqual(invalid.reasoning_coverage, 0.0)
        self.assertEqual(shortest.base_reward, 1.0)
        self.assertEqual(shortest.reasoning_coverage, 1.0)
        self.assertEqual(shortest.total_reward, 1.4)
        self.assertAlmostEqual(longer.base_reward, 0.5 * 2 / 3)
        self.assertEqual(longer.reasoning_coverage, 1.0)


class AuxiliaryObjectiveTests(unittest.TestCase):
    def setUp(self) -> None:
        self.example, self.vocabulary = make_example()
        self.source = self.example.label_by_vertex[0]
        self.neighbor = self.example.label_by_vertex[1]

    def token(self, label: int) -> int:
        return self.vocabulary.node_token(label)

    def test_protocol_state_rejects_invalid_actions_without_editing_the_trace(self) -> None:
        completion = (
            self.token(self.neighbor),
            PAD,
            END_REASON,
            BEGIN_ANSWER,
            self.token(self.source),
            self.token(self.neighbor),
            self.token(self.source),
        )
        valid = valid_next_token_sets(self.example, completion, self.vocabulary)
        self.assertNotIn(JUMP, valid[0])
        self.assertNotIn(END_REASON, valid[0])
        self.assertIn(JUMP, valid[1])
        self.assertIn(END_REASON, valid[1])
        self.assertEqual(valid[1], valid[2])
        self.assertEqual(valid[3], frozenset({BEGIN_ANSWER}))
        self.assertEqual(valid[4], frozenset({self.token(self.source)}))
        self.assertNotIn(self.token(self.source), valid[6])

    def test_packing_marks_only_sampled_completion_tokens_as_actions(self) -> None:
        path = labeled_path(self.example, (0, 1, 4))
        completion = completion_for(self.vocabulary, path, path)
        sample = SampledCompletion(self.example, completion, 0, True)
        rollout = evaluate_samples([sample], self.vocabulary)[0]
        packed = pack_rollouts([rollout], self.vocabulary, device=torch.device("cpu"))
        self.assertEqual(int(packed.action_mask.sum().item()), len(completion))
        self.assertTrue(
            bool(packed.action_mask[0, len(self.example.prompt) - 1].item())
        )
        self.assertFalse(bool(packed.action_mask[0, 0].item()))


class GRPOUpdateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.example, self.vocabulary = make_example()
        shortest = labeled_path(self.example, (0, 1, 4))
        longer = labeled_path(self.example, (0, 2, 3, 4))
        completions = (
            (EOS,),
            completion_for(
                self.vocabulary, shortest, (self.example.source,)
            ),
            completion_for(self.vocabulary, shortest, shortest),
            completion_for(self.vocabulary, longer, longer),
        )
        samples = [
            SampledCompletion(self.example, completion, index // 2, completion[-1] == EOS)
            for index, completion in enumerate(completions)
        ]
        self.rollouts = evaluate_samples(samples, self.vocabulary)

    def model(self) -> Transformer:
        return Transformer(
            TransformerConfig(
                vocab_size=self.vocabulary.size,
                max_context_length=64,
                d_model=24,
                n_layers=1,
                n_heads=3,
                mlp_dim=48,
            )
        )

    def test_advantages_are_group_local_and_zero_for_ties(self) -> None:
        advantages, zero_fraction = group_relative_advantages(
            torch.tensor([1.0, 1.0, 1.0, 3.0]),
            torch.tensor([4, 4, 9, 9]),
            group_size=2,
        )
        torch.testing.assert_close(advantages, torch.tensor([0.0, 0.0, -1.0, 1.0]))
        self.assertEqual(zero_fraction, 0.5)

    def test_microbatching_preserves_the_global_update(self) -> None:
        torch.manual_seed(31)
        full_model = self.model()
        micro_model = deepcopy(full_model)
        reference = deepcopy(full_model)
        packed = pack_rollouts(
            self.rollouts, self.vocabulary, device=torch.device("cpu")
        )
        full_optimizer = torch.optim.SGD(full_model.parameters(), lr=0.01)
        micro_optimizer = torch.optim.SGD(micro_model.parameters(), lr=0.01)
        common = dict(
            group_size=2,
            update_epochs=1,
            kl_coefficient=0.1,
            valid_coefficient=0.2,
        )

        full_metrics = grpo_update(
            full_model,
            reference,
            full_optimizer,
            packed,
            GRPOConfig(microbatch_size=4, **common),
        )
        micro_metrics = grpo_update(
            micro_model,
            reference,
            micro_optimizer,
            packed,
            GRPOConfig(microbatch_size=1, **common),
        )

        for full, micro in zip(full_model.parameters(), micro_model.parameters()):
            torch.testing.assert_close(full, micro, rtol=1e-5, atol=1e-7)
        self.assertAlmostEqual(full_metrics.policy_loss, micro_metrics.policy_loss, places=6)
        self.assertAlmostEqual(full_metrics.valid_loss, micro_metrics.valid_loss, places=6)


if __name__ == "__main__":
    unittest.main()
