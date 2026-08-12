from __future__ import annotations

import math
import random
import unittest

import torch
from torch import nn

from pure_rl_shortest_path.experiment import (
    CurriculumConfig,
    CurriculumState,
    EvaluationProblemSet,
    EvaluationProtocol,
    EvaluationSamplingConfig,
    FrontierValidation,
    GeneratedEvaluationCompletion,
    aggregate_evaluation_outcomes,
    apply_frontier_validation,
    fixed_evaluation_problem_set,
    frontier_stage_probabilities,
    generate_evaluation_completions,
    generate_evaluation_problem_set,
    generate_training_problem_batch,
    run_evaluation,
    select_evaluation_tokens,
)
from pure_rl_shortest_path.model import ModelOutput
from pure_rl_shortest_path.task import (
    BEGIN_ANSWER,
    END_ANSWER,
    END_REASON,
    EOS,
    AbstractGraphProblem,
    GraphExample,
    GraphGenerationError,
    GraphProblemConfig,
    Vocabulary,
    label_and_serialize_problem,
)


def stages() -> tuple[GraphProblemConfig, ...]:
    return (
        GraphProblemConfig(4, 3, 1, 2),
        GraphProblemConfig(5, 5, 1, 3),
        GraphProblemConfig(6, 7, 1, 4),
        GraphProblemConfig(7, 8, 1, 5),
        GraphProblemConfig(8, 10, 1, 6),
    )


class CurriculumDistributionTests(unittest.TestCase):
    def test_configuration_requires_a_broader_past_tail(self) -> None:
        with self.assertRaises(ValueError):
            CurriculumConfig(())
        with self.assertRaises(ValueError):
            CurriculumConfig(stages(), past_decay_scale=0.25, future_decay_scale=0.25)
        with self.assertRaises(ValueError):
            CurriculumConfig(stages(), advancement_threshold=1.1)

    def test_distribution_is_normalized_modal_and_asymmetric(self) -> None:
        curriculum = CurriculumConfig(
            stages(), past_decay_scale=2.0, future_decay_scale=0.25
        )
        probabilities = frontier_stage_probabilities(curriculum, frontier=2)
        self.assertAlmostEqual(sum(probabilities), 1.0)
        self.assertTrue(all(probability > 0.0 for probability in probabilities))
        self.assertEqual(probabilities.index(max(probabilities)), 2)
        self.assertAlmostEqual(probabilities[1] / probabilities[2], math.exp(-0.5))
        self.assertAlmostEqual(probabilities[3] / probabilities[2], math.exp(-4.0))
        self.assertGreater(probabilities[0], probabilities[3])
        self.assertGreater(probabilities[3], probabilities[4])

    def test_each_problem_draws_one_stage_and_records_the_realized_mixture(
        self,
    ) -> None:
        curriculum = CurriculumConfig(
            stages(), past_decay_scale=2.0, future_decay_scale=0.5
        )
        batch = generate_training_problem_batch(
            curriculum,
            CurriculumState(frontier=2),
            Vocabulary(16),
            problem_count=40,
            rng=random.Random(19),
        )
        repeated = generate_training_problem_batch(
            curriculum,
            CurriculumState(frontier=2),
            Vocabulary(16),
            problem_count=40,
            rng=random.Random(19),
        )
        self.assertEqual(batch, repeated)
        self.assertEqual(len(batch.examples), 40)
        self.assertEqual(len(batch.stage_indices), 40)
        self.assertEqual(sum(batch.stage_counts), 40)
        self.assertEqual(batch.frontier, 2)
        for example, stage_index in zip(batch.examples, batch.stage_indices):
            self.assertEqual(
                example.problem.vertex_count,
                curriculum.stages[stage_index].vertices,
            )
            self.assertTrue(all(label < 16 for label in example.active_labels))

    def test_generation_failure_is_not_replaced_by_another_stage(self) -> None:
        impossible = GraphProblemConfig(
            vertices=4,
            edges=6,
            min_distance=2,
            max_distance=2,
            generation_attempts=1,
        )
        curriculum = CurriculumConfig(
            (impossible,), past_decay_scale=2.0, future_decay_scale=0.25
        )
        with self.assertRaises(GraphGenerationError):
            generate_training_problem_batch(
                curriculum,
                CurriculumState(),
                Vocabulary(8),
                problem_count=1,
                rng=random.Random(5),
            )


class FrontierAdvancementTests(unittest.TestCase):
    def setUp(self) -> None:
        self.curriculum = CurriculumConfig(
            stages(),
            advancement_threshold=0.75,
            advancement_patience=2,
        )

    def validation(
        self, step: int, frontier: int, successes: int
    ) -> FrontierValidation:
        return FrontierValidation(step, frontier, 4, successes)

    def test_qualifying_streak_advances_exactly_one_stage(self) -> None:
        first = apply_frontier_validation(
            self.curriculum,
            CurriculumState(frontier=1),
            self.validation(10, 1, 3),
        )
        self.assertEqual(first.resulting_state, CurriculumState(1, 1))
        self.assertIsNone(first.transition)

        second = apply_frontier_validation(
            self.curriculum,
            first.resulting_state,
            self.validation(20, 1, 4),
        )
        self.assertEqual(second.resulting_state, CurriculumState(2, 0))
        assert second.transition is not None
        self.assertEqual(second.transition.completed_frontier, 1)
        self.assertEqual(second.transition.new_frontier, 2)
        self.assertEqual(second.transition.training_step, 20)

    def test_nonqualifying_result_resets_streak(self) -> None:
        decision = apply_frontier_validation(
            self.curriculum,
            CurriculumState(frontier=2, advancement_streak=1),
            self.validation(11, 2, 2),
        )
        self.assertFalse(decision.qualified)
        self.assertEqual(decision.resulting_state, CurriculumState(2, 0))

    def test_frontier_never_advances_past_the_final_stage(self) -> None:
        final = len(self.curriculum.stages) - 1
        decision = apply_frontier_validation(
            self.curriculum,
            CurriculumState(final, 1),
            self.validation(12, final, 4),
        )
        self.assertEqual(decision.resulting_state.frontier, final)
        self.assertIsNone(decision.transition)

    def test_validation_must_name_the_current_frontier(self) -> None:
        with self.assertRaises(ValueError):
            apply_frontier_validation(
                self.curriculum,
                CurriculumState(frontier=2),
                self.validation(13, 1, 4),
            )


def evaluation_example() -> tuple[GraphExample, Vocabulary, tuple[int, ...]]:
    vocabulary = Vocabulary(12)
    problem = AbstractGraphProblem(
        vertex_count=4,
        edges=((0, 1), (1, 3), (0, 2), (2, 3)),
        source=0,
        target=3,
        shortest_distance=2,
    )
    example = label_and_serialize_problem(problem, vocabulary, random.Random(7))
    path = tuple(example.label_by_vertex[index] for index in (0, 1, 3))
    return example, vocabulary, path


def completed_path(
    vocabulary: Vocabulary, path: tuple[int, ...]
) -> tuple[int, ...]:
    return (
        *(vocabulary.node_token(label) for label in path[1:]),
        END_REASON,
        BEGIN_ANSWER,
        *(vocabulary.node_token(label) for label in path),
        END_ANSWER,
        EOS,
    )


class EosPolicy(nn.Module):
    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.eos_logit = nn.Parameter(torch.tensor(0.0))
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
        logits[..., EOS] = self.eos_logit
        return ModelOutput(logits, () if use_cache else None)


class IndependentEvaluationSetTests(unittest.TestCase):
    def test_fresh_seed_set_is_reproducible_and_identified(self) -> None:
        config = GraphProblemConfig(6, 7, 1, 4)
        vocabulary = Vocabulary(16)
        first = generate_evaluation_problem_set(config, vocabulary, 5, 913)
        second = generate_evaluation_problem_set(config, vocabulary, 5, 913)
        self.assertEqual(first, second)
        self.assertEqual(first.identity, "seed:913")
        self.assertEqual(len(first.examples), 5)

    def test_fixed_set_requires_identity_instead_of_seed(self) -> None:
        example, _, _ = evaluation_example()
        config = GraphProblemConfig(4, 4, 2, 2)
        fixed = fixed_evaluation_problem_set("held-out-a", config, [example])
        self.assertEqual(fixed.identity, "fixed:held-out-a")
        with self.assertRaises(ValueError):
            EvaluationProblemSet(config, (example,), 3, "both")


class EvaluationSamplingTests(unittest.TestCase):
    def test_greedy_selects_argmax_and_rejects_unused_settings(self) -> None:
        sampling = EvaluationSamplingConfig(max_new_tokens=4)
        selected = select_evaluation_tokens(
            torch.tensor([[0.0, 3.0, 1.0]]), sampling
        )
        self.assertEqual(selected.tolist(), [1])
        with self.assertRaises(ValueError):
            EvaluationSamplingConfig(4, mode="greedy", temperature=1.0)

    def test_top_p_keeps_the_token_that_crosses_the_threshold(self) -> None:
        sampling = EvaluationSamplingConfig(
            4, mode="stochastic", temperature=1.0, top_p=0.7
        )
        logits = torch.tensor([[math.log(0.6), math.log(0.3), math.log(0.1)]]).repeat(
            128, 1
        )
        selected = select_evaluation_tokens(
            logits, sampling, torch.Generator().manual_seed(4)
        )
        self.assertEqual(set(selected.tolist()), {0, 1})

    def test_generation_is_unconstrained_and_restores_model_mode(self) -> None:
        example, vocabulary, _ = evaluation_example()
        problem_set = fixed_evaluation_problem_set(
            "single", GraphProblemConfig(4, 4, 2, 2), [example]
        )
        model = EosPolicy(vocabulary.size).train()
        completions = generate_evaluation_completions(
            model,
            problem_set,
            EvaluationSamplingConfig(4),
            device=torch.device("cpu"),
        )
        self.assertTrue(model.training)
        self.assertEqual(completions[0].completion, (EOS,))
        self.assertTrue(completions[0].terminated_by_eos)


class EvaluationAggregationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.example, self.vocabulary, self.path = evaluation_example()
        self.problem_set = fixed_evaluation_problem_set(
            "metrics",
            GraphProblemConfig(4, 4, 2, 2),
            [self.example, self.example, self.example],
        )

    def test_primary_outcomes_retain_exact_counts_and_rates(self) -> None:
        shortest = completed_path(self.vocabulary, self.path)
        invalid = (
            *(self.vocabulary.node_token(label) for label in self.path[1:]),
            END_REASON,
            BEGIN_ANSWER,
            self.vocabulary.node_token(self.example.source),
            END_ANSWER,
            EOS,
        )
        completions = (
            GeneratedEvaluationCompletion(self.example, shortest, True),
            GeneratedEvaluationCompletion(self.example, invalid, True),
            GeneratedEvaluationCompletion(self.example, (EOS,), True),
        )
        result = aggregate_evaluation_outcomes(
            EvaluationProtocol("checkpoint-10", EvaluationSamplingConfig(16)),
            self.problem_set,
            completions,
            self.vocabulary,
        )
        metrics = result.metrics
        self.assertEqual(metrics.example_count, 3)
        self.assertEqual(metrics.format_successes, 2)
        self.assertEqual(metrics.valid_path_successes, 1)
        self.assertEqual(metrics.shortest_path_successes, 1)
        self.assertAlmostEqual(metrics.format_success_rate, 2 / 3)
        self.assertAlmostEqual(metrics.valid_path_success_rate, 1 / 3)
        self.assertAlmostEqual(metrics.shortest_path_success_rate, 1 / 3)
        self.assertEqual(metrics.mean_valid_path_length, 2.0)
        self.assertEqual(metrics.mean_valid_excess_length, 0.0)

    def test_intervention_identity_and_comparison_are_paired(self) -> None:
        sampling = EvaluationSamplingConfig(8)
        with self.assertRaises(ValueError):
            EvaluationProtocol("checkpoint", sampling, intervention_id="random-trace")
        protocol = EvaluationProtocol(
            "checkpoint",
            sampling,
            intervention_id="random-trace",
            comparison_condition="ordinary-greedy",
        )
        self.assertEqual(protocol.kind, "intervention")

    def test_run_evaluation_records_checkpoint_and_does_not_update_policy(self) -> None:
        single_set = fixed_evaluation_problem_set(
            "single",
            GraphProblemConfig(4, 4, 2, 2),
            [self.example],
        )
        model = EosPolicy(self.vocabulary.size)
        parameters_before = tuple(
            parameter.detach().clone() for parameter in model.parameters()
        )
        result = run_evaluation(
            model,
            EvaluationProtocol("checkpoint-22", EvaluationSamplingConfig(4)),
            single_set,
            self.vocabulary,
            device=torch.device("cpu"),
        )
        self.assertEqual(result.protocol.policy_checkpoint, "checkpoint-22")
        self.assertEqual(result.problem_set_identity, "fixed:single")
        self.assertEqual(result.metrics.example_count, 1)
        self.assertEqual(result.metrics.format_successes, 0)
        for before, after in zip(parameters_before, model.parameters()):
            torch.testing.assert_close(before, after)


if __name__ == "__main__":
    unittest.main()
