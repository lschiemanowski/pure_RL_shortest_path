from __future__ import annotations

import math
import random
import unittest

from pure_rl_shortest_path.experiment import (
    CurriculumConfig,
    CurriculumState,
    FrontierValidation,
    apply_frontier_validation,
    frontier_stage_probabilities,
    generate_training_problem_batch,
)
from pure_rl_shortest_path.task import (
    GraphGenerationError,
    GraphProblemConfig,
    Vocabulary,
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


if __name__ == "__main__":
    unittest.main()
