from __future__ import annotations

from collections.abc import Sequence
import random
import unittest

from pure_rl_shortest_path.task import (
    BEGIN_ANSWER,
    END_ANSWER,
    END_REASON,
    EOS,
    JUMP,
    AbstractGraphProblem,
    GraphProblemConfig,
    Vocabulary,
    adjacency_from_edges,
    canonical_edge,
    generate_example,
    generate_graph_problem,
    label_and_serialize_problem,
    parse_completion,
    qualifying_queries,
    verify_completion,
)


class RecordingRandom(random.Random):
    def __init__(self, seed: int) -> None:
        super().__init__(seed)
        self.choice_populations: list[tuple[object, ...]] = []
        self.sample_calls: list[tuple[tuple[object, ...], int]] = []

    def choice(self, seq: Sequence[object]) -> object:
        self.choice_populations.append(tuple(seq))
        return super().choice(seq)

    def sample(
        self, population: Sequence[object], k: int, *, counts: Sequence[int] | None = None
    ) -> list[object]:
        materialized = tuple(population)
        self.sample_calls.append((materialized, k))
        return super().sample(materialized, k, counts=counts)


class GraphConstructionTests(unittest.TestCase):
    def test_configuration_rejects_impossible_values(self) -> None:
        invalid = (
            (1, 0, 1, 1),
            (4, 2, 1, 1),
            (4, 7, 1, 1),
            (4, 3, 0, 1),
            (4, 3, 2, 1),
            (4, 3, 1, 4),
        )
        for values in invalid:
            with self.subTest(values=values), self.assertRaises(ValueError):
                GraphProblemConfig(*values)

    def test_generation_is_reproducible_and_satisfies_constraints(self) -> None:
        config = GraphProblemConfig(12, 18, 2, 4)
        first = generate_graph_problem(config, random.Random(91))
        second = generate_graph_problem(config, random.Random(91))
        self.assertEqual(first, second)
        self.assertEqual(len(first.edges), config.edges)
        self.assertEqual(len(set(first.edges)), config.edges)
        self.assertTrue(all(u < v for u, v in first.edges))
        self.assertNotEqual(first.source, first.target)
        self.assertGreaterEqual(first.shortest_distance, config.min_distance)
        self.assertLessEqual(first.shortest_distance, config.max_distance)

    def test_query_is_chosen_from_all_qualifying_ordered_pairs(self) -> None:
        config = GraphProblemConfig(8, 9, 2, 3)
        rng = RecordingRandom(17)
        problem = generate_graph_problem(config, rng)
        adjacency = adjacency_from_edges(problem.vertex_count, problem.edges)
        expected = qualifying_queries(adjacency, config.min_distance, config.max_distance)
        self.assertEqual(rng.choice_populations[-1], expected)
        self.assertIn(
            (problem.source, problem.target, problem.shortest_distance), expected
        )
        self.assertTrue(
            any((target, source, distance) in expected for source, target, distance in expected)
        )


class LabelingAndSerializationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.problem = AbstractGraphProblem(
            vertex_count=4,
            edges=((0, 1), (1, 2), (2, 3), (0, 3)),
            source=0,
            target=2,
            shortest_distance=2,
        )
        self.vocabulary = Vocabulary(16)

    def test_labels_are_sampled_from_the_complete_pool(self) -> None:
        rng = RecordingRandom(5)
        example = label_and_serialize_problem(self.problem, self.vocabulary, rng)
        self.assertIn((tuple(range(16)), 4), rng.sample_calls)
        self.assertEqual(len(set(example.label_by_vertex)), 4)
        self.assertTrue(all(0 <= label < 16 for label in example.label_by_vertex))
        self.assertTrue(any(label >= 4 for label in example.label_by_vertex))

    def test_serialization_preserves_graph_and_contains_only_the_query(self) -> None:
        example = label_and_serialize_problem(
            self.problem, self.vocabulary, random.Random(3)
        )
        serialized = {canonical_edge(u, v) for u, v in example.serialized_edges}
        self.assertEqual(serialized, set(example.edges))
        self.assertEqual(len(example.serialized_edges), len(example.edges))
        self.assertEqual(example.prompt[-1], self.vocabulary.node_token(example.source))
        self.assertEqual(len(example.prompt), 2 + 2 * len(example.edges) + 5)
        self.assertNotIn(END_REASON, example.prompt)
        self.assertNotIn(BEGIN_ANSWER, example.prompt)
        self.assertNotIn(END_ANSWER, example.prompt)
        self.assertNotIn(EOS, example.prompt)

    def test_complete_example_generation_uses_global_vocabulary(self) -> None:
        example = generate_example(
            GraphProblemConfig(4, 4, 1, 3), Vocabulary(32), random.Random(22)
        )
        self.assertEqual(len(example.active_labels), 4)
        self.assertTrue(all(label < 32 for label in example.active_labels))


class CompletionAndVerificationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.vocabulary = Vocabulary(12)
        problem = AbstractGraphProblem(
            vertex_count=4,
            edges=((0, 1), (1, 2), (0, 3), (2, 3)),
            source=0,
            target=2,
            shortest_distance=2,
        )
        self.example = label_and_serialize_problem(
            problem, self.vocabulary, random.Random(7)
        )

    def token(self, label: int) -> int:
        return self.vocabulary.node_token(label)

    def completion(
        self, reasoning: Sequence[int | None], answer: Sequence[int]
    ) -> tuple[int, ...]:
        return (
            *(JUMP if node is None else self.token(node) for node in reasoning),
            END_REASON,
            BEGIN_ANSWER,
            *(self.token(node) for node in answer),
            END_ANSWER,
            EOS,
        )

    def shortest_path(self) -> tuple[int, ...]:
        source_vertex = self.example.problem.source
        target_vertex = self.example.problem.target
        abstract_adjacency = adjacency_from_edges(
            self.example.problem.vertex_count, self.example.problem.edges
        )
        queue = [(source_vertex, (source_vertex,))]
        seen = {source_vertex}
        for vertex, path in queue:
            if vertex == target_vertex:
                return tuple(self.example.label_by_vertex[item] for item in path)
            for neighbor in abstract_adjacency[vertex]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    queue.append((neighbor, (*path, neighbor)))
        self.fail("test graph is disconnected")

    def test_parser_returns_structured_walks_and_answer(self) -> None:
        path = self.shortest_path()
        completion = self.completion((*path[1:], None, path[0]), path)
        parsed = parse_completion(completion, self.vocabulary)
        self.assertTrue(parsed.format_ok)
        self.assertEqual(parsed.reasoning_walks, (path[1:], (path[0],)))
        self.assertEqual(parsed.answer, path)

    def test_parser_separates_format_from_graph_legality(self) -> None:
        inactive = next(label for label in range(12) if label not in self.example.active_labels)
        completion = self.completion((inactive,), (self.example.source, inactive))
        parsed = parse_completion(completion, self.vocabulary)
        facts = verify_completion(self.example, completion, self.vocabulary)
        self.assertTrue(parsed.format_ok)
        self.assertFalse(facts.valid_path)

    def test_parser_rejects_empty_or_malformed_segments(self) -> None:
        path = self.shortest_path()
        malformed = (
            (END_REASON, BEGIN_ANSWER, self.token(path[0]), END_ANSWER, EOS),
            (self.token(path[0]), JUMP, END_REASON, BEGIN_ANSWER, self.token(path[0]), END_ANSWER, EOS),
            (self.token(path[0]), END_REASON, BEGIN_ANSWER, END_ANSWER, EOS),
            (self.token(path[0]), END_REASON, BEGIN_ANSWER, self.token(path[0]), END_ANSWER),
        )
        for completion in malformed:
            with self.subTest(completion=completion):
                self.assertFalse(parse_completion(completion, self.vocabulary).format_ok)

    def test_verifier_recognizes_shortest_path_and_reasoning_coverage(self) -> None:
        path = self.shortest_path()
        completion = self.completion(path[1:], path)
        facts = verify_completion(self.example, completion, self.vocabulary)
        self.assertTrue(facts.format_ok)
        self.assertTrue(facts.valid_path)
        self.assertTrue(facts.shortest)
        self.assertEqual(facts.answer_length, self.example.shortest_distance)
        self.assertEqual(facts.excess_length, 0)
        assert facts.reasoning is not None
        self.assertEqual(facts.reasoning.illegal_transitions, 0)
        self.assertEqual(facts.reasoning.answer_edge_coverage, 1.0)
        self.assertTrue(facts.reasoning.reaches_target)

    def test_verifier_counts_restart_without_treating_it_as_an_edge(self) -> None:
        path = self.shortest_path()
        completion = self.completion((path[1], None, path[-1]), path)
        facts = verify_completion(self.example, completion, self.vocabulary)
        assert facts.reasoning is not None
        self.assertEqual(facts.reasoning.restarts, 1)
        self.assertEqual(facts.reasoning.legal_transitions, 1)
        self.assertTrue(facts.reasoning.reaches_target)

    def test_verifier_distinguishes_productive_and_sterile_repetition(self) -> None:
        path = self.shortest_path()
        answer_edges = {
            canonical_edge(u, v) for u, v in zip(path, path[1:])
        }
        off_answer_neighbor = next(
            neighbor
            for neighbor in self.example.adjacency[self.example.source]
            if canonical_edge(self.example.source, neighbor) not in answer_edges
        )
        reasoning = (
            off_answer_neighbor,
            self.example.source,
            path[1],
            self.example.source,
            off_answer_neighbor,
            None,
            self.example.source,
            off_answer_neighbor,
        )
        facts = verify_completion(
            self.example,
            self.completion(reasoning, path),
            self.vocabulary,
        )
        assert facts.reasoning is not None
        self.assertEqual(facts.reasoning.restarts, 1)
        self.assertEqual(facts.reasoning.legal_transitions, 6)
        self.assertEqual(facts.reasoning.sterile_repetitions_all, 1)
        self.assertAlmostEqual(
            facts.reasoning.sterile_repetition_rate_all, 1 / 6
        )
        self.assertEqual(facts.reasoning.off_answer_legal_transitions, 4)
        self.assertEqual(facts.reasoning.sterile_repetitions_off_answer, 1)
        self.assertAlmostEqual(
            facts.reasoning.sterile_repetition_rate_off_answer, 1 / 4
        )

        invalid = verify_completion(
            self.example,
            self.completion(reasoning, (self.example.source,)),
            self.vocabulary,
        )
        assert invalid.reasoning is not None
        self.assertFalse(invalid.valid_path)
        self.assertEqual(invalid.reasoning.off_answer_legal_transitions, 6)
        self.assertAlmostEqual(
            invalid.reasoning.sterile_repetition_rate_off_answer, 1 / 6
        )


if __name__ == "__main__":
    unittest.main()
