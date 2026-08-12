"""Synthetic shortest-path tasks and policy-independent verification.

This module deliberately contains no model or learning code.  It constructs
graph/query prompts without answers, parses freely generated completions, and
derives deterministic facts that a trainer may later turn into rewards.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import heapq
import random
from typing import Iterable, Sequence


# Structural token IDs are stable experiment semantics.  Node label ``i`` is
# represented by ``NODE_OFFSET + i``.
PAD = 0
BOS = 1
EDGES = 2
QUERY = 3
BEGIN_REASON = 4
JUMP = 5
END_REASON = 6
BEGIN_ANSWER = 7
END_ANSWER = 8
EOS = 9
NODE_OFFSET = 10

TOKEN_NAMES = {
    PAD: "PAD",
    BOS: "BOS",
    EDGES: "EDGES",
    QUERY: "QUERY",
    BEGIN_REASON: "BEGIN_REASON",
    JUMP: "JUMP",
    END_REASON: "END_REASON",
    BEGIN_ANSWER: "BEGIN_ANSWER",
    END_ANSWER: "END_ANSWER",
    EOS: "EOS",
}

Edge = tuple[int, int]
Query = tuple[int, int, int]


def _plain_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    return value


@dataclass(frozen=True)
class GraphProblemConfig:
    """Distribution parameters for one graph/query difficulty bucket."""

    vertices: int
    edges: int
    min_distance: int
    max_distance: int
    generation_attempts: int = 100

    def __post_init__(self) -> None:
        vertices = _plain_int(self.vertices, "vertices")
        edges = _plain_int(self.edges, "edges")
        min_distance = _plain_int(self.min_distance, "min_distance")
        max_distance = _plain_int(self.max_distance, "max_distance")
        attempts = _plain_int(self.generation_attempts, "generation_attempts")
        if vertices < 2:
            raise ValueError("vertices must be at least two")
        maximum_edges = vertices * (vertices - 1) // 2
        if not vertices - 1 <= edges <= maximum_edges:
            raise ValueError(
                f"edges must lie in [{vertices - 1}, {maximum_edges}] for "
                f"vertices={vertices}"
            )
        if not 1 <= min_distance <= max_distance <= vertices - 1:
            raise ValueError(
                "distance interval must satisfy "
                "1 <= min_distance <= max_distance <= vertices - 1"
            )
        if attempts < 1:
            raise ValueError("generation_attempts must be positive")

    @property
    def label(self) -> str:
        distance = (
            str(self.min_distance)
            if self.min_distance == self.max_distance
            else f"{self.min_distance}-{self.max_distance}"
        )
        return f"n{self.vertices}_e{self.edges}_d{distance}"


@dataclass(frozen=True)
class Vocabulary:
    """The fixed structural symbols and global pool of node labels."""

    node_label_count: int

    def __post_init__(self) -> None:
        count = _plain_int(self.node_label_count, "node_label_count")
        if count < 2:
            raise ValueError("node_label_count must be at least two")

    @property
    def size(self) -> int:
        return NODE_OFFSET + self.node_label_count

    def node_token(self, label: int) -> int:
        label = _plain_int(label, "label")
        if not 0 <= label < self.node_label_count:
            raise ValueError(f"node label {label} is outside the configured pool")
        return NODE_OFFSET + label

    def token_node(self, token: int) -> int | None:
        label = token - NODE_OFFSET
        return label if 0 <= label < self.node_label_count else None

    def render(self, tokens: Sequence[int]) -> str:
        rendered: list[str] = []
        for token in tokens:
            label = self.token_node(token)
            rendered.append(
                str(label) if label is not None else TOKEN_NAMES.get(token, f"?{token}")
            )
        return " ".join(rendered)


@dataclass(frozen=True)
class AbstractGraphProblem:
    """A graph and qualifying query before policy-visible labels are assigned."""

    vertex_count: int
    edges: tuple[Edge, ...]
    source: int
    target: int
    shortest_distance: int


@dataclass(frozen=True)
class GraphExample:
    """One labeled, serialized graph/query prompt with no target completion."""

    problem: AbstractGraphProblem
    label_by_vertex: tuple[int, ...]
    active_labels: tuple[int, ...]
    edges: tuple[Edge, ...]
    serialized_edges: tuple[Edge, ...]
    source: int
    target: int
    prompt: tuple[int, ...]
    adjacency: tuple[frozenset[int], ...]

    @property
    def shortest_distance(self) -> int:
        return self.problem.shortest_distance


class GraphGenerationError(RuntimeError):
    """Raised when bounded graph sampling cannot satisfy a query constraint."""


def canonical_edge(u: int, v: int) -> Edge:
    if u == v:
        raise ValueError("self-edges are not permitted")
    return (u, v) if u < v else (v, u)


def adjacency_from_edges(vertex_count: int, edges: Iterable[Edge]) -> tuple[frozenset[int], ...]:
    adjacency = [set() for _ in range(vertex_count)]
    for u, v in edges:
        if not 0 <= u < vertex_count or not 0 <= v < vertex_count:
            raise ValueError("edge endpoint is outside the declared vertex set")
        if u == v:
            raise ValueError("self-edges are not permitted")
        adjacency[u].add(v)
        adjacency[v].add(u)
    return tuple(frozenset(neighbors) for neighbors in adjacency)


def bfs_distances(
    adjacency: Sequence[frozenset[int]], source: int
) -> tuple[int | None, ...]:
    if not 0 <= source < len(adjacency):
        raise ValueError("source is outside the adjacency table")
    distances: list[int | None] = [None] * len(adjacency)
    distances[source] = 0
    queue: deque[int] = deque([source])
    while queue:
        vertex = queue.popleft()
        distance = distances[vertex]
        assert distance is not None
        for neighbor in adjacency[vertex]:
            if distances[neighbor] is None:
                distances[neighbor] = distance + 1
                queue.append(neighbor)
    return tuple(distances)


def qualifying_queries(
    adjacency: Sequence[frozenset[int]], min_distance: int, max_distance: int
) -> tuple[Query, ...]:
    """Return every qualifying ordered pair in canonical order."""

    queries: list[Query] = []
    for source in range(len(adjacency)):
        distances = bfs_distances(adjacency, source)
        for target, distance in enumerate(distances):
            if distance is not None and min_distance <= distance <= max_distance:
                queries.append((source, target, distance))
    return tuple(queries)


def _uniform_random_spanning_tree(vertex_count: int, rng: random.Random) -> set[Edge]:
    """Sample a uniform labeled tree using a uniformly sampled Pruefer sequence."""

    if vertex_count == 2:
        return {(0, 1)}
    pruefer = [rng.randrange(vertex_count) for _ in range(vertex_count - 2)]
    degree = [1] * vertex_count
    for vertex in pruefer:
        degree[vertex] += 1
    leaves = [vertex for vertex, value in enumerate(degree) if value == 1]
    heapq.heapify(leaves)
    edges: set[Edge] = set()
    for vertex in pruefer:
        leaf = heapq.heappop(leaves)
        edges.add(canonical_edge(leaf, vertex))
        degree[leaf] -= 1
        degree[vertex] -= 1
        if degree[vertex] == 1:
            heapq.heappush(leaves, vertex)
    edges.add(canonical_edge(heapq.heappop(leaves), heapq.heappop(leaves)))
    return edges


def generate_graph_problem(
    config: GraphProblemConfig, rng: random.Random
) -> AbstractGraphProblem:
    """Generate a connected graph and sample uniformly from its qualifying queries."""

    all_edges = tuple(
        (u, v) for u in range(config.vertices) for v in range(u + 1, config.vertices)
    )
    extra_count = config.edges - (config.vertices - 1)
    for _ in range(config.generation_attempts):
        tree = _uniform_random_spanning_tree(config.vertices, rng)
        remaining = tuple(edge for edge in all_edges if edge not in tree)
        extras = rng.sample(remaining, extra_count)
        edges = tuple(sorted(tree | set(extras)))
        adjacency = adjacency_from_edges(config.vertices, edges)
        queries = qualifying_queries(adjacency, config.min_distance, config.max_distance)
        if queries:
            source, target, distance = rng.choice(queries)
            return AbstractGraphProblem(
                vertex_count=config.vertices,
                edges=edges,
                source=source,
                target=target,
                shortest_distance=distance,
            )
    raise GraphGenerationError(
        f"could not sample {config.label} after "
        f"{config.generation_attempts} graph attempts"
    )


def label_and_serialize_problem(
    problem: AbstractGraphProblem,
    vocabulary: Vocabulary,
    rng: random.Random,
) -> GraphExample:
    """Assign labels from the full pool and serialize a solution-free prompt."""

    if problem.vertex_count > vocabulary.node_label_count:
        raise ValueError(
            f"problem needs {problem.vertex_count} labels but the vocabulary has "
            f"{vocabulary.node_label_count}"
        )
    label_by_vertex = tuple(
        rng.sample(range(vocabulary.node_label_count), problem.vertex_count)
    )
    labeled_edges = tuple(
        sorted(
            canonical_edge(label_by_vertex[u], label_by_vertex[v])
            for u, v in problem.edges
        )
    )
    serialized_edges = list(labeled_edges)
    rng.shuffle(serialized_edges)
    serialized_edges = [
        (v, u) if rng.random() < 0.5 else (u, v) for u, v in serialized_edges
    ]
    source = label_by_vertex[problem.source]
    target = label_by_vertex[problem.target]
    prompt = [BOS, EDGES]
    for u, v in serialized_edges:
        prompt.extend((vocabulary.node_token(u), vocabulary.node_token(v)))
    prompt.extend(
        (
            QUERY,
            vocabulary.node_token(source),
            vocabulary.node_token(target),
            BEGIN_REASON,
            vocabulary.node_token(source),
        )
    )
    adjacency_mutable = [set() for _ in range(vocabulary.node_label_count)]
    for u, v in labeled_edges:
        adjacency_mutable[u].add(v)
        adjacency_mutable[v].add(u)
    return GraphExample(
        problem=problem,
        label_by_vertex=label_by_vertex,
        active_labels=tuple(sorted(label_by_vertex)),
        edges=labeled_edges,
        serialized_edges=tuple(serialized_edges),
        source=source,
        target=target,
        prompt=tuple(prompt),
        adjacency=tuple(frozenset(neighbors) for neighbors in adjacency_mutable),
    )


def generate_example(
    config: GraphProblemConfig, vocabulary: Vocabulary, rng: random.Random
) -> GraphExample:
    """Generate and serialize one graph/query training or evaluation prompt."""

    return label_and_serialize_problem(generate_graph_problem(config, rng), vocabulary, rng)


@dataclass(frozen=True)
class ParsedCompletion:
    format_ok: bool
    reasoning_tokens: tuple[int, ...]
    reasoning_walks: tuple[tuple[int, ...], ...]
    answer: tuple[int, ...]
    format_error: str | None


def _format_failure(reason: str) -> ParsedCompletion:
    return ParsedCompletion(False, (), (), (), reason)


def parse_completion(
    completion: Sequence[int], vocabulary: Vocabulary
) -> ParsedCompletion:
    """Parse generated tokens without consulting a graph or judging correctness."""

    tokens = list(completion)
    if not tokens or tokens[-1] != EOS or EOS in tokens[:-1]:
        return _format_failure("completion must end with exactly one EOS")
    try:
        end_reason = tokens.index(END_REASON)
    except ValueError:
        return _format_failure("completion has no END_REASON")
    reasoning_tokens = tokens[:end_reason]
    if not reasoning_tokens:
        return _format_failure("reasoning segment must be nonempty")

    walks: list[tuple[int, ...]] = []
    current_walk: list[int] = []
    for token in reasoning_tokens:
        if token == JUMP:
            if not current_walk:
                return _format_failure("JUMP must follow a nonempty reasoning walk")
            walks.append(tuple(current_walk))
            current_walk = []
            continue
        label = vocabulary.token_node(token)
        if label is None:
            return _format_failure("reasoning segment contains a non-node token")
        current_walk.append(label)
    if not current_walk:
        return _format_failure("JUMP must be followed by a node label")
    walks.append(tuple(current_walk))

    suffix = tokens[end_reason + 1 :]
    if not suffix or suffix[0] != BEGIN_ANSWER:
        return _format_failure("END_REASON must be followed by BEGIN_ANSWER")
    try:
        end_answer = suffix.index(END_ANSWER, 1)
    except ValueError:
        return _format_failure("answer segment has no END_ANSWER")
    if suffix[end_answer + 1 :] != [EOS]:
        return _format_failure("END_ANSWER must be followed immediately by EOS")
    answer_tokens = suffix[1:end_answer]
    if not answer_tokens:
        return _format_failure("answer path must be nonempty")
    answer: list[int] = []
    for token in answer_tokens:
        label = vocabulary.token_node(token)
        if label is None:
            return _format_failure("answer segment contains a non-node token")
        answer.append(label)
    return ParsedCompletion(
        format_ok=True,
        reasoning_tokens=tuple(reasoning_tokens),
        reasoning_walks=tuple(walks),
        answer=tuple(answer),
        format_error=None,
    )


@dataclass(frozen=True)
class ReasoningTraceFacts:
    legal_transitions: int
    illegal_transitions: int
    restarts: int
    legal_directed_transitions: tuple[tuple[int, int], ...]
    visited_nodes: tuple[int, ...]
    traversed_edges: tuple[Edge, ...]
    reaches_target: bool
    answer_edge_coverage: float
    answer_edge_precision: float


@dataclass(frozen=True)
class OutcomeFacts:
    parsed: ParsedCompletion
    valid_path: bool
    answer_length: int | None
    shortest_distance: int
    shortest: bool
    excess_length: int | None
    reasoning: ReasoningTraceFacts | None

    @property
    def format_ok(self) -> bool:
        return self.parsed.format_ok


def _shortest_labeled_distance(example: GraphExample) -> int:
    distances = bfs_distances(example.adjacency, example.source)
    distance = distances[example.target]
    if distance is None:
        raise ValueError("example graph is not connected between source and target")
    return distance


def _reasoning_trace_facts(
    example: GraphExample,
    parsed: ParsedCompletion,
) -> ReasoningTraceFacts:
    active = set(example.active_labels)
    current: int | None = example.source
    visited = {example.source}
    traversed: set[Edge] = set()
    directed_transitions: list[tuple[int, int]] = []
    legal = 0
    illegal = 0
    restarts = 0
    for walk_index, walk in enumerate(parsed.reasoning_walks):
        start = 0
        if walk_index:
            restarts += 1
            restart_node = walk[0]
            start = 1
            if restart_node in active:
                visited.add(restart_node)
                current = restart_node
            else:
                illegal += 1
                current = None
        for node in walk[start:]:
            if node in active:
                visited.add(node)
            if current is not None and node in example.adjacency[current]:
                legal += 1
                directed_transitions.append((current, node))
                traversed.add(canonical_edge(current, node))
            else:
                illegal += 1
            current = node if node in active else None

    answer_edges = {
        canonical_edge(u, v) for u, v in zip(parsed.answer, parsed.answer[1:]) if u != v
    }
    covered = len(answer_edges & traversed)
    coverage = covered / len(answer_edges) if answer_edges else 0.0
    precision = covered / len(traversed) if traversed else 0.0
    return ReasoningTraceFacts(
        legal_transitions=legal,
        illegal_transitions=illegal,
        restarts=restarts,
        legal_directed_transitions=tuple(directed_transitions),
        visited_nodes=tuple(sorted(visited)),
        traversed_edges=tuple(sorted(traversed)),
        reaches_target=example.target in visited,
        answer_edge_coverage=coverage,
        answer_edge_precision=precision,
    )


def verify_completion(
    example: GraphExample,
    completion: Sequence[int],
    vocabulary: Vocabulary,
) -> OutcomeFacts:
    """Derive exact task facts without assigning a scalar learning reward."""

    shortest_distance = _shortest_labeled_distance(example)
    parsed = parse_completion(completion, vocabulary)
    if not parsed.format_ok:
        return OutcomeFacts(parsed, False, None, shortest_distance, False, None, None)

    path = parsed.answer
    active = set(example.active_labels)
    valid_path = (
        path[0] == example.source
        and path[-1] == example.target
        and len(set(path)) == len(path)
        and all(node in active for node in path)
        and all(v in example.adjacency[u] for u, v in zip(path, path[1:]))
    )
    answer_length = len(path) - 1
    shortest = valid_path and answer_length == shortest_distance
    excess_length = answer_length - shortest_distance if valid_path else None
    reasoning = _reasoning_trace_facts(example, parsed)
    return OutcomeFacts(
        parsed=parsed,
        valid_path=valid_path,
        answer_length=answer_length,
        shortest_distance=shortest_distance,
        shortest=shortest,
        excess_length=excess_length,
        reasoning=reasoning,
    )
