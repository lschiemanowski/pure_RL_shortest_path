"""Shared bounded-state mechanics for Stratic spec-led propagation."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path, PurePosixPath
import subprocess
from typing import Any, Iterable

from stratic.workflow_api import (
    CandidateGraphError,
    OperationError,
    apply_graph_mutations,
    collect_changed_paths,
    path_is_within,
    relationship_value,
    snapshot_from_commit,
    worktree_snapshot,
    yaml_text,
)


class CandidateGraph:
    def __init__(self, repository: Path, config: Any, snapshot: Any) -> None:
        self.repository = repository
        self.config = config
        self.snapshot = snapshot
        self.nodes = snapshot.nodes
        self._nodes = {node.node_id: node for node in snapshot.nodes}

    def node(self, node_id: str) -> Any:
        try:
            return self._nodes[node_id]
        except KeyError as error:
            raise OperationError(
                f"description node does not exist: {node_id}"
            ) from error

    def relationship_path(self, node: Any) -> str:
        source = PurePosixPath(node.source_path)
        descriptions = PurePosixPath(self.config.descriptions)
        try:
            relative = source.relative_to(descriptions)
        except ValueError as error:
            raise OperationError(
                "candidate description path is out of scope"
            ) from error
        assert self.config.relationships is not None
        return (
            PurePosixPath(self.config.relationships) / relative.with_suffix(".yaml")
        ).as_posix()

    def expansions(self, node_id: str) -> dict[str, tuple[str, ...]]:
        reasons: dict[str, set[str]] = defaultdict(set)
        node = self.node(node_id)
        for child in self.nodes:
            if child.parent == node_id:
                reasons[child.node_id].add("child")
        for reference in node.references:
            if reference.kind != "informational":
                reasons[reference.target].add(f"reference:{reference.kind}")
        for anchor in node.anchors:
            for link in anchor.links:
                if link.target_node_id is not None and link.kind != "informational":
                    reasons[link.target_node_id].add(f"span:{link.kind}")
        return {
            target: tuple(sorted(values)) for target, values in sorted(reasons.items())
        }

    def code_paths(self, node_id: str) -> tuple[str, ...]:
        node = self.node(node_id)
        if node.level != "L3":
            return ()
        return tuple(
            sorted(
                code.path
                for code in self.snapshot.code
                if any(
                    coverage_matches(selector, code.path)
                    for selector in node.covers
                )
            )
        )


def coverage_matches(coverage: Any, path: str) -> bool:
    """Match code using whole-path or exact-range coverage semantics."""

    if coverage.precise:
        return path == coverage.path
    return path_is_within(path, coverage.path)


def database_path(repository: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repository / path


def git(
    repository: Path, arguments: Iterable[str], *, accepted: set[int] = {0}
) -> bytes:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=repository,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode not in accepted:
        message = completed.stderr.decode("utf-8", errors="replace").strip()
        raise OperationError(message or "git command failed")
    return completed.stdout


def git_text(repository: Path, arguments: Iterable[str]) -> str:
    return git(repository, arguments).decode("utf-8", errors="strict").strip()


def changed_paths(repository: Path, base: str) -> set[str]:
    paths, issues = collect_changed_paths(repository, base, None)
    if issues:
        raise OperationError("; ".join(issue.render() for issue in issues))
    return set(paths)


def task_cursor(task: dict[str, Any]) -> dict[str, Any]:
    cursor = task.get("cursor")
    if not isinstance(cursor, dict):
        raise OperationError("propagation task has no workflow cursor")
    required = {
        "phase",
        "root_node",
        "request",
        "required_nodes",
        "required_code",
        "reviewed_nodes",
        "reviewed_code",
        "node_decisions",
        "code_decisions",
        "spent",
        "context_bytes",
        "changeset_id",
    }
    if not required <= set(cursor):
        raise OperationError("propagation task uses an unsupported cursor format")
    result = dict(cursor)
    result.setdefault("staged_nodes", [])
    result.setdefault("staged_paths", [])
    result.setdefault("code_offsets", {})
    result.setdefault("uncertainties", [])
    result.setdefault("node_reasons", {result["root_node"]: ["requested"]})
    result.setdefault("code_reasons", {})
    return result


def node_decisions(cursor: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return _decisions(cursor, "node_decisions", "node")


def code_decisions(cursor: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return _decisions(cursor, "code_decisions", "path")


def _decisions(
    cursor: dict[str, Any], field: str, identity: str
) -> dict[str, dict[str, Any]]:
    values = cursor.get(field)
    if not isinstance(values, list):
        raise OperationError(f"task {field} must be an array")
    result: dict[str, dict[str, Any]] = {}
    for item in values:
        if not isinstance(item, dict) or not isinstance(item.get(identity), str):
            raise OperationError(f"task {field} contains an invalid decision")
        result[item[identity]] = item
    return result


def candidate_graph(
    repository: Path, operations: Any, task: dict[str, Any]
) -> CandidateGraph:
    cursor = task_cursor(task)
    changeset = operations.get_changeset(cursor["changeset_id"])
    base = snapshot_from_commit(repository, str(task["base_version"]))
    current, config = worktree_snapshot(repository, template=base)
    try:
        candidate = apply_graph_mutations(
            base, changeset["graph_mutations"], code=current.code
        )
    except CandidateGraphError:
        # A propagation may introduce code and its covering L3 in the same
        # changeset. Until that L3 is staged, validate the intermediate graph
        # against base code; record validates the finished graph against the
        # complete worktree.
        candidate = apply_graph_mutations(base, (), code=base.code)
        candidate = apply_graph_mutations(
            candidate, changeset["graph_mutations"], code=base.code
        )
    return CandidateGraph(repository, config, candidate)


def compute_plan(
    graph: CandidateGraph,
    root: str,
    decisions: dict[str, dict[str, Any]],
) -> tuple[set[str], set[str], dict[str, set[str]], dict[str, set[str]]]:
    required_nodes = {root}
    node_reasons: dict[str, set[str]] = {root: {"requested"}}
    required_code: set[str] = set()
    code_reasons: dict[str, set[str]] = defaultdict(set)
    while True:
        previous = set(required_nodes)
        for node_id in sorted(previous):
            decision = decisions.get(node_id)
            if decision is None or decision.get("propagate") is not True:
                continue
            for target, reasons in graph.expansions(node_id).items():
                required_nodes.add(target)
                node_reasons.setdefault(target, set()).update(
                    f"{node_id}:{reason}" for reason in reasons
                )
            for path in graph.code_paths(node_id):
                required_code.add(path)
                code_reasons[path].add(f"{node_id}:covered-code")
        if required_nodes == previous:
            break
    return required_nodes, required_code, node_reasons, code_reasons


def node_summary(graph: CandidateGraph, node_id: str) -> dict[str, Any]:
    node = graph.node(node_id)
    return {
        "node_id": node.node_id,
        "level": node.level,
        "scope": node.scope,
        "realization": node.realization,
        "source_path": node.source_path,
        "relationship_path": graph.relationship_path(node),
        "parent": node.parent,
    }


def relationship_text(graph: CandidateGraph, node_id: str) -> str:
    return yaml_text(relationship_value(graph.node(node_id)))


def task_frontier(cursor: dict[str, Any]) -> list[dict[str, Any]]:
    nodes = node_decisions(cursor)
    code = code_decisions(cursor)
    return [
        {
            "kind": "node",
            "node": node_id,
            "status": "decided" if node_id in nodes else "pending",
            "reasons": cursor["node_reasons"].get(node_id, []),
        }
        for node_id in cursor["required_nodes"]
    ] + [
        {
            "kind": "code",
            "path": path,
            "status": "decided" if path in code else "pending",
            "reasons": cursor["code_reasons"].get(path, []),
        }
        for path in cursor["required_code"]
    ]


def base_blob(repository: Path, base: str, path: str) -> bytes | None:
    completed = subprocess.run(
        ["git", "show", f"{base}:{path}"],
        cwd=repository,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    return completed.stdout if completed.returncode == 0 else None
