"""Shared mechanics for the repository-owned reconciliation skill."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from pathlib import PurePosixPath
import subprocess
from typing import Any, Iterable

from stratic.workflow_api import (
    DescriptionGraph,
    CandidateGraphError,
    OperationError,
    apply_graph_mutations,
    canonical_json,
    load_description_graph,
    relationship_value,
    resolve_snapshot_code_anchors,
    snapshot_from_commit,
    worktree_code_snapshot,
    yaml_text,
)


class NativeDescriptionGraph:
    """Bounded graph view obtained by replaying a task's native mutations."""

    def __init__(
        self,
        repository: Path,
        config: Any,
        snapshot: Any,
        code_anchor_issues: tuple[Any, ...] = (),
    ) -> None:
        self.repository = repository
        self.config = config
        self.nodes = snapshot.nodes
        self._by_id = {node.node_id: node for node in snapshot.nodes}
        self.code_anchor_issues = code_anchor_issues

    def node(self, node_id: str) -> Any:
        try:
            return self._by_id[node_id]
        except KeyError as error:
            raise OperationError(
                f"description node does not exist: {node_id}"
            ) from error

    def relationship_path(self, node: Any) -> str | None:
        if self.config.relationships is None:
            return None
        source = PurePosixPath(node.source_path)
        descriptions = PurePosixPath(self.config.descriptions)
        try:
            relative = source.relative_to(descriptions)
        except ValueError as error:
            raise OperationError(
                "candidate description path is out of scope"
            ) from error
        return (
            PurePosixPath(self.config.relationships) / relative.with_suffix(".yaml")
        ).as_posix()

    def expansion_targets(self, node_id: str) -> tuple[str, ...]:
        node = self.node(node_id)
        targets = {node.parent} if node.parent is not None else set()
        for source in self.nodes:
            if any(
                reference.target == node_id and reference.kind != "informational"
                for reference in source.references
            ) or any(
                link.target_node_id == node_id and link.kind != "informational"
                for anchor in source.anchors
                for link in anchor.links
            ):
                targets.add(source.node_id)
        return tuple(sorted(target for target in targets if target is not None))


def native_graph(
    repository: Path,
    operations: Any,
    task: dict[str, Any],
) -> NativeDescriptionGraph:
    cursor = task_cursor(task)
    changeset_id = cursor.get("changeset_id")
    if not isinstance(changeset_id, str):
        raise OperationError("reconciliation task has no native changeset")
    changeset = operations.get_changeset(changeset_id)
    base = snapshot_from_commit(repository, str(task["base_version"]))
    accepted = load_description_graph(
        repository, revision=str(task["base_version"])
    )
    try:
        candidate = apply_graph_mutations(
            base,
            changeset["graph_mutations"],
            code=worktree_code_snapshot(repository, accepted.config),
        )
    except CandidateGraphError as worktree_error:
        try:
            candidate = apply_graph_mutations(
                base, changeset["graph_mutations"], code=base.code
            )
        except CandidateGraphError:
            raise worktree_error
    candidate, issues = resolve_snapshot_code_anchors(candidate, repository)
    return NativeDescriptionGraph(
        repository, accepted.config, candidate, code_anchor_issues=issues
    )


def git(repository: Path, arguments: list[str], *, accepted: set[int] = {0}) -> bytes:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=repository,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode not in accepted:
        message = completed.stderr.decode("utf-8", errors="replace").strip()
        raise OperationError(message or f"git {' '.join(arguments)} failed")
    return completed.stdout


def git_text(repository: Path, arguments: list[str]) -> str:
    return git(repository, arguments).decode("utf-8", errors="strict").strip()


def task_node(item: dict[str, Any]) -> str:
    value = item.get("node", item.get("node_id"))
    if not isinstance(value, str):
        raise OperationError("reconciliation frontier item has no node identity")
    return value


def direct_nodes(task: dict[str, Any]) -> list[str]:
    return sorted({task_node(item) for item in task["frontier"]})


def initial_cursor(origin: str, nodes: Iterable[str]) -> dict[str, Any]:
    required = sorted(set(nodes))
    return {
        "phase": "inspect",
        "origin": origin,
        "reviewed": [],
        "inspection_offsets": {},
        "seeds": required,
        "required": required,
        "decisions": [],
        "spent": 0,
        "context_bytes": 0,
    }


def task_cursor(task: dict[str, Any]) -> dict[str, Any]:
    cursor = task.get("cursor")
    if not isinstance(cursor, dict):
        raise OperationError("reconciliation task has no workflow cursor")
    required_fields = {
        "phase",
        "origin",
        "reviewed",
        "required",
        "decisions",
        "spent",
        "context_bytes",
    }
    if not required_fields <= set(cursor):
        raise OperationError("reconciliation task uses an unsupported cursor format")
    result = dict(cursor)
    offsets = result.setdefault("inspection_offsets", {})
    if not isinstance(offsets, dict) or not all(
        isinstance(node_id, str) and isinstance(offset, int) and offset >= 0
        for node_id, offset in offsets.items()
    ):
        raise OperationError("task inspection offsets are invalid")
    seeds = result.setdefault("seeds", list(result["required"]))
    if not isinstance(seeds, list) or not all(
        isinstance(node_id, str) for node_id in seeds
    ):
        raise OperationError("task reconciliation seeds are invalid")
    return result


def decisions_by_node(cursor: dict[str, Any]) -> dict[str, dict[str, Any]]:
    value = cursor.get("decisions")
    if not isinstance(value, list):
        raise OperationError("task decisions must be an array")
    decisions: dict[str, dict[str, Any]] = {}
    for item in value:
        if not isinstance(item, dict) or not isinstance(item.get("node"), str):
            raise OperationError("task contains an invalid decision")
        decisions[item["node"]] = item
    return decisions


def required_nodes(
    graph: DescriptionGraph | NativeDescriptionGraph,
    direct: Iterable[str],
    decisions: dict[str, dict[str, Any]],
) -> set[str]:
    required = set(direct)
    while True:
        expanded = set(required)
        for node_id in required:
            decision = decisions.get(node_id)
            if decision is not None and decision.get("abstraction_changed") is True:
                expanded.update(graph.expansion_targets(node_id))
        if expanded == required:
            return required
        required = expanded


def node_summary(
    graph: DescriptionGraph | NativeDescriptionGraph, node_id: str
) -> dict[str, Any]:
    node = graph.node(node_id)
    source_path = getattr(node, "path", getattr(node, "source_path", None))
    relationship_path = getattr(node, "relationship_path", None)
    if isinstance(graph, NativeDescriptionGraph):
        relationship_path = graph.relationship_path(node)
    return {
        "node_id": node_id,
        "level": node.level,
        "scope": node.scope,
        "realization": node.realization,
        "source_path": source_path,
        "relationship_path": relationship_path,
        "parent": node.parent,
    }


def relationship_text(graph: NativeDescriptionGraph, node_id: str) -> str:
    return yaml_text(relationship_value(graph.node(node_id)))


def changed_paths_for_node(task: dict[str, Any], node_id: str) -> list[str]:
    paths: set[str] = set()
    for item in task["frontier"]:
        if task_node(item) != node_id:
            continue
        value = item.get("changed_paths", [])
        if not isinstance(value, list) or not all(
            isinstance(path, str) for path in value
        ):
            raise OperationError(f"frontier paths are invalid for {node_id}")
        paths.update(value)
    return sorted(paths)


def task_changed_paths(task: dict[str, Any]) -> list[str]:
    return sorted(
        {
            path
            for node_id in direct_nodes(task)
            for path in changed_paths_for_node(task, node_id)
        }
    )


def patch_bytes(
    repository: Path,
    *,
    base: str,
    origin: str,
    paths: Iterable[str],
) -> bytes:
    selected = sorted(set(paths))
    if not selected:
        return b""
    if origin == "synchronized":
        parents = git_text(repository, ["show", "-s", "--format=%P", base]).split()
        if not parents:
            return b""
        return git(
            repository,
            ["diff", "--binary", "--no-ext-diff", parents[0], base, "--", *selected],
        )
    if origin != "worktree":
        raise OperationError(f"unsupported reconciliation origin {origin!r}")
    pieces = [
        git(
            repository,
            [
                "diff",
                "--binary",
                "--no-ext-diff",
                "--no-renames",
                base,
                "--",
                *selected,
            ],
        )
    ]
    untracked = {
        value
        for value in git(
            repository, ["ls-files", "--others", "--exclude-standard", "-z"]
        )
        .decode("utf-8", errors="surrogateescape")
        .split("\0")
        if value
    }
    for path in sorted(set(selected) & untracked):
        pieces.append(
            git(
                repository,
                ["diff", "--no-index", "--binary", "--", "/dev/null", path],
                accepted={0, 1},
            )
        )
    return b"".join(pieces)


def patch_summary(value: bytes) -> dict[str, Any]:
    return {
        "bytes": len(value),
        "sha256": sha256(value).hexdigest(),
    }


def context_size(value: dict[str, Any]) -> int:
    return len(canonical_json(value).encode("utf-8"))


def database_path(repository: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repository / path


def json_error(error: Exception) -> str:
    return json.dumps({"error": str(error)}, sort_keys=True, separators=(",", ":"))
