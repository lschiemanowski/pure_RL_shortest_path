#!/usr/bin/env python3
"""Create or resume a metadata-only reconciliation task."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Sequence

from stratic.workflow_api import (
    FrontierError,
    OperationError,
    OperationalStore,
    ProjectionError,
    ProjectionStore,
    discover_worktree_frontier,
    load_description_graph,
)

from workflow import (
    changed_paths_for_node,
    database_path,
    direct_nodes,
    git_text,
    initial_cursor,
    json_error,
    node_summary,
    patch_bytes,
    patch_summary,
    task_changed_paths,
    task_cursor,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", default=".")
    parser.add_argument("--database", default=".stratic/graph.sqlite3")
    parser.add_argument("--actor", required=True)
    parser.add_argument("--base", default="HEAD")
    parser.add_argument("--task")
    parser.add_argument("--budget", type=int, default=20)
    parser.add_argument(
        "--assign-uncovered",
        action="append",
        default=[],
        metavar="PATH=NODE",
        help="assign one deliberate new path to an existing L3 frontier node",
    )
    return parser


def _uncovered_owners(values: Sequence[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        path, separator, node = value.partition("=")
        if not separator or not path or not node:
            raise OperationError("--assign-uncovered must use PATH=NODE")
        if path in result:
            raise OperationError(f"duplicate uncovered assignment for {path}")
        result[path] = node
    return result


def _frontier_metadata(graph: Any, task: dict[str, Any]) -> list[dict[str, Any]]:
    result = []
    for node_id in direct_nodes(task):
        result.append(
            {
                **node_summary(graph, node_id),
                "changed_paths": changed_paths_for_node(task, node_id),
                "status": next(
                    item["status"]
                    for item in task["frontier"]
                    if item.get("node", item.get("node_id")) == node_id
                ),
            }
        )
    return result


def _begin_changeset(
    operations: OperationalStore,
    task: dict[str, Any],
    cursor: dict[str, Any],
    actor: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    changeset_id = cursor.get("changeset_id")
    if isinstance(changeset_id, str):
        changeset = operations.get_changeset(changeset_id)
        if changeset["base_version"] != task["base_version"]:
            raise OperationError("task changeset has the wrong base version")
        if changeset["status"] not in {"draft", "captured", "validated"}:
            raise OperationError(
                f"task changeset is not resumable: {changeset['status']}"
            )
        return task, changeset
    changeset = operations.begin_changeset(
        base_commit=str(task["base_version"]),
        base_version=str(task["base_version"]),
        actor=actor,
        rationale="Reconcile the task frontier through native graph mutations.",
    )
    cursor["changeset_id"] = changeset["changeset_id"]
    task = operations.update_task(
        task["task_id"],
        expected_revision=task["revision"],
        actor=actor,
        cursor=cursor,
    )
    return task, changeset


def _changed_description_nodes(graph: Any, paths: Sequence[str]) -> list[str]:
    changed = set(paths)
    return sorted(
        node.node_id
        for node in graph.nodes
        if node.node_id is not None
        and (
            node.path in changed
            or (
                node.relationship_path is not None and node.relationship_path in changed
            )
        )
    )


def _prepare_existing(
    repository: Path,
    database: Path,
    actor: str,
    task_id: str,
) -> dict[str, Any]:
    with ProjectionStore(database) as projection:
        operations = OperationalStore(connection=projection.connection)
        task = operations.get_task(task_id)
        if task["task_type"] != "reconciliation":
            raise OperationError(f"task {task_id} is not a reconciliation task")
        if task["status"] == "pending":
            task = operations.claim_task(
                task_id, expected_revision=task["revision"], actor=actor
            )
        elif task["status"] != "claimed" or task["claimant"] != actor:
            raise OperationError(f"task {task_id} is not resumable by {actor}")
        if task["cursor"] is None:
            task = operations.update_task(
                task_id,
                expected_revision=task["revision"],
                actor=actor,
                cursor=initial_cursor("synchronized", direct_nodes(task)),
            )
        else:
            task_cursor(task)
        projection.resolve_version(str(task["base_version"]))
        graph = load_description_graph(
            repository, revision=str(task["base_version"])
        )
        cursor = task_cursor(task)
        task, changeset = _begin_changeset(operations, task, cursor, actor)
        cursor = task_cursor(task)
        paths = task_changed_paths(task)
        if cursor["origin"] == "worktree":
            frontier = discover_worktree_frontier(
                repository,
                base=str(task["base_version"]),
                reject_managed_changes=False,
            )
            paths = list(frontier.changed_paths)
            seeds = set(direct_nodes(task))
            seeds.update(_changed_description_nodes(graph, paths))
            required = set(cursor["required"])
            required.update(seeds)
            if required != set(cursor["required"]) or seeds != set(cursor["seeds"]):
                cursor["seeds"] = sorted(seeds)
                cursor["required"] = sorted(required)
                task = operations.update_task(
                    task["task_id"],
                    expected_revision=task["revision"],
                    actor=actor,
                    cursor=cursor,
                )
    cursor = task_cursor(task)
    patch = patch_bytes(
        repository,
        base=str(task["base_version"]),
        origin=str(cursor["origin"]),
        paths=paths,
    )
    return {
        "base_commit": task["base_version"],
        "changed_paths": paths,
        "patch": patch_summary(patch),
        "task": task,
        "changeset": changeset,
        "frontier": _frontier_metadata(graph, task),
    }


def _prepare_worktree(
    args: argparse.Namespace, repository: Path, database: Path
) -> dict[str, Any]:
    frontier = discover_worktree_frontier(
        repository,
        base=args.base,
        uncovered_owners=_uncovered_owners(args.assign_uncovered),
    )
    if not frontier.items:
        raise OperationError("working tree has no changed covered code")
    with ProjectionStore(database) as projection:
        projection.resolve_version(frontier.base_commit)
        operations = OperationalStore(connection=projection.connection)
        active = [
            task
            for task in operations.list_tasks()
            if task["task_type"] == "reconciliation"
            and task["status"] in {"pending", "claimed"}
            and task["base_version"] == frontier.base_commit
        ]
        if active:
            raise OperationError(
                "active reconciliation task already exists: "
                + str(active[0]["task_id"])
            )
        task = operations.create_task(
            task_type="reconciliation",
            scope=repository.name,
            creator=args.actor,
            rationale="Reconcile covered working-tree changes.",
            base_version=frontier.base_commit,
            frontier=[
                {
                    "node": item.node_id,
                    "status": "pending",
                    "changed_paths": list(item.changed_paths),
                }
                for item in frontier.items
            ],
            budget=args.budget,
        )
        task = operations.claim_task(
            task["task_id"], expected_revision=task["revision"], actor=args.actor
        )
        task = operations.update_task(
            task["task_id"],
            expected_revision=task["revision"],
            actor=args.actor,
            cursor=initial_cursor(
                "worktree",
                [
                    *direct_nodes(task),
                    *_changed_description_nodes(frontier.graph, frontier.changed_paths),
                ],
            ),
        )
        task, changeset = _begin_changeset(
            operations, task, task_cursor(task), args.actor
        )
    patch = patch_bytes(
        repository,
        base=frontier.base_commit,
        origin="worktree",
        paths=frontier.changed_paths,
    )
    return {
        "base_commit": frontier.base_commit,
        "changed_paths": list(frontier.changed_paths),
        "patch": patch_summary(patch),
        "task": task,
        "changeset": changeset,
        "frontier": _frontier_metadata(frontier.graph, task),
    }


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    repository = Path(args.repository).resolve()
    database = database_path(repository, args.database)
    if args.task is not None:
        return _prepare_existing(repository, database, args.actor, args.task)
    git_text(repository, ["rev-parse", "--verify", "HEAD^{commit}"])
    return _prepare_worktree(args, repository, database)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = prepare(args)
    except (
        OSError,
        UnicodeError,
        FrontierError,
        OperationError,
        ProjectionError,
    ) as error:
        print(json_error(error), file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
