#!/usr/bin/env python3
"""Persist one semantic decision and expand its bounded frontier."""

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
    ProjectionStore,
    require_approved_description_review_for_node,
)

from workflow import (
    database_path,
    decisions_by_node,
    json_error,
    native_graph,
    required_nodes,
    task_cursor,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", default=".")
    parser.add_argument("--database", default=".stratic/graph.sqlite3")
    parser.add_argument("--actor", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--node", required=True)
    parser.add_argument(
        "--outcome", choices=("revised", "confirmed_unchanged"), required=True
    )
    parser.add_argument("--abstraction-changed", action="store_true")
    parser.add_argument("--rationale", required=True)
    return parser


def decide(args: argparse.Namespace) -> dict[str, Any]:
    if not args.rationale.strip():
        raise OperationError("decision rationale must be non-empty")
    repository = Path(args.repository).resolve()
    database = database_path(repository, args.database)
    with ProjectionStore(database) as projection:
        operations = OperationalStore(connection=projection.connection)
        task = operations.get_task(args.task)
        if task["status"] != "claimed" or task["claimant"] != args.actor:
            raise OperationError("reconciliation task is not claimed by this actor")
        graph = native_graph(repository, operations, task)
        cursor = task_cursor(task)
        if args.node not in set(cursor["required"]):
            raise OperationError(
                f"node is outside the current bounded frontier: {args.node}"
            )
        if args.node not in set(cursor["reviewed"]):
            raise OperationError(f"inspect node before deciding it: {args.node}")
        if args.outcome == "confirmed_unchanged" and args.node in set(
            cursor.get("staged_nodes", [])
        ):
            raise OperationError("a node with staged native mutations must be revised")
        if args.outcome == "revised":
            changeset_id = cursor.get("changeset_id")
            if not isinstance(changeset_id, str):
                raise OperationError("reconciliation task has no native changeset")
            require_approved_description_review_for_node(
                repository,
                operations,
                task,
                operations.get_changeset(changeset_id),
                args.node,
            )
        node = graph.node(args.node)
        if args.abstraction_changed and node.parent is None:
            raise OperationError("the L0 root cannot escalate to a parent")

        decisions = decisions_by_node(cursor)
        decisions[args.node] = {
            "node": args.node,
            "outcome": args.outcome,
            "abstraction_changed": bool(args.abstraction_changed),
            "rationale": args.rationale.strip(),
        }
        previous_required = set(cursor["required"])
        required = required_nodes(graph, cursor["seeds"], decisions)
        decisions = {
            node_id: decision
            for node_id, decision in decisions.items()
            if node_id in required
        }
        cursor["decisions"] = [decisions[node_id] for node_id in sorted(decisions)]
        cursor["required"] = sorted(required)
        cursor["phase"] = "decide"
        task = operations.update_task(
            task["task_id"],
            expected_revision=task["revision"],
            actor=args.actor,
            cursor=cursor,
        )
    return {
        "added_frontier": sorted(required - previous_required),
        "remaining": sorted(required - set(decisions)),
        "task": task,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = decide(args)
    except (OSError, UnicodeError, FrontierError, OperationError) as error:
        print(json_error(error), file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
