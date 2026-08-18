#!/usr/bin/env python3
"""Validate persisted task decisions and write one reconciliation sidecar."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Sequence

from stratic.workflow_api import (
    AuthoritySerializationError,
    CandidateGraphError,
    FrontierError,
    OperationError,
    OperationalStore,
    ProjectionStore,
    SerializationError,
    apply_graph_mutations,
    check_impact,
    load_description_graph,
    materialize_changeset,
    require_resolved_code_anchors,
    require_approved_description_reviews,
    snapshot_from_commit,
    uuid7,
    worktree_code_snapshot,
)

from workflow import (
    database_path,
    decisions_by_node,
    direct_nodes,
    json_error,
    task_cursor,
    task_node,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", default=".")
    parser.add_argument("--database", default=".stratic/graph.sqlite3")
    parser.add_argument("--actor", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--rationale", required=True)
    parser.add_argument(
        "--direction", choices=("code-led", "spec-led"), default="code-led"
    )
    return parser


def record(args: argparse.Namespace) -> dict[str, Any]:
    if not args.rationale.strip():
        raise OperationError("record rationale must be non-empty")
    repository = Path(args.repository).resolve()
    database = database_path(repository, args.database)
    with ProjectionStore(database) as projection:
        operations = OperationalStore(connection=projection.connection)
        task = operations.get_task(args.task)
        if task["status"] != "claimed" or task["claimant"] != args.actor:
            raise OperationError("reconciliation task is not claimed by this actor")
        graph = load_description_graph(
            repository, revision=str(task["base_version"])
        )
        cursor = task_cursor(task)
        decisions = decisions_by_node(cursor)
        required = set(cursor["required"])
        missing = sorted(required - set(decisions))
        if missing:
            raise OperationError("missing required decisions: " + ", ".join(missing))
        if not set(direct_nodes(task)) <= set(decisions):
            raise OperationError("direct frontier is not fully decided")

        staged = set(cursor.get("staged_nodes", []))
        revised = {
            node_id
            for node_id, decision in decisions.items()
            if decision["outcome"] == "revised"
        }
        unstaged = sorted(revised - staged)
        if unstaged:
            raise OperationError(
                "revised nodes have no native mutation: " + ", ".join(unstaged)
            )
        undecided_mutations = sorted(staged - revised)
        if undecided_mutations:
            raise OperationError(
                "staged native mutations lack revised decisions: "
                + ", ".join(undecided_mutations)
            )
        changeset_id = cursor.get("changeset_id")
        if not isinstance(changeset_id, str):
            raise OperationError("reconciliation task has no native changeset")
        changeset = operations.get_changeset(changeset_id)
        require_approved_description_reviews(
            repository, operations, task, changeset
        )
        record_id = uuid7()
        relative = Path(graph.config.reconciliations) / f"{record_id}.yaml"
        payload = {
            "id": record_id,
            "direction": args.direction,
            "rationale": args.rationale.strip(),
            "decisions": [decisions[node_id] for node_id in sorted(required)],
        }
        mutation = {
            "operation": "reconciliation.record",
            "record": {
                "record_id": record_id,
                "direction": args.direction,
                "rationale": args.rationale.strip(),
                "source_path": relative.as_posix(),
                "payload": payload,
            },
        }
        base = snapshot_from_commit(repository, str(task["base_version"]))
        try:
            candidate = apply_graph_mutations(
                base,
                [*changeset["graph_mutations"], mutation],
                code=worktree_code_snapshot(repository, graph.config),
            )
            require_resolved_code_anchors(candidate, repository)
        except CandidateGraphError as error:
            raise OperationError(
                f"native candidate graph is invalid: {error}"
            ) from error
        changeset = operations.append_graph_mutation(
            changeset_id,
            expected_revision=changeset["revision"],
            expected_base_version=str(task["base_version"]),
            actor=args.actor,
            rationale=args.rationale.strip(),
            mutation=mutation,
        )
        materialization = materialize_changeset(
            operations,
            repository,
            changeset_id,
            expected_revision=changeset["revision"],
            expected_base_version=str(task["base_version"]),
        )
        impact = check_impact(repository, base=str(task["base_version"]))
        if not impact.valid:
            raise OperationError(
                "impact validation failed: "
                + "; ".join(issue.render() for issue in impact.issues)
            )
        frontier = [
            {
                **item,
                "status": "resolved",
                "outcome": decisions[task_node(item)]["outcome"],
            }
            for item in task["frontier"]
        ]
        cursor["phase"] = "changeset"
        cursor["record"] = relative.as_posix()
        task = operations.update_task(
            task["task_id"],
            expected_revision=task["revision"],
            actor=args.actor,
            frontier=frontier,
            cursor=cursor,
        )
    return {
        "changeset": changeset,
        "materialization": materialization,
        "record": relative.as_posix(),
        "task": task,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = record(args)
    except (
        OSError,
        UnicodeError,
        AuthoritySerializationError,
        FrontierError,
        OperationError,
        SerializationError,
    ) as error:
        print(json_error(error), file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
