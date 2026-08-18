#!/usr/bin/env python3
"""Stage attributed native graph mutations for a reconciliation task."""

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
    load_description_graph,
    materialize_changeset,
    resolve_snapshot_code_anchors,
    snapshot_from_commit,
    validate_mutation_shape,
    worktree_code_snapshot,
)

from workflow import database_path, decisions_by_node, json_error, task_cursor


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", default=".")
    parser.add_argument("--database", default=".stratic/graph.sqlite3")
    parser.add_argument("--actor", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--rationale", required=True)
    mutation = parser.add_mutually_exclusive_group(required=True)
    mutation.add_argument("--mutation-json")
    mutation.add_argument("--mutation-file", type=Path)
    return parser


def _mutations(value: str) -> list[dict[str, Any]]:
    try:
        result = json.loads(value)
    except json.JSONDecodeError as error:
        raise OperationError(f"mutation-json is invalid JSON: {error}") from error
    mutations = result if isinstance(result, list) else [result]
    if not mutations or not all(isinstance(item, dict) for item in mutations):
        raise OperationError("mutation input must be an object or non-empty array")
    for mutation in mutations:
        try:
            validate_mutation_shape(mutation)
        except CandidateGraphError as error:
            raise OperationError(str(error)) from error
    return mutations


def _target(mutation: dict[str, Any]) -> str:
    operation = mutation["operation"]
    if operation == "description.create":
        raise OperationError("reconciliation tasks cannot create descriptions")
    if operation == "reconciliation.record":
        raise OperationError("record.py owns immutable reconciliation evidence")
    value = mutation.get("node_id", mutation.get("source_node_id"))
    if not isinstance(value, str):
        raise OperationError("native graph mutation has no source node")
    return value


def _touched_selectors(
    mutations: Sequence[dict[str, Any]],
) -> tuple[set[str], set[str]]:
    spans: set[str] = set()
    code_anchors: set[str] = set()
    for mutation in mutations:
        operation = mutation.get("operation")
        if operation == "span.set":
            span = mutation.get("span")
            if isinstance(span, dict) and isinstance(span.get("anchor_id"), str):
                spans.add(span["anchor_id"])
            links = span.get("links", []) if isinstance(span, dict) else []
        elif operation == "link.set":
            anchor_id = mutation.get("anchor_id")
            if isinstance(anchor_id, str):
                spans.add(anchor_id)
            links = [mutation.get("link")]
        else:
            links = []
        for link in links:
            if not isinstance(link, dict):
                continue
            code_anchor = link.get("code_anchor")
            if isinstance(code_anchor, dict) and isinstance(
                code_anchor.get("code_anchor_id"), str
            ):
                code_anchors.add(code_anchor["code_anchor_id"])
    return spans, code_anchors


def stage(args: argparse.Namespace) -> dict[str, Any]:
    if not args.rationale.strip():
        raise OperationError("mutation rationale must be non-empty")
    repository = Path(args.repository).resolve()
    database = database_path(repository, args.database)
    mutation_text = (
        args.mutation_file.read_text(encoding="utf-8")
        if args.mutation_file is not None
        else args.mutation_json
    )
    mutations = _mutations(mutation_text)
    node_ids = {_target(mutation) for mutation in mutations}
    with ProjectionStore(database) as projection:
        operations = OperationalStore(connection=projection.connection)
        task = operations.get_task(args.task)
        if task["status"] != "claimed" or task["claimant"] != args.actor:
            raise OperationError("reconciliation task is not claimed by this actor")
        cursor = task_cursor(task)
        recorded = cursor["phase"] == "changeset"
        outside = sorted(node_ids - set(cursor["required"]))
        if outside:
            raise OperationError(
                "nodes are outside the bounded frontier: " + ", ".join(outside)
            )
        unreviewed = sorted(node_ids - set(cursor["reviewed"]))
        if unreviewed:
            raise OperationError(
                "inspect nodes before staging: " + ", ".join(unreviewed)
            )
        decisions = decisions_by_node(cursor)
        contradictory = sorted(
            node_id
            for node_id in node_ids
            if (decision := decisions.get(node_id)) is not None
            and decision["outcome"] != "revised"
        )
        if contradictory:
            raise OperationError(
                "cannot mutate nodes confirmed unchanged: " + ", ".join(contradictory)
            )
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
                [*changeset["graph_mutations"], *mutations],
                code=worktree_code_snapshot(repository, accepted.config),
            )
            _, selector_issues = resolve_snapshot_code_anchors(
                candidate, repository
            )
        except CandidateGraphError as error:
            raise OperationError(
                f"native candidate graph is invalid: {error}"
            ) from error
        touched_spans, touched_code_anchors = _touched_selectors(mutations)
        blocking = [
            issue
            for issue in selector_issues
            if issue.anchor_id in touched_spans
            or issue.code_anchor_id in touched_code_anchors
        ]
        if blocking:
            raise OperationError(
                "staged code anchor repair is invalid: "
                + "; ".join(issue.render() for issue in blocking)
            )
        for mutation in mutations:
            changeset = operations.append_graph_mutation(
                changeset_id,
                expected_revision=changeset["revision"],
                expected_base_version=str(task["base_version"]),
                actor=args.actor,
                rationale=args.rationale.strip(),
                mutation=mutation,
            )
        staged = set(cursor.get("staged_nodes", []))
        staged.update(node_ids)
        cursor["staged_nodes"] = sorted(staged)
        materialization = None
        if recorded:
            materialization = materialize_changeset(
                operations,
                repository,
                changeset_id,
                expected_revision=changeset["revision"],
                expected_base_version=str(task["base_version"]),
            )
        else:
            cursor["phase"] = "stage"
        task = operations.update_task(
            task["task_id"],
            expected_revision=task["revision"],
            actor=args.actor,
            cursor=cursor,
        )
    return {
        "changeset": changeset,
        "mutation": mutations[0] if len(mutations) == 1 else None,
        "mutations": mutations,
        "materialization": materialization,
        "task": task,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = stage(args)
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
