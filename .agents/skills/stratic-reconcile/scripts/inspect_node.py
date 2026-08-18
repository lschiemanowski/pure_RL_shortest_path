#!/usr/bin/env python3
"""Load one bounded node context window and persist inspection progress."""

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
)

from workflow import (
    changed_paths_for_node,
    context_size,
    database_path,
    json_error,
    native_graph,
    node_summary,
    patch_bytes,
    patch_summary,
    relationship_text,
    task_cursor,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", default=".")
    parser.add_argument("--database", default=".stratic/graph.sqlite3")
    parser.add_argument("--actor", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--node", required=True)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--max-patch-bytes", type=int, default=8_000)
    return parser


def _reference_context(
    graph: Any, node_id: str
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    node = graph.node(node_id)
    outgoing = [
        {"kind": reference.kind, "node": node_summary(graph, reference.target)}
        for reference in node.references
    ]
    incoming = [
        {"kind": reference.kind, "node": node_summary(graph, source.node_id)}
        for source in graph.nodes
        if source.node_id is not None
        for reference in source.references
        if reference.target == node_id
    ]

    def anchor_link_summary(link: Any) -> dict[str, Any]:
        if link.target_node_id is not None:
            return {
                "kind": link.kind,
                "node": node_summary(graph, link.target_node_id),
            }
        code: dict[str, Any] = {"path": link.target_path}
        if link.code_anchor is not None:
            anchor = link.code_anchor
            code["anchor"] = {
                "code_anchor_id": anchor.code_anchor_id,
                "quote": anchor.quote,
                "prefix": anchor.prefix,
                "suffix": anchor.suffix,
                "start": anchor.start,
                "end": anchor.end,
                "start_line": anchor.start_line,
                "start_column": anchor.start_column,
                "end_line": anchor.end_line,
                "end_column": anchor.end_column,
                "content_sha256": anchor.content_sha256,
            }
        return {"kind": link.kind, "code": code}

    outgoing_anchors = [
        {
            "anchor_id": anchor.anchor_id,
            "quote": anchor.quote,
            "prefix": anchor.prefix,
            "suffix": anchor.suffix,
            "links": [anchor_link_summary(link) for link in anchor.links],
        }
        for anchor in node.anchors
    ]
    incoming_anchors = [
        {
            "anchor_id": anchor.anchor_id,
            "quote": anchor.quote,
            "kind": link.kind,
            "node": node_summary(graph, source.node_id),
        }
        for source in graph.nodes
        if source.node_id is not None
        for anchor in source.anchors
        for link in anchor.links
        if link.target_node_id == node_id
    ]
    return outgoing, incoming, outgoing_anchors, incoming_anchors


def inspect(args: argparse.Namespace) -> dict[str, Any]:
    if args.offset < 0:
        raise OperationError("patch offset must be non-negative")
    if args.max_patch_bytes <= 0:
        raise OperationError("maximum patch bytes must be positive")
    repository = Path(args.repository).resolve()
    database = database_path(repository, args.database)
    with ProjectionStore(database) as projection:
        operations = OperationalStore(connection=projection.connection)
        task = operations.get_task(args.task)
        if task["status"] != "claimed" or task["claimant"] != args.actor:
            raise OperationError("reconciliation task is not claimed by this actor")
        graph = native_graph(repository, operations, task)
        cursor = task_cursor(task)
        required = set(cursor["required"])
        if args.node not in required:
            raise OperationError(
                f"node is outside the current bounded frontier: {args.node}"
            )
        reviewed = set(cursor["reviewed"])
        offsets = dict(cursor["inspection_offsets"])
        is_reviewed = args.node in reviewed
        expected_offset = int(offsets.get(args.node, 0))
        if not is_reviewed and args.offset != expected_offset:
            raise OperationError(
                f"next patch offset for {args.node} is {expected_offset}, "
                f"not {args.offset}"
            )
        is_new = not is_reviewed
        budget = task["budget"]
        if is_new and budget is not None and int(cursor["spent"]) >= int(budget):
            raise OperationError(
                f"task budget exhausted at {cursor['spent']} inspected node(s)"
            )

        node = graph.node(args.node)
        origin = str(cursor["origin"])
        base = str(task["base_version"])
        base_node = None
        try:
            base_node = projection.get_node(args.node, base)
        except ProjectionError:
            pass
        if origin == "synchronized" and base_node is None:
            raise OperationError(f"node {args.node} is absent at task version {base}")
        description = node.body
        relationship = relationship_text(graph, args.node)

        parent = None if node.parent is None else node_summary(graph, node.parent)
        outgoing, incoming, outgoing_anchors, incoming_anchors = _reference_context(
            graph, args.node
        )
        paths = changed_paths_for_node(task, args.node)
        if origin == "worktree":
            summary = node_summary(graph, args.node)
            paths = sorted({*paths, str(summary["source_path"])})
            if summary["relationship_path"] is not None:
                paths = sorted({*paths, str(summary["relationship_path"])})
        patch = patch_bytes(repository, base=base, origin=origin, paths=paths)
        start = min(args.offset, len(patch))
        end = min(start + args.max_patch_bytes, len(patch))
        complete = end == len(patch)
        include_structure = args.offset == 0
        context = {
            "node": node_summary(graph, args.node),
            "description": description if include_structure else None,
            "relationship": relationship if include_structure else None,
            "base_description": (
                None
                if not include_structure or base_node is None
                else base_node["body"]
            ),
            "parent": parent if include_structure else None,
            "outgoing_references": outgoing if include_structure else [],
            "incoming_references": incoming if include_structure else [],
                "outgoing_anchor_links": outgoing_anchors if include_structure else [],
                "incoming_anchor_links": incoming_anchors if include_structure else [],
                "unresolved_code_anchors": (
                    [
                        {
                            "anchor_id": issue.anchor_id,
                            "code_anchor_id": issue.code_anchor_id,
                            "path": issue.path,
                            "detail": issue.detail,
                        }
                        for issue in graph.code_anchor_issues
                        if issue.node_id == args.node
                    ]
                    if include_structure
                    else []
                ),
                "changed_paths": changed_paths_for_node(task, args.node),
            "patch": patch[start:end].decode("utf-8", errors="replace"),
            "patch_window": {
                **patch_summary(patch),
                "offset": start,
                "next_offset": None if complete else end,
                "complete": complete,
            },
        }
        size = context_size(context)
        if is_new:
            cursor["spent"] = int(cursor["spent"]) + 1
            cursor["context_bytes"] = int(cursor["context_bytes"]) + size
            if complete:
                reviewed.add(args.node)
                offsets.pop(args.node, None)
            else:
                offsets[args.node] = end
        if is_new:
            cursor["reviewed"] = sorted(reviewed)
            cursor["inspection_offsets"] = offsets
            cursor["phase"] = "inspect"
            task = operations.update_task(
                task["task_id"],
                expected_revision=task["revision"],
                actor=args.actor,
                cursor=cursor,
            )
    return {
        "context": context,
        "context_bytes": size,
        "next_offset": None if complete else end,
        "task": task,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = inspect(args)
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
