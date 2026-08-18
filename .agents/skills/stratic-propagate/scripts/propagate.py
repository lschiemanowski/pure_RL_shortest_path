#!/usr/bin/env python3
"""Run Stratic's bounded, task-backed description-to-code propagation workflow."""

from __future__ import annotations

import argparse
import base64
from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
import sys
from typing import Any, Sequence

from stratic.workflow_api import (
    AuthoritySerializationError,
    CandidateGraphError,
    GENERATED_STATE_PATH,
    FrontierError,
    OperationError,
    OperationalStore,
    ProjectionError,
    ProjectionStore,
    SerializationError,
    apply_graph_mutations,
    canonical_json,
    check_impact,
    materialize_changeset,
    require_approved_description_review_for_node,
    require_approved_description_reviews,
    snapshot_from_commit,
    uuid7,
    validate_mutation_shape,
    worktree_snapshot,
)

from workflow import (
    base_blob,
    candidate_graph,
    changed_paths,
    code_decisions,
    compute_plan,
    database_path,
    git_text,
    node_decisions,
    node_summary,
    relationship_text,
    task_cursor,
    task_frontier,
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--repository", default=".")
    result.add_argument("--database", default=".stratic/graph.sqlite3")
    result.add_argument("--actor", required=True)
    commands = result.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare")
    prepare.add_argument("--base", default="HEAD")
    prepare.add_argument("--task")
    prepare.add_argument("--node")
    request = prepare.add_mutually_exclusive_group()
    request.add_argument("--request")
    request.add_argument("--request-file", type=Path)
    prepare.add_argument("--budget", type=int, default=30)

    inspect_node = commands.add_parser("inspect-node")
    inspect_node.add_argument("--task", required=True)
    inspect_node.add_argument("--node", required=True)
    inspect_node.add_argument("--max-context-bytes", type=int, default=32_000)

    stage_graph = commands.add_parser("stage-graph")
    stage_graph.add_argument("--task", required=True)
    stage_graph.add_argument("--rationale", required=True)
    graph_input = stage_graph.add_mutually_exclusive_group(required=True)
    graph_input.add_argument("--mutation-json")
    graph_input.add_argument("--mutation-file", type=Path)

    decide_node = commands.add_parser("decide-node")
    decide_node.add_argument("--task", required=True)
    decide_node.add_argument("--node", required=True)
    decide_node.add_argument(
        "--outcome", choices=("revised", "confirmed_unchanged", "pruned"), required=True
    )
    decide_node.add_argument("--propagate", action="store_true")
    decide_node.add_argument("--rationale", required=True)

    inspect_code = commands.add_parser("inspect-code")
    inspect_code.add_argument("--task", required=True)
    inspect_code.add_argument("--path", required=True)
    inspect_code.add_argument("--offset", type=int, default=0)
    inspect_code.add_argument("--max-bytes", type=int, default=16_000)

    stage_code = commands.add_parser("stage-code")
    stage_code.add_argument("--task", required=True)
    stage_code.add_argument("--path", required=True)
    stage_code.add_argument("--rationale", required=True)
    code_input = stage_code.add_mutually_exclusive_group(required=True)
    code_input.add_argument("--content-file", type=Path)
    code_input.add_argument("--from-worktree", action="store_true")
    code_input.add_argument("--delete", action="store_true")

    decide_code = commands.add_parser("decide-code")
    decide_code.add_argument("--task", required=True)
    decide_code.add_argument("--path", required=True)
    decide_code.add_argument(
        "--outcome", choices=("revised", "confirmed_unchanged"), required=True
    )
    decide_code.add_argument("--rationale", required=True)

    note = commands.add_parser("note-uncertainty")
    note.add_argument("--task", required=True)
    note.add_argument("--summary", required=True)
    note.add_argument("--node")
    note.add_argument("--path")

    record = commands.add_parser("record")
    record.add_argument("--task", required=True)
    record.add_argument("--rationale", required=True)
    return result


def _repository(args: argparse.Namespace) -> tuple[Path, Path]:
    repository = Path(args.repository).resolve()
    return repository, database_path(repository, args.database)


def _claimed(operations: OperationalStore, task_id: str, actor: str) -> dict[str, Any]:
    task = operations.get_task(task_id)
    if task["task_type"] != "propagation":
        raise OperationError(f"task {task_id} is not a propagation task")
    if task["status"] != "claimed" or task["claimant"] != actor:
        raise OperationError("propagation task is not claimed by this actor")
    return task


def _request(args: argparse.Namespace) -> str | None:
    if args.request_file is not None:
        return args.request_file.read_text(encoding="utf-8").strip()
    return None if args.request is None else args.request.strip()


def _managed_path(path: str, config: Any) -> bool:
    roots = [config.descriptions, config.reconciliations]
    if config.relationships is not None:
        roots.append(config.relationships)
    return path == GENERATED_STATE_PATH or any(
        path == root or path.startswith(root.rstrip("/") + "/") for root in roots
    )


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    repository, database = _repository(args)
    with ProjectionStore(database) as projection:
        operations = OperationalStore(connection=projection.connection)
        if args.task is not None:
            task = operations.get_task(args.task)
            if task["task_type"] != "propagation":
                raise OperationError("requested task is not a propagation task")
            if task["status"] == "pending":
                task = operations.claim_task(
                    task["task_id"],
                    expected_revision=task["revision"],
                    actor=args.actor,
                )
            elif task["status"] != "claimed" or task["claimant"] != args.actor:
                raise OperationError("propagation task is not resumable by this actor")
            cursor = task_cursor(task)
            graph = candidate_graph(repository, operations, task)
            return {
                "task": task,
                "changeset": operations.get_changeset(cursor["changeset_id"]),
                "root": node_summary(graph, cursor["root_node"]),
                "frontier": task_frontier(cursor),
            }

        request = _request(args)
        if not args.node or not request:
            raise OperationError(
                "new propagation requires --node and a non-empty request"
            )
        if args.budget < 0:
            raise OperationError("budget must be non-negative")
        base = git_text(
            repository, ["rev-parse", "--verify", f"{args.base}^{{commit}}"]
        )
        projection.resolve_version(base)
        base_snapshot = snapshot_from_commit(repository, base)
        _, config = worktree_snapshot(repository, template=base_snapshot)
        if args.node not in {node.node_id for node in base_snapshot.nodes}:
            raise OperationError(f"description node does not exist: {args.node}")
        dirty_graph = sorted(
            path
            for path in changed_paths(repository, base)
            if _managed_path(path, config)
        )
        if dirty_graph:
            raise OperationError(
                "managed graph must be materialized from the new propagation task: "
                + ", ".join(dirty_graph)
            )
        active = [
            task
            for task in operations.list_tasks()
            if task["task_type"] == "propagation"
            and task["status"] in {"pending", "claimed"}
            and task["base_version"] == base
        ]
        if active:
            raise OperationError(
                "active propagation task already exists: " + str(active[0]["task_id"])
            )
        task = operations.create_task(
            task_type="propagation",
            scope=repository.name,
            creator=args.actor,
            rationale=request,
            base_version=base,
            frontier=[
                {
                    "kind": "node",
                    "node": args.node,
                    "status": "pending",
                    "reasons": ["requested"],
                }
            ],
            budget=args.budget,
        )
        task = operations.claim_task(
            task["task_id"], expected_revision=task["revision"], actor=args.actor
        )
        changeset = operations.begin_changeset(
            base_commit=base,
            base_version=base,
            actor=args.actor,
            rationale=request,
        )
        cursor = {
            "phase": "inspect",
            "root_node": args.node,
            "request": request,
            "required_nodes": [args.node],
            "required_code": [],
            "reviewed_nodes": [],
            "reviewed_code": [],
            "node_decisions": [],
            "code_decisions": [],
            "staged_nodes": [],
            "staged_paths": [],
            "node_reasons": {args.node: ["requested"]},
            "code_reasons": {},
            "code_offsets": {},
            "uncertainties": [],
            "spent": 0,
            "context_bytes": 0,
            "changeset_id": changeset["changeset_id"],
            "initial_changed_paths": sorted(changed_paths(repository, base)),
        }
        task = operations.update_task(
            task["task_id"],
            expected_revision=task["revision"],
            actor=args.actor,
            cursor=cursor,
        )
        graph = candidate_graph(repository, operations, task)
    return {
        "task": task,
        "changeset": changeset,
        "root": node_summary(graph, args.node),
        "candidate_expansions": graph.expansions(args.node),
        "initial_changed_paths": cursor["initial_changed_paths"],
        "frontier": task_frontier(cursor),
    }


def inspect_node(args: argparse.Namespace) -> dict[str, Any]:
    if args.max_context_bytes <= 0:
        raise OperationError("maximum context bytes must be positive")
    repository, database = _repository(args)
    with ProjectionStore(database) as projection:
        operations = OperationalStore(connection=projection.connection)
        task = _claimed(operations, args.task, args.actor)
        cursor = task_cursor(task)
        if args.node not in set(cursor["required_nodes"]):
            raise OperationError("node is outside the bounded propagation frontier")
        graph = candidate_graph(repository, operations, task)
        node = graph.node(args.node)
        context = {
            "request": cursor["request"],
            "node": node_summary(graph, args.node),
            "description": node.body,
            "relationship": relationship_text(graph, args.node),
            "candidate_expansions": [
                {"node": node_summary(graph, target), "reasons": list(reasons)}
                for target, reasons in graph.expansions(args.node).items()
            ],
            "candidate_code": list(graph.code_paths(args.node)),
            "reasons": cursor["node_reasons"].get(args.node, []),
        }
        size = len(canonical_json(context).encode("utf-8"))
        if size > args.max_context_bytes:
            raise OperationError(
                f"node context is {size} bytes; raise --max-context-bytes explicitly"
            )
        reviewed = set(cursor["reviewed_nodes"])
        if args.node not in reviewed:
            if task["budget"] is not None and cursor["spent"] >= task["budget"]:
                raise OperationError("propagation task budget is exhausted")
            reviewed.add(args.node)
            cursor["reviewed_nodes"] = sorted(reviewed)
            cursor["spent"] += 1
            cursor["context_bytes"] += size
            cursor["phase"] = "inspect"
            task = operations.update_task(
                task["task_id"],
                expected_revision=task["revision"],
                actor=args.actor,
                cursor=cursor,
            )
    return {"context": context, "context_bytes": size, "task": task}


def _graph_mutations(args: argparse.Namespace) -> list[dict[str, Any]]:
    text = (
        args.mutation_file.read_text(encoding="utf-8")
        if args.mutation_file is not None
        else args.mutation_json
    )
    try:
        value = json.loads(text)
    except json.JSONDecodeError as error:
        raise OperationError(f"mutation input is invalid JSON: {error}") from error
    mutations = value if isinstance(value, list) else [value]
    if not mutations or not all(isinstance(item, dict) for item in mutations):
        raise OperationError("mutation input must be an object or non-empty array")
    for mutation in mutations:
        try:
            validate_mutation_shape(mutation)
        except CandidateGraphError as error:
            raise OperationError(str(error)) from error
        if mutation["operation"] == "reconciliation.record":
            raise OperationError("record owns immutable reconciliation evidence")
    return mutations


def _graph_target(mutation: dict[str, Any]) -> tuple[str, str | None]:
    if mutation["operation"] == "description.create":
        node = mutation.get("node")
        if not isinstance(node, dict):
            raise OperationError("description.create node is invalid")
        node_id = node.get("node_id")
        parent = node.get("parent")
        if not isinstance(node_id, str) or not isinstance(parent, str):
            raise OperationError("created description needs node_id and parent")
        return node_id, parent
    value = mutation.get("node_id", mutation.get("source_node_id"))
    if not isinstance(value, str):
        raise OperationError("graph mutation has no source node")
    return value, None


def stage_graph(args: argparse.Namespace) -> dict[str, Any]:
    if not args.rationale.strip():
        raise OperationError("mutation rationale must be non-empty")
    repository, database = _repository(args)
    mutations = _graph_mutations(args)
    with ProjectionStore(database) as projection:
        operations = OperationalStore(connection=projection.connection)
        task = _claimed(operations, args.task, args.actor)
        cursor = task_cursor(task)
        required = set(cursor["required_nodes"])
        reviewed = set(cursor["reviewed_nodes"])
        targets: set[str] = set()
        for mutation in mutations:
            target, parent = _graph_target(mutation)
            targets.add(target)
            authorized = parent if parent is not None else target
            if authorized not in required:
                raise OperationError(
                    f"graph mutation is outside the bounded frontier: {authorized}"
                )
            if authorized not in reviewed:
                raise OperationError(f"inspect node before staging: {authorized}")
        decisions = node_decisions(cursor)
        contradictory = sorted(
            target
            for target in targets
            if (decision := decisions.get(target)) is not None
            and decision["outcome"] != "revised"
        )
        if contradictory:
            raise OperationError(
                "cannot mutate nodes not decided revised: " + ", ".join(contradictory)
            )
        changeset = operations.get_changeset(cursor["changeset_id"])
        base = snapshot_from_commit(repository, str(task["base_version"]))
        current, _ = worktree_snapshot(repository, template=base)
        try:
            apply_graph_mutations(
                base,
                [*changeset["graph_mutations"], *mutations],
                code=current.code,
            )
        except CandidateGraphError as error:
            try:
                apply_graph_mutations(
                    base,
                    [*changeset["graph_mutations"], *mutations],
                    code=base.code,
                )
            except CandidateGraphError:
                raise OperationError(
                    f"native candidate graph is invalid: {error}"
                ) from error
        for mutation in mutations:
            changeset = operations.append_graph_mutation(
                cursor["changeset_id"],
                expected_revision=changeset["revision"],
                expected_base_version=str(task["base_version"]),
                actor=args.actor,
                rationale=args.rationale.strip(),
                mutation=mutation,
            )
        cursor["staged_nodes"] = sorted(set(cursor["staged_nodes"]) | targets)
        materialization = None
        if cursor["phase"] == "changeset":
            materialization = materialize_changeset(
                operations,
                repository,
                cursor["changeset_id"],
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
        "changeset_revision": changeset["revision"],
        "mutations": mutations,
        "materialization": materialization,
        "task": task,
    }


def decide_node(args: argparse.Namespace) -> dict[str, Any]:
    if not args.rationale.strip():
        raise OperationError("decision rationale must be non-empty")
    repository, database = _repository(args)
    with ProjectionStore(database) as projection:
        operations = OperationalStore(connection=projection.connection)
        task = _claimed(operations, args.task, args.actor)
        cursor = task_cursor(task)
        if args.node not in set(cursor["required_nodes"]):
            raise OperationError("node is outside the bounded propagation frontier")
        if args.node not in set(cursor["reviewed_nodes"]):
            raise OperationError("inspect node before deciding it")
        if args.outcome == "pruned" and args.node == cursor["root_node"]:
            raise OperationError("the requested root cannot be pruned")
        if args.outcome == "pruned" and args.propagate:
            raise OperationError("a pruned branch cannot propagate")
        staged = set(cursor["staged_nodes"])
        if args.outcome == "revised" and args.node not in staged:
            raise OperationError("a revised node requires a staged native mutation")
        if args.outcome == "revised":
            require_approved_description_review_for_node(
                repository,
                operations,
                task,
                operations.get_changeset(cursor["changeset_id"]),
                args.node,
            )
        if args.outcome != "revised" and args.node in staged:
            raise OperationError("a staged node must be decided revised")
        decisions = node_decisions(cursor)
        decisions[args.node] = {
            "node": args.node,
            "outcome": args.outcome,
            "propagate": bool(args.propagate),
            "rationale": args.rationale.strip(),
        }
        previous_nodes = set(cursor["required_nodes"])
        graph = candidate_graph(repository, operations, task)
        nodes, code, node_reasons, code_reasons = compute_plan(
            graph, cursor["root_node"], decisions
        )
        orphaned_nodes = sorted(staged - nodes)
        orphaned_code = sorted(set(cursor["staged_paths"]) - code)
        if orphaned_nodes or orphaned_code:
            raise OperationError(
                "decision would orphan staged mutations: "
                + ", ".join([*orphaned_nodes, *orphaned_code])
            )
        decisions = {key: value for key, value in decisions.items() if key in nodes}
        existing_code = {
            key: value for key, value in code_decisions(cursor).items() if key in code
        }
        cursor["node_decisions"] = [decisions[key] for key in sorted(decisions)]
        cursor["code_decisions"] = [existing_code[key] for key in sorted(existing_code)]
        cursor["required_nodes"] = sorted(nodes)
        cursor["required_code"] = sorted(code)
        cursor["node_reasons"] = {
            key: sorted(values) for key, values in sorted(node_reasons.items())
        }
        cursor["code_reasons"] = {
            key: sorted(values) for key, values in sorted(code_reasons.items())
        }
        cursor["phase"] = "plan"
        frontier = task_frontier(cursor)
        task = operations.update_task(
            task["task_id"],
            expected_revision=task["revision"],
            actor=args.actor,
            cursor=cursor,
            frontier=frontier,
        )
    return {
        "added_nodes": sorted(nodes - previous_nodes),
        "remaining_nodes": sorted(nodes - set(decisions)),
        "remaining_code": sorted(code - set(existing_code)),
        "frontier": frontier,
        "task": task,
    }


def inspect_code(args: argparse.Namespace) -> dict[str, Any]:
    if args.offset < 0 or args.max_bytes <= 0:
        raise OperationError("code offset and maximum bytes are invalid")
    repository, database = _repository(args)
    with ProjectionStore(database) as projection:
        operations = OperationalStore(connection=projection.connection)
        task = _claimed(operations, args.task, args.actor)
        cursor = task_cursor(task)
        if args.path not in set(cursor["required_code"]):
            raise OperationError(
                "code path is outside the bounded propagation frontier"
            )
        expected = int(cursor["code_offsets"].get(args.path, 0))
        reviewed = set(cursor["reviewed_code"])
        if args.path not in reviewed and args.offset != expected:
            raise OperationError(f"next code offset is {expected}, not {args.offset}")
        path = repository / args.path
        content = path.read_bytes() if path.is_file() else b""
        start = min(args.offset, len(content))
        end = min(start + args.max_bytes, len(content))
        complete = end == len(content)
        context = {
            "request": cursor["request"],
            "path": args.path,
            "reasons": cursor["code_reasons"].get(args.path, []),
            "base_sha256": (
                None
                if (base := base_blob(repository, task["base_version"], args.path))
                is None
                else sha256(base).hexdigest()
            ),
            "current_sha256": sha256(content).hexdigest() if path.is_file() else None,
            "content": content[start:end].decode("utf-8", errors="replace"),
            "offset": start,
            "next_offset": None if complete else end,
            "complete": complete,
        }
        size = len(canonical_json(context).encode("utf-8"))
        if args.path not in reviewed:
            if task["budget"] is not None and cursor["spent"] >= task["budget"]:
                raise OperationError("propagation task budget is exhausted")
            cursor["spent"] += 1
            cursor["context_bytes"] += size
            if complete:
                reviewed.add(args.path)
                cursor["code_offsets"].pop(args.path, None)
            else:
                cursor["code_offsets"][args.path] = end
            cursor["reviewed_code"] = sorted(reviewed)
            cursor["phase"] = "inspect"
            task = operations.update_task(
                task["task_id"],
                expected_revision=task["revision"],
                actor=args.actor,
                cursor=cursor,
            )
    return {"context": context, "context_bytes": size, "task": task}


def _safe_code_path(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise OperationError(f"unsafe code path {value!r}")
    return path


def stage_code(args: argparse.Namespace) -> dict[str, Any]:
    if not args.rationale.strip():
        raise OperationError("code mutation rationale must be non-empty")
    repository, database = _repository(args)
    _safe_code_path(args.path)
    with ProjectionStore(database) as projection:
        operations = OperationalStore(connection=projection.connection)
        task = _claimed(operations, args.task, args.actor)
        cursor = task_cursor(task)
        if args.path not in set(cursor["required_code"]):
            raise OperationError(
                "code path is outside the bounded propagation frontier"
            )
        if args.path not in set(cursor["reviewed_code"]):
            raise OperationError("inspect code before staging it")
        decision = code_decisions(cursor).get(args.path)
        if decision is not None and decision["outcome"] != "revised":
            raise OperationError("cannot mutate code confirmed unchanged")
        path = repository / args.path
        original = path.read_bytes() if path.is_file() else None
        original_mode = path.stat().st_mode if path.is_file() else None
        if args.delete:
            desired = None
        elif args.from_worktree:
            if not path.is_file():
                raise OperationError("worktree source is not a regular file")
            desired = path.read_bytes()
        else:
            desired = args.content_file.read_bytes()
        base = base_blob(repository, task["base_version"], args.path)
        if base is None and desired is None:
            raise OperationError("cannot delete a path absent from the base")
        mode = None
        if desired is not None:
            mode = (
                "100755" if path.is_file() and path.stat().st_mode & 0o111 else "100644"
            )
        mutation = {
            "path": args.path,
            "operation": "delete" if desired is None else "write",
            "base_sha256": None if base is None else sha256(base).hexdigest(),
            "content_sha256": None if desired is None else sha256(desired).hexdigest(),
            "mode": mode,
            "content_base64": (
                None if desired is None else base64.b64encode(desired).decode("ascii")
            ),
        }
        if not args.from_worktree:
            if desired is None:
                path.unlink(missing_ok=True)
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(desired)
                path.chmod(0o755 if mode == "100755" else 0o644)
        changeset = operations.get_changeset(cursor["changeset_id"])
        try:
            changeset = operations.stage_file_mutation(
                cursor["changeset_id"],
                expected_revision=changeset["revision"],
                expected_base_version=str(task["base_version"]),
                actor=args.actor,
                rationale=args.rationale.strip(),
                mutation=mutation,
            )
        except Exception:
            if not args.from_worktree:
                if original is None:
                    path.unlink(missing_ok=True)
                else:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(original)
                    assert original_mode is not None
                    path.chmod(original_mode)
            raise
        cursor["staged_paths"] = sorted(set(cursor["staged_paths"]) | {args.path})
        if cursor["phase"] != "changeset":
            cursor["phase"] = "stage"
        task = operations.update_task(
            task["task_id"],
            expected_revision=task["revision"],
            actor=args.actor,
            cursor=cursor,
        )
    return {
        "mutation": {
            key: value for key, value in mutation.items() if key != "content_base64"
        },
        "changeset_revision": changeset["revision"],
        "task": task,
    }


def decide_code(args: argparse.Namespace) -> dict[str, Any]:
    if not args.rationale.strip():
        raise OperationError("code decision rationale must be non-empty")
    repository, database = _repository(args)
    with ProjectionStore(database) as projection:
        operations = OperationalStore(connection=projection.connection)
        task = _claimed(operations, args.task, args.actor)
        cursor = task_cursor(task)
        if args.path not in set(cursor["required_code"]):
            raise OperationError(
                "code path is outside the bounded propagation frontier"
            )
        if args.path not in set(cursor["reviewed_code"]):
            raise OperationError("inspect code before deciding it")
        staged = args.path in set(cursor["staged_paths"])
        if args.outcome == "revised" and not staged:
            raise OperationError("revised code requires an attributed staged mutation")
        if args.outcome != "revised" and staged:
            raise OperationError("staged code must be decided revised")
        decisions = code_decisions(cursor)
        decisions[args.path] = {
            "path": args.path,
            "outcome": args.outcome,
            "rationale": args.rationale.strip(),
        }
        cursor["code_decisions"] = [decisions[key] for key in sorted(decisions)]
        cursor["phase"] = "decide"
        frontier = task_frontier(cursor)
        task = operations.update_task(
            task["task_id"],
            expected_revision=task["revision"],
            actor=args.actor,
            cursor=cursor,
            frontier=frontier,
        )
    return {
        "remaining_code": sorted(set(cursor["required_code"]) - set(decisions)),
        "task": task,
    }


def note_uncertainty(args: argparse.Namespace) -> dict[str, Any]:
    if not args.summary.strip():
        raise OperationError("uncertainty summary must be non-empty")
    repository, database = _repository(args)
    with ProjectionStore(database) as projection:
        operations = OperationalStore(connection=projection.connection)
        task = _claimed(operations, args.task, args.actor)
        cursor = task_cursor(task)
        uncertainty = {"summary": args.summary.strip()}
        if args.node is not None:
            if args.node not in set(cursor["required_nodes"]):
                raise OperationError("uncertainty node is outside the frontier")
            uncertainty["node"] = args.node
        if args.path is not None:
            if args.path not in set(cursor["required_code"]):
                raise OperationError("uncertainty path is outside the frontier")
            uncertainty["path"] = args.path
        changeset = operations.get_changeset(cursor["changeset_id"])
        changeset = operations.append_uncertainty(
            cursor["changeset_id"],
            expected_revision=changeset["revision"],
            expected_base_version=str(task["base_version"]),
            actor=args.actor,
            uncertainty=uncertainty,
        )
        cursor["uncertainties"] = [
            *cursor["uncertainties"],
            {**uncertainty, "actor": args.actor},
        ]
        task = operations.update_task(
            task["task_id"],
            expected_revision=task["revision"],
            actor=args.actor,
            cursor=cursor,
        )
    return {
        "changeset_revision": changeset["revision"],
        "uncertainty": uncertainty,
        "task": task,
    }


def _scope_issues(
    repository: Path, base: str, staged_paths: set[str], config: Any
) -> list[str]:
    return sorted(
        path
        for path in changed_paths(repository, base)
        if path not in staged_paths and not _managed_path(path, config)
    )


def record(args: argparse.Namespace) -> dict[str, Any]:
    if not args.rationale.strip():
        raise OperationError("record rationale must be non-empty")
    repository, database = _repository(args)
    with ProjectionStore(database) as projection:
        operations = OperationalStore(connection=projection.connection)
        task = _claimed(operations, args.task, args.actor)
        base = snapshot_from_commit(repository, str(task["base_version"]))
        _, config = worktree_snapshot(repository, template=base)
        cursor = task_cursor(task)
        nodes = node_decisions(cursor)
        code = code_decisions(cursor)
        missing_nodes = sorted(set(cursor["required_nodes"]) - set(nodes))
        missing_code = sorted(set(cursor["required_code"]) - set(code))
        if missing_nodes or missing_code:
            raise OperationError(
                "propagation decisions are incomplete: "
                + ", ".join([*missing_nodes, *missing_code])
            )
        revised_nodes = {
            node_id
            for node_id, decision in nodes.items()
            if decision["outcome"] == "revised"
        }
        revised_code = {
            path for path, decision in code.items() if decision["outcome"] == "revised"
        }
        if revised_nodes != set(cursor["staged_nodes"]):
            raise OperationError(
                "staged graph mutations do not match revised node decisions"
            )
        if revised_code != set(cursor["staged_paths"]):
            raise OperationError(
                "staged file mutations do not match revised code decisions"
            )
        outside = _scope_issues(
            repository,
            str(task["base_version"]),
            set(cursor["staged_paths"]),
            config,
        )
        if outside:
            raise OperationError(
                "changed paths are outside the impact plan: " + ", ".join(outside)
            )
        changeset = operations.get_changeset(cursor["changeset_id"])
        require_approved_description_reviews(
            repository, operations, task, changeset
        )
        record_id = uuid7()
        relative = Path(config.reconciliations) / f"{record_id}.yaml"
        payload = {
            "id": record_id,
            "direction": "spec-led",
            "rationale": args.rationale.strip(),
            "decisions": [
                {
                    "node": node_id,
                    "outcome": (
                        "revised"
                        if nodes[node_id]["outcome"] == "revised"
                        else "confirmed_unchanged"
                    ),
                    "abstraction_changed": False,
                    "rationale": nodes[node_id]["rationale"],
                }
                for node_id in sorted(nodes)
            ],
        }
        mutation = {
            "operation": "reconciliation.record",
            "record": {
                "record_id": record_id,
                "direction": "spec-led",
                "rationale": args.rationale.strip(),
                "source_path": relative.as_posix(),
                "payload": payload,
            },
        }
        current, _ = worktree_snapshot(repository, template=base)
        try:
            apply_graph_mutations(
                base,
                [*changeset["graph_mutations"], mutation],
                code=current.code,
            )
        except CandidateGraphError as error:
            raise OperationError(
                f"native candidate graph is invalid: {error}"
            ) from error
        changeset = operations.append_graph_mutation(
            cursor["changeset_id"],
            expected_revision=changeset["revision"],
            expected_base_version=str(task["base_version"]),
            actor=args.actor,
            rationale=args.rationale.strip(),
            mutation=mutation,
        )
        materialization = materialize_changeset(
            operations,
            repository,
            cursor["changeset_id"],
            expected_revision=changeset["revision"],
            expected_base_version=str(task["base_version"]),
        )
        impact = check_impact(repository, base=str(task["base_version"]))
        if not impact.valid:
            raise OperationError(
                "impact validation failed: "
                + "; ".join(issue.render() for issue in impact.issues)
            )
        cursor["phase"] = "changeset"
        cursor["record"] = relative.as_posix()
        frontier = [
            {
                **item,
                "status": "resolved" if item["status"] == "decided" else item["status"],
            }
            for item in task_frontier(cursor)
        ]
        task = operations.update_task(
            task["task_id"],
            expected_revision=task["revision"],
            actor=args.actor,
            cursor=cursor,
            frontier=frontier,
        )
    return {
        "changeset_revision": changeset["revision"],
        "materialization": materialization,
        "record": relative.as_posix(),
        "task": task,
    }


def dispatch(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "prepare": prepare,
        "inspect-node": inspect_node,
        "stage-graph": stage_graph,
        "decide-node": decide_node,
        "inspect-code": inspect_code,
        "stage-code": stage_code,
        "decide-code": decide_code,
        "note-uncertainty": note_uncertainty,
        "record": record,
    }[args.command](args)


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        result = dispatch(args)
    except (
        OSError,
        UnicodeError,
        AuthoritySerializationError,
        CandidateGraphError,
        FrontierError,
        OperationError,
        ProjectionError,
        SerializationError,
    ) as error:
        print(json.dumps({"error": str(error)}, sort_keys=True), file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
