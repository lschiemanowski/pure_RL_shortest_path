"""Small operational CLI for TOML-declared experiments."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from .run import (
    execute_standalone_evaluation,
    resume_training_run,
    start_training_run,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pure_rl_shortest_path",
        description="Train and evaluate the pure-RL shortest-path policy.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    train = commands.add_parser("train", help="start a fresh TOML-declared run")
    train.add_argument("configuration", type=Path)

    resume = commands.add_parser("resume", help="continue from a checkpoint")
    resume.add_argument("checkpoint", type=Path)
    resume.add_argument(
        "--config",
        type=Path,
        help="create a derived run with compatible changed settings",
    )
    resume.add_argument(
        "--max-steps",
        type=int,
        help="operational stopping-step override",
    )

    evaluate = commands.add_parser(
        "evaluate", help="evaluate one checkpoint without training"
    )
    evaluate.add_argument("checkpoint", type=Path)
    evaluate.add_argument("configuration", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    command = tuple(sys.argv if argv is None else ("pure_rl_shortest_path", *argv))
    repository = Path.cwd()
    if arguments.command == "train":
        result = start_training_run(
            arguments.configuration,
            source_repository=repository,
            command=command,
        )
        print(result.final_checkpoint)
    elif arguments.command == "resume":
        result = resume_training_run(
            arguments.checkpoint,
            source_repository=repository,
            configuration_path=arguments.config,
            max_steps=arguments.max_steps,
            command=command,
        )
        print(result.final_checkpoint)
    else:
        result = execute_standalone_evaluation(
            arguments.checkpoint, arguments.configuration
        )
        print(result.directory)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
