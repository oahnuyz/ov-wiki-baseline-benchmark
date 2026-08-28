"""CLI for the Nashsu LLM Wiki baseline benchmark."""

from __future__ import annotations

import argparse
import signal
import sys
from pathlib import Path

from ..specs import load_specs, repository_root
from .config import BenchmarkConfig
from .runner import BenchmarkRunner, PreparedExperiment, group_prepared_experiments


def build_parser() -> argparse.ArgumentParser:
    root = repository_root()
    parser = argparse.ArgumentParser(prog="ov-wiki-nashsu")
    parser.add_argument("experiments", nargs="+")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument(
        "--config", default=str(root / "baseline_configs" / "nashsu_llm_wiki.yaml")
    )
    parser.add_argument(
        "--resume",
        "--resume-ingest",
        dest="resume",
        action="store_true",
        help=(
            "Resume an interrupted group from its manifest and saved QA outputs. An "
            "immediately preceding ingestion batch that completed with incomplete "
            "provider usage telemetry is accepted as ingested."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    previous_sigterm = signal.getsignal(signal.SIGTERM)

    def interrupt_on_sigterm(signum: int, frame: object) -> None:
        raise KeyboardInterrupt("Benchmark interrupted by SIGTERM")

    signal.signal(signal.SIGTERM, interrupt_on_sigterm)
    args = build_parser().parse_args(argv)
    try:
        specs = load_specs()
        unknown = sorted(set(args.experiments) - set(specs))
        if unknown:
            raise ValueError(f"Unknown experiments: {unknown}")
        data_dir = Path(args.data_dir).expanduser().resolve()
        prepared = [
            PreparedExperiment.load(specs[experiment_id], data_dir)
            for experiment_id in args.experiments
        ]
        config = BenchmarkConfig.from_yaml(Path(args.config).expanduser().resolve())
        root = repository_root()
        runner = BenchmarkRunner(
            config,
            answer_prompt_path=root / "prompts" / "ov_wiki_bot_answer.txt",
            judge_prompt_path=root / "prompts" / "generic_llm_judge_user.txt",
        )
        for group in group_prepared_experiments(prepared):
            runner.run_group(group, resume_ingest=args.resume)
        return 0
    except KeyboardInterrupt as exc:
        print(f"Interrupted: {exc}", file=sys.stderr)
        return 130
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)


if __name__ == "__main__":
    raise SystemExit(main())
