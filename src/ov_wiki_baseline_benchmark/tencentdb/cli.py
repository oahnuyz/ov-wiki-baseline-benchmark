"""Command line entry point for the TencentDB baseline."""

from __future__ import annotations

import argparse
from pathlib import Path

from ..specs import load_specs, repository_root
from .config import TencentDBConfig
from .runner import PreparedTencentExperiment, TencentDBRunner


def main(argv: list[str] | None = None) -> int:
    root = repository_root()
    parser = argparse.ArgumentParser(prog="ov-wiki-tencentdb")
    parser.add_argument("experiments", nargs="+")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--config", default=str(root / "baseline_configs" / "tencentdb_agent_memory.yaml"))
    parser.add_argument("--round", choices=["all", "ingest", "qa", "judge", "delete"], default="all")
    parser.add_argument("--retry-failed-pdfs", action="store_true")
    args = parser.parse_args(argv)
    specs = load_specs()
    unknown = sorted(set(args.experiments) - set(specs))
    if unknown:
        parser.error(f"unknown experiments: {unknown}")
    data_dir = Path(args.data_dir).expanduser().resolve()
    config = TencentDBConfig.from_yaml(Path(args.config).expanduser().resolve())
    prepared = [PreparedTencentExperiment.load(specs[e], data_dir) for e in args.experiments]
    runner = TencentDBRunner(config, answer_prompt_path=root / "prompts" / "ov_wiki_bot_answer.txt", judge_prompt_path=root / "prompts" / "generic_llm_judge_user.txt")
    # Multiple variants sharing a corpus are ingested once and QA'd independently.
    groups: dict[str, list[PreparedTencentExperiment]] = {}
    for experiment in prepared:
        groups.setdefault(experiment.corpus_fingerprint, []).append(experiment)
    for group in groups.values():
        runner.run_group(group, round_name=args.round, retry_failed_pdfs=args.retry_failed_pdfs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
