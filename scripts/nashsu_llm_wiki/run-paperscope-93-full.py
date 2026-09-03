from __future__ import annotations

import json
import traceback
from dataclasses import replace
from pathlib import Path
from typing import Any

from ov_wiki_baseline_benchmark.nashsu_llm_wiki.config import BenchmarkConfig
from ov_wiki_baseline_benchmark.nashsu_llm_wiki.runner import (
    BenchmarkRunner,
    PreparedExperiment,
    group_prepared_experiments,
)
from ov_wiki_baseline_benchmark.specs import load_specs, repository_root


EXPERIMENT_IDS = [
    "paperscope_summary_93_gap",
    "paperscope_summary_93_results_comparison",
    "paperscope_summary_93_trend",
]
DATA_ROOT = Path("/noraiddata/ZhangYunhao/ov-wiki-benchmark-data")
OUTPUT_ROOT = Path.home() / (
    "nashsu-llm-wiki-baseline/results/"
    "paperscope_93_rerun_20260901_v5_15_retrievals_no_delete"
)


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> int:
    root = repository_root()
    config = replace(
        BenchmarkConfig.from_yaml(
            root / "baseline_configs" / "nashsu_llm_wiki.yaml"
        ),
        output_dir=OUTPUT_ROOT,
    )
    specs = load_specs()
    prepared = [
        PreparedExperiment.load(specs[experiment_id], DATA_ROOT)
        for experiment_id in EXPERIMENT_IDS
    ]
    groups = group_prepared_experiments(prepared)
    if len(groups) != 1:
        raise RuntimeError(f"Expected one shared-corpus group, got {len(groups)}")
    total_questions = sum(len(experiment.qas) for experiment in groups[0])
    if total_questions != 352:
        raise RuntimeError(f"Expected 352 questions, got {total_questions}")

    write_json(
        OUTPUT_ROOT / "PIPELINE_STARTED.json",
        {
            "status": "started",
            "experiments": EXPERIMENT_IDS,
            "document_count": len(groups[0][0].documents),
            "question_count": total_questions,
            "config": config.public_manifest(),
        },
    )
    runner = BenchmarkRunner(
        config,
        answer_prompt_path=root / "prompts" / "ov_wiki_bot_answer.txt",
        judge_prompt_path=root / "prompts" / "generic_llm_judge_user.txt",
    )
    try:
        result = runner.run_group(
            groups[0],
            resume_ingest=False,
            skip_deletion=True,
        )
        write_json(
            OUTPUT_ROOT / "PIPELINE_COMPLETE.json",
            {
                "status": "completed",
                "experiments": EXPERIMENT_IDS,
                "document_count": len(groups[0][0].documents),
                "question_count": total_questions,
                "final_manifest_status": result.get("status"),
                "ingestion": result.get("ingestion"),
                "deletion_skipped": result.get("deletion_skipped"),
                "preserved_project_path": result.get("preserved_project_path"),
            },
        )
        return 0
    except BaseException as exc:
        write_json(
            OUTPUT_ROOT / "PIPELINE_FAILED.json",
            {
                "status": "failed",
                "error": str(exc),
                "traceback": traceback.format_exc(),
            },
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
