from __future__ import annotations

import argparse
import json
import traceback
from dataclasses import replace
from pathlib import Path
from typing import Any

from ov_wiki_baseline_benchmark.nashsu_llm_wiki.client import LlmWikiBridgeClient
from ov_wiki_baseline_benchmark.nashsu_llm_wiki.config import BenchmarkConfig
from ov_wiki_baseline_benchmark.nashsu_llm_wiki.runner import (
    BenchmarkRunner,
    PreparedExperiment,
    _stage_dict,
    group_prepared_experiments,
)
from ov_wiki_baseline_benchmark.specs import load_specs, repository_root


EXPERIMENT_IDS = [
    "paperscope_summary_93_gap",
    "paperscope_summary_93_results_comparison",
    "paperscope_summary_93_trend",
]


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


class AuditedBridge:
    def __init__(self, delegate: LlmWikiBridgeClient) -> None:
        self.delegate = delegate
        self.successful_ingest_audits: list[dict[str, Any]] = []

    def __getattr__(self, name: str):
        return getattr(self.delegate, name)

    def ingest(self, *args, **kwargs):
        result = self.delegate.ingest(*args, **kwargs)
        self.successful_ingest_audits.append(
            {
                "duration_seconds": result.duration_seconds,
                **result.usage.as_dict(),
                "token_usage_complete": result.payload.get(
                    "tokenUsageComplete", True
                ),
                "llm_token_usage_complete": result.payload.get(
                    "llmTokenUsageComplete", True
                ),
                "embedding_token_usage_complete": result.payload.get(
                    "embeddingTokenUsageComplete", True
                ),
                "embedding_dimensions": result.payload.get("embeddingDimensions"),
            }
        )
        return result


class IngestOnlyRunner(BenchmarkRunner):
    def __init__(self, *args, output_root: Path, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.output_root = output_root
        self.checked_batches = 0

    def _write_group_manifest(self, corpus_id: str, value: dict[str, Any]) -> None:
        records = value.get("ingestion_batches", [])
        while self.checked_batches < len(records):
            batch_index = self.checked_batches
            record = records[batch_index]
            audit = self.bridge.successful_ingest_audits[batch_index]
            record.update(
                {
                    "llm_token_usage_complete": audit[
                        "llm_token_usage_complete"
                    ],
                    "embedding_token_usage_complete": audit[
                        "embedding_token_usage_complete"
                    ],
                    "embedding_dimensions": audit["embedding_dimensions"],
                }
            )
            warnings: list[str] = []
            errors: list[str] = []
            if not isinstance(audit["duration_seconds"], (int, float)) or audit[
                "duration_seconds"
            ] <= 0:
                errors.append("duration_seconds is not positive")
            components = (
                audit["input_tokens"],
                audit["output_tokens"],
                audit["embedding_tokens"],
            )
            if any(
                isinstance(component, bool)
                or not isinstance(component, int)
                or component < 0
                for component in components
            ):
                errors.append("one or more token components are invalid")
            elif sum(components) != audit["total_tokens"]:
                errors.append("token total does not equal component sum")
            if any(component == 0 for component in components):
                warnings.append("one or more known token components are zero")
            for field in (
                "token_usage_complete",
                "llm_token_usage_complete",
                "embedding_token_usage_complete",
            ):
                if audit[field] is not True:
                    warnings.append(f"{field}=false")
            if audit["embedding_dimensions"] != 1024:
                errors.append(
                    "embedding_dimensions="
                    f"{audit['embedding_dimensions']!r}, expected 1024"
                )
            super()._write_group_manifest(corpus_id, value)
            event = {
                "event": "batch_metrics",
                "batch_index": batch_index,
                "document_offset": record.get("document_offset"),
                "document_count": record.get("document_count"),
                "audit": audit,
                "warnings": warnings,
                "errors": errors,
                "continued_despite_incomplete_telemetry": bool(warnings)
                and not errors,
            }
            print(json.dumps(event, ensure_ascii=False), flush=True)
            self.checked_batches += 1
            if errors:
                write_json(
                    self.output_root / "METRIC_ISSUE.json",
                    {
                        "status": "metric_issue",
                        "corpus_id": corpus_id,
                        "batch_index": batch_index,
                        "batch_record": record,
                        "audit": audit,
                        "warnings": warnings,
                        "errors": errors,
                    },
                )
                raise RuntimeError(
                    f"Batch {batch_index + 1} metric audit failed: "
                    + "; ".join(errors)
                )
        super()._write_group_manifest(corpus_id, value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PaperScope ingestion only")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/noraiddata/ZhangYunhao/ov-wiki-benchmark-data"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_root = args.output_root.expanduser().resolve()
    root = repository_root()
    config = replace(
        BenchmarkConfig.from_yaml(
            root / "baseline_configs" / "nashsu_llm_wiki.yaml"
        ),
        output_dir=output_root,
    )
    specs = load_specs()
    prepared = [
        PreparedExperiment.load(specs[experiment_id], args.data_root)
        for experiment_id in EXPERIMENT_IDS
    ]
    groups = group_prepared_experiments(prepared)
    if len(groups) != 1:
        raise RuntimeError(f"Expected one shared-corpus group, got {len(groups)}")
    experiments = groups[0]
    canonical = experiments[0]
    if len(canonical.documents) != 93:
        raise RuntimeError(
            f"Expected 93 PaperScope documents, got {len(canonical.documents)}"
        )
    corpus_id = f"{canonical.spec.dataset}-{canonical.corpus_fingerprint[:16]}"
    bridge = AuditedBridge(LlmWikiBridgeClient(config))
    runner = IngestOnlyRunner(
        config,
        answer_prompt_path=root / "prompts" / "ov_wiki_bot_answer.txt",
        judge_prompt_path=root / "prompts" / "generic_llm_judge_user.txt",
        bridge=bridge,
        output_root=output_root,
    )
    stale_cleanup = runner.snapshots.cleanup_all().duration_seconds
    bridge.wait_until_ready(config.project_path)
    run = bridge.create_run(corpus_id=corpus_id, project_path=config.project_path)
    manifest: dict[str, Any] = {
        "schema_version": "1.0",
        "run_id": run.run_id,
        "corpus_id": corpus_id,
        "corpus_fingerprint": canonical.corpus_fingerprint,
        "experiments": EXPERIMENT_IDS,
        "config": config.public_manifest(),
        "project_scaffold": dict(run.project_scaffold),
        "snapshot_audit": {
            "stale_cleanup_seconds": stale_cleanup,
            "excluded_from_primary_metrics": True,
            "excluded_from_deletion_metrics": True,
        },
        "status": "created",
        "ingest_only": True,
        "document_count": len(canonical.documents),
        "output_root": str(output_root),
    }
    runner._write_group_manifest(corpus_id, manifest)
    try:
        ingestion, run_id, run_ids = runner._run_ingest_batches(
            canonical,
            corpus_id=corpus_id,
            initial_run=run,
            group_manifest=manifest,
            resume_ingest=False,
        )
        manifest["status"] = "ingested"
        manifest["run_id"] = run_id
        manifest["run_ids"] = run_ids
        manifest["ingestion"] = _stage_dict(ingestion)
        runner._write_group_manifest(corpus_id, manifest)
        write_json(
            output_root / "INGESTION_COMPLETE.json",
            {
                "status": "ingested",
                "corpus_id": corpus_id,
                "run_id": run_id,
                "run_ids": run_ids,
                "ingestion": _stage_dict(ingestion),
            },
        )
        print(
            json.dumps({"event": "ingestion_complete", "run_id": run_id}),
            flush=True,
        )
        return 0
    except BaseException as exc:
        if not (output_root / "METRIC_ISSUE.json").exists():
            write_json(
                output_root / "INGESTION_FAILED.json",
                {
                    "status": "failed",
                    "corpus_id": corpus_id,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                },
            )
        try:
            bridge.stop_service()
        finally:
            runner.snapshots.cleanup_all()
        raise


if __name__ == "__main__":
    raise SystemExit(main())
