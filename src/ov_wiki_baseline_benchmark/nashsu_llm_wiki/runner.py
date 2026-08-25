"""Three-stage benchmark orchestration for Nashsu LLM Wiki."""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from ..io import load_jsonl
from ..schema import validate_documents, validate_qas
from ..specs import ExperimentSpec
from .client import LlmWikiBridgeClient
from .config import BenchmarkConfig
from .judge import ArkJudge, JudgeResult
from .metrics import is_refusal, max_token_f1
from .models import QaResult, RunInfo, StageResult, TokenUsage


class Bridge(Protocol):
    def wait_until_ready(self, project_path: Path) -> None: ...
    def create_run(self, *, corpus_id: str, project_path: Path) -> RunInfo: ...
    def ingest(self, run_id: str, documents: list[Path]) -> StageResult: ...
    def answer(self, run_id: str, *, prompt: str, session_id: str) -> QaResult: ...
    def delete(self, run_id: str) -> StageResult: ...


class Judge(Protocol):
    def grade(self, question: str, gold_answers: list[str], answer: str) -> JudgeResult: ...


@dataclass(frozen=True)
class PreparedExperiment:
    spec: ExperimentSpec
    root: Path
    documents: list[dict[str, Any]]
    qas: list[dict[str, Any]]
    corpus_fingerprint: str

    @classmethod
    def load(cls, spec: ExperimentSpec, data_dir: Path) -> "PreparedExperiment":
        root = data_dir / "prepared" / spec.id
        documents = load_jsonl(root / "documents.jsonl")
        qas = load_jsonl(root / "qa.jsonl")
        validate_documents(root, documents, expected_count=spec.expected_documents)
        validate_qas(qas, documents, expected_count=spec.expected_qas)
        digest = hashlib.sha256()
        for document in sorted(documents, key=lambda item: str(item["sha256"])):
            digest.update(str(document["sha256"]).encode("ascii"))
            digest.update(b"\0")
        return cls(spec, root, documents, qas, digest.hexdigest())

    @property
    def document_paths(self) -> list[Path]:
        return [self.root / str(document["path"]) for document in self.documents]


class BenchmarkRunner:
    def __init__(
        self,
        config: BenchmarkConfig,
        *,
        answer_prompt_path: Path,
        judge_prompt_path: Path,
        bridge: Bridge | None = None,
        judge: Judge | None = None,
    ) -> None:
        self.config = config
        self.answer_prompt = answer_prompt_path.read_text(encoding="utf-8")
        _validate_answer_prompt(self.answer_prompt)
        self.bridge = bridge or LlmWikiBridgeClient(config)
        self.judge = judge or ArkJudge(
            api_key=config.ark_api_key(),
            base_url=config.model_base_url,
            model=config.model,
            prompt_path=judge_prompt_path,
            timeout_seconds=config.request_timeout_seconds,
        )

    def run_group(self, experiments: list[PreparedExperiment]) -> dict[str, Any]:
        if not experiments:
            raise ValueError("At least one prepared experiment is required")
        fingerprints = {experiment.corpus_fingerprint for experiment in experiments}
        if len(fingerprints) != 1:
            raise ValueError("run_group requires experiments with an identical corpus")
        canonical = experiments[0]
        for experiment in experiments[1:]:
            if len(experiment.documents) != len(canonical.documents):
                raise ValueError("Shared-corpus group has inconsistent document counts")

        corpus_id = f"{canonical.spec.dataset}-{canonical.corpus_fingerprint[:16]}"
        # Reuse one already-open dedicated project. Each corpus group is fully
        # deleted before the next group, so no project switching is needed.
        project_path = self.config.project_path
        self.bridge.wait_until_ready(project_path)
        run = self.bridge.create_run(corpus_id=corpus_id, project_path=project_path)
        run_id = run.run_id
        group_manifest: dict[str, Any] = {
            "schema_version": "1.0",
            "run_id": run_id,
            "corpus_id": corpus_id,
            "corpus_fingerprint": canonical.corpus_fingerprint,
            "experiments": [experiment.spec.id for experiment in experiments],
            "config": self.config.public_manifest(),
            "project_scaffold": dict(run.project_scaffold),
            "status": "created",
        }
        self._write_group_manifest(corpus_id, group_manifest)

        ingestion = self.bridge.ingest(run_id, canonical.document_paths)
        group_manifest["status"] = "ingested"
        group_manifest["ingestion"] = _stage_dict(ingestion)
        self._write_group_manifest(corpus_id, group_manifest)

        completed: list[tuple[PreparedExperiment, list[dict[str, Any]]]] = []
        for experiment in experiments:
            records = self._run_qa(experiment, run_id, ingestion)
            self._run_judge(experiment, records, ingestion)
            completed.append((experiment, records))

        deletion = self.bridge.delete(run_id)
        for experiment, records in completed:
            self._write_final_report(experiment, records, ingestion, deletion)
        group_manifest["status"] = "completed"
        group_manifest["deletion"] = _stage_dict(deletion)
        self._write_group_manifest(corpus_id, group_manifest)
        return group_manifest

    def _run_qa(
        self,
        experiment: PreparedExperiment,
        run_id: str,
        ingestion: StageResult,
    ) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for index, qa in enumerate(experiment.qas):
            session_id = f"benchmark-{experiment.spec.id}-{index}-{uuid.uuid4().hex}"
            prompt = self.answer_prompt.format(question=qa["question"])
            result = self.bridge.answer(run_id, prompt=prompt, session_id=session_id)
            records.append(
                {
                    "_global_index": index,
                    "sample_id": qa["id"],
                    "question": qa["question"],
                    "gold_answers": qa["gold_answers"],
                    "category": str(qa.get("category", "")),
                    "evidence": qa.get("evidence", []),
                    "document_ids": qa.get("document_ids", []),
                    "retrieval": {
                        "latency_sec": result.duration_seconds,
                        "uris": list(result.payload.get("references", [])),
                    },
                    "llm": {"final_answer": result.answer},
                    "llm_wiki": {
                        "session_id": result.session_id,
                        "mode": "standard",
                        "retrieval_mode": "standard",
                        "resolved_top_k": 5,
                        "trace": result.payload.get("trace", []),
                    },
                    "metrics": {"Recall": 0.0},
                    "token_usage": {
                        "total_input_tokens": result.usage.input_tokens,
                        "llm_output_tokens": result.usage.output_tokens,
                        "retrieval_embedding_tokens": result.usage.embedding_tokens,
                        "agent_prompt_tokens": result.usage.agent_input_tokens,
                        "agent_completion_tokens": result.usage.agent_output_tokens,
                        "search_llm_input_tokens": result.usage.search_input_tokens,
                        "search_llm_output_tokens": result.usage.search_output_tokens,
                        "retrieval_total_tokens": result.usage.total,
                        "total_tokens": result.usage.total,
                    },
                }
            )
        output = self._experiment_output(experiment.spec.id)
        _write_json(
            output / "generated_answers.json",
            {
                "summary": {
                    "dataset": experiment.spec.id,
                    "total_queries": len(records),
                },
                "results": records,
            },
        )
        self._write_report(experiment, records, ingestion, deletion=None)
        return records

    def _run_judge(
        self,
        experiment: PreparedExperiment,
        records: list[dict[str, Any]],
        ingestion: StageResult,
    ) -> None:
        judge_usage = TokenUsage()
        judge_usage_complete = True
        for record in records:
            answer = record["llm"]["final_answer"]
            gold_answers = record["gold_answers"]
            f1 = max_token_f1(answer, gold_answers)
            result = self.judge.grade(record["question"], gold_answers, answer)
            score: float = float(result.score)
            reasoning = result.reasoning
            prompt_type = result.prompt_type
            if is_refusal(answer) and any(is_refusal(gold) for gold in gold_answers):
                f1 = 1.0
                score = 4.0
                reasoning = "System successfully identified Unanswerable/Refusal condition."
                prompt_type = "Heuristic_Refusal_Check"
            record["metrics"].update({"F1": f1, "Accuracy": score})
            record["llm_evaluation"] = {
                "prompt_used": prompt_type,
                "reasoning": reasoning,
                "normalized_score": score,
            }
            if result.usage is None:
                judge_usage_complete = False
            else:
                judge_usage = judge_usage + result.usage

        output = self._experiment_output(experiment.spec.id)
        _write_json(output / "qa_eval_detailed_results.json", {"results": records})
        _write_json(
            output / "judge_telemetry.json",
            {
                "usage_complete": judge_usage_complete,
                **judge_usage.as_dict(),
            },
        )
        self._write_report(experiment, records, ingestion, deletion=None)

    def _write_final_report(
        self,
        experiment: PreparedExperiment,
        records: list[dict[str, Any]],
        ingestion: StageResult,
        deletion: StageResult,
    ) -> None:
        self._write_report(experiment, records, ingestion, deletion=deletion)

    def _write_report(
        self,
        experiment: PreparedExperiment,
        records: list[dict[str, Any]],
        ingestion: StageResult,
        *,
        deletion: StageResult | None,
    ) -> None:
        count = len(records)
        qa_time = sum(float(record["retrieval"]["latency_sec"]) for record in records)
        qa_input = sum(int(record["token_usage"]["total_input_tokens"]) for record in records)
        qa_output = sum(int(record["token_usage"]["llm_output_tokens"]) for record in records)
        qa_embedding = sum(
            int(record["token_usage"]["retrieval_embedding_tokens"]) for record in records
        )
        agent_input = sum(
            int(record["token_usage"]["agent_prompt_tokens"]) for record in records
        )
        agent_output = sum(
            int(record["token_usage"]["agent_completion_tokens"]) for record in records
        )
        search_input = sum(
            int(record["token_usage"]["search_llm_input_tokens"]) for record in records
        )
        search_output = sum(
            int(record["token_usage"]["search_llm_output_tokens"]) for record in records
        )
        report: dict[str, Any] = {
            "Dataset": experiment.spec.id,
            "Total Queries Evaluated": (
                count if records and "Accuracy" in records[0]["metrics"] else 0
            ),
            "Benchmark Contract": self.config.public_manifest(),
            "Insertion Efficiency (Total Dataset)": {
                "Total Insertion Time (s)": ingestion.duration_seconds,
                "Total Source Documents": len(experiment.documents),
                "Total Input Tokens": ingestion.usage.input_tokens,
                "Total Output Tokens": ingestion.usage.output_tokens,
                "Total Embedding Tokens": ingestion.usage.embedding_tokens,
                "Total Insertion Token Cost": ingestion.usage.total,
                "Includes Review Sweep": True,
            },
            "Query Efficiency (Average Per Query)": {
                "Average Retrieval Time (s)": qa_time / count if count else 0.0,
                "Average Retrieval Token Cost": (
                    (qa_input + qa_output + qa_embedding) / count if count else 0.0
                ),
            },
            "Query Efficiency (Total Dataset)": {
                "Total Retrieval Time (s)": qa_time,
                "Total Agent LLM Input Tokens": agent_input,
                "Total Agent LLM Output Tokens": agent_output,
                "Total Search LLM Input Tokens": search_input,
                "Total Search LLM Output Tokens": search_output,
                "Total Retrieval Embedding Tokens": qa_embedding,
                "Total Retrieval Token Cost": qa_input + qa_output + qa_embedding,
            },
        }
        if records and "Accuracy" in records[0]["metrics"]:
            raw_accuracy = (
                sum(float(record["metrics"]["Accuracy"]) for record in records) / count
            )
            report["Performance Metrics"] = {
                "Average F1 Score": sum(
                    float(record["metrics"]["F1"]) for record in records
                )
                / count,
                "Average Recall": sum(
                    float(record["metrics"].get("Recall", 0.0)) for record in records
                )
                / count,
                "Average Accuracy (Hit 0-4)": raw_accuracy,
                "Average Accuracy (normalization)": raw_accuracy / 4,
                "Normalized Accuracy (0-1)": raw_accuracy / 4,
            }
        if deletion is not None:
            report["Deletion Efficiency (Total Dataset)"] = {
                "Total Deletion Time (s)": deletion.duration_seconds,
                "Total Input Tokens": deletion.usage.input_tokens,
                "Total Output Tokens": deletion.usage.output_tokens,
                "Total Embedding Tokens": deletion.usage.embedding_tokens,
                "Total Deletion Token Cost": deletion.usage.total,
            }
        _write_json(
            self._experiment_output(experiment.spec.id) / "benchmark_metrics_report.json",
            report,
        )

    def _experiment_output(self, experiment_id: str) -> Path:
        path = self.config.output_dir / experiment_id / "wiki"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _write_group_manifest(self, corpus_id: str, value: dict[str, Any]) -> None:
        _write_json(self.config.output_dir / "groups" / corpus_id / "run.json", value)


def group_prepared_experiments(
    experiments: list[PreparedExperiment],
) -> list[list[PreparedExperiment]]:
    groups: dict[str, list[PreparedExperiment]] = {}
    for experiment in experiments:
        groups.setdefault(experiment.corpus_fingerprint, []).append(experiment)
    return list(groups.values())


def _validate_answer_prompt(prompt: str) -> None:
    expected = (
        "Answer this question as briefly as possible. Use only the information "
        "available in the database. Do not use any external source.\n\n"
        "Question: {question}\n"
    )
    if prompt != expected:
        raise ValueError("Answer prompt does not exactly match the approved benchmark prompt")


def _stage_dict(stage: StageResult) -> dict[str, Any]:
    return {"duration_seconds": stage.duration_seconds, **stage.usage.as_dict()}


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)
