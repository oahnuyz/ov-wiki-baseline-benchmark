"""Three-stage benchmark orchestration for Nashsu LLM Wiki."""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from ..io import load_jsonl
from ..schema import validate_documents, validate_qas
from ..specs import ExperimentSpec
from .client import BridgeRequestError, LlmWikiBridgeClient
from .config import BenchmarkConfig
from .judge import ArkJudge, JudgeResult
from .metrics import is_refusal, max_token_f1
from .models import QaResult, RunInfo, StageResult, TokenUsage
from .snapshots import ProjectSnapshotManager


class Bridge(Protocol):
    def wait_until_ready(self, project_path: Path) -> None: ...
    def create_run(
        self,
        *,
        corpus_id: str,
        project_path: Path,
        continuation: bool = False,
    ) -> RunInfo: ...
    def ingest(
        self,
        run_id: str,
        documents: list[Path],
        *,
        document_offset: int = 0,
        final_batch: bool = True,
    ) -> StageResult: ...
    def restart_service(self, project_path: Path) -> None: ...
    def stop_service(self) -> None: ...
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
        self.snapshots = ProjectSnapshotManager(
            project_path=config.project_path,
            snapshot_root=config.snapshot_root,
        )
        self.judge = judge or ArkJudge(
            api_key=config.ark_api_key(),
            base_url=config.model_base_url,
            model=config.model,
            prompt_path=judge_prompt_path,
            timeout_seconds=config.request_timeout_seconds,
        )

    def run_group(
        self,
        experiments: list[PreparedExperiment],
        *,
        resume_ingest: bool = False,
        skip_deletion: bool = False,
    ) -> dict[str, Any]:
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
        project_path = self.config.project_path
        stale_cleanup = self.snapshots.cleanup_all().duration_seconds
        self.bridge.wait_until_ready(project_path)
        if resume_ingest:
            group_manifest = self._read_group_manifest(corpus_id)
            self._validate_resume_manifest(
                group_manifest,
                corpus_id=corpus_id,
                experiments=experiments,
            )
            resume_from_status = str(
                group_manifest.get("resume_from_status", group_manifest["status"])
            )
            run = self.bridge.create_run(
                corpus_id=corpus_id,
                project_path=project_path,
                continuation=True,
            )
            run_id = run.run_id
            group_manifest["run_id"] = run_id
            group_manifest["resume_from_status"] = resume_from_status
            group_manifest["status"] = "resuming_ingest"
            group_manifest.setdefault("snapshot_audit", {})[
                "resume_stale_cleanup_seconds"
            ] = stale_cleanup
        else:
            # Reuse one already-open dedicated project. Each corpus group is fully
            # deleted before the next group, so no project switching is needed.
            run = self.bridge.create_run(corpus_id=corpus_id, project_path=project_path)
            run_id = run.run_id
            group_manifest = {
                "schema_version": "1.0",
                "run_id": run_id,
                "corpus_id": corpus_id,
                "corpus_fingerprint": canonical.corpus_fingerprint,
                "experiments": [experiment.spec.id for experiment in experiments],
                "config": self.config.public_manifest(),
                "project_scaffold": dict(run.project_scaffold),
                "snapshot_audit": {
                    "stale_cleanup_seconds": stale_cleanup,
                    "excluded_from_primary_metrics": True,
                    "excluded_from_deletion_metrics": True,
                },
                "status": "created",
            }
        self._write_group_manifest(corpus_id, group_manifest)
        try:
            if resume_ingest and group_manifest.get("resume_from_status") in {
                "ingested",
                "answering",
                "judging",
            }:
                ingestion = _stage_from_manifest_dict(group_manifest.get("ingestion"))
                run_ids = list(group_manifest.get("run_ids", []))
                if run_id not in run_ids:
                    run_ids.append(run_id)
            else:
                ingestion, run_id, run_ids = self._run_ingest_batches(
                    canonical,
                    corpus_id=corpus_id,
                    initial_run=run,
                    group_manifest=group_manifest,
                    resume_ingest=resume_ingest,
                )
                group_manifest["status"] = "ingested"
                group_manifest["run_id"] = run_id
                group_manifest["run_ids"] = run_ids
                group_manifest["ingestion"] = _stage_dict(ingestion)
                self._write_group_manifest(corpus_id, group_manifest)

            completed: list[tuple[PreparedExperiment, list[dict[str, Any]]]] = []
            for experiment in experiments:
                existing_records = (
                    self._load_existing_records(experiment) if resume_ingest else []
                )
                group_manifest["status"] = "answering"
                group_manifest["active_experiment"] = experiment.spec.id
                group_manifest["completed_answers"] = len(existing_records)
                self._write_group_manifest(corpus_id, group_manifest)
                records, run_id = self._run_qa(
                    experiment,
                    run_id,
                    ingestion,
                    corpus_id=corpus_id,
                    group_manifest=group_manifest,
                    existing_records=existing_records,
                )
                group_manifest["run_id"] = run_id
                group_manifest["status"] = "judging"
                group_manifest["completed_answers"] = len(records)
                group_manifest["completed_judgements"] = sum(
                    "llm_evaluation" in record for record in records
                )
                self._write_group_manifest(corpus_id, group_manifest)
                self._run_judge(experiment, records, ingestion)
                completed.append((experiment, records))

            if skip_deletion:
                group_manifest["status"] = "completed_without_deletion"
                group_manifest.pop("resume_from_status", None)
                group_manifest.pop("active_experiment", None)
                group_manifest["deletion_skipped"] = True
                group_manifest["preserved_project_path"] = str(project_path)
                self._write_group_manifest(corpus_id, group_manifest)
                return group_manifest

            deletion = self.bridge.delete(run_id)
            for experiment, records in completed:
                self._write_final_report(experiment, records, ingestion, deletion)
            group_manifest["status"] = "completed"
            group_manifest.pop("resume_from_status", None)
            group_manifest.pop("active_experiment", None)
            group_manifest["deletion"] = _stage_dict(deletion)
            self._write_group_manifest(corpus_id, group_manifest)
            return group_manifest
        except BaseException:
            try:
                self.bridge.stop_service()
            finally:
                self.snapshots.cleanup_all()
            raise

    def _run_ingest_batches(
        self,
        experiment: PreparedExperiment,
        *,
        corpus_id: str,
        initial_run: RunInfo,
        group_manifest: dict[str, Any],
        resume_ingest: bool = False,
    ) -> tuple[StageResult, str, list[str]]:
        documents = experiment.document_paths
        batch_size = self.config.ingest_batch_size
        batches = [
            documents[index : index + batch_size]
            for index in range(0, len(documents), batch_size)
        ]
        if not batches:
            raise ValueError("Prepared experiment contains no source documents")

        batch_records: list[dict[str, Any]] = (
            list(group_manifest.get("ingestion_batches", [])) if resume_ingest else []
        )
        self._validate_completed_batches(batch_records, batches)
        active_seconds = sum(
            float(record["duration_seconds"]) for record in batch_records
        )
        usage = TokenUsage()
        usage_complete = True
        for record in batch_records:
            usage_complete = usage_complete and record.get(
                "token_usage_complete", True
            )
            usage = usage + _known_usage_from_stage_dict(record)
        run = initial_run
        run_ids = list(
            dict.fromkeys(
                [
                    str(record["run_id"])
                    for record in batch_records
                    if record.get("run_id")
                ]
                + [run.run_id]
            )
        )
        previous_progress = group_manifest.get("ingestion_progress", {})
        previous_restart_count = int(previous_progress.get("restart_count", 0))
        resume_restart_count = 1 if resume_ingest and batch_records else 0
        restart_count = previous_restart_count + resume_restart_count
        planned_restart_count = previous_restart_count
        retry_restart_count = 0
        batch_retry_count = 0
        discarded_retry_seconds = 0.0
        snapshot_creation_seconds = 0.0
        snapshot_restore_seconds = 0.0
        snapshot_cleanup_seconds = 0.0
        retry_records: list[dict[str, Any]] = (
            list(group_manifest.get("discarded_ingest_attempts", []))
            if resume_ingest
            else []
        )
        accepted_incomplete_records: list[dict[str, Any]] = list(
            group_manifest.get("accepted_incomplete_ingest_telemetry", [])
        )
        if resume_ingest:
            recovered = self._recover_completed_incomplete_telemetry_batch(
                batches=batches,
                batch_records=batch_records,
                retry_records=retry_records,
            )
            if recovered is not None:
                batch_records.append(recovered)
                active_seconds += float(recovered["duration_seconds"])
                usage_complete = False
                accepted_incomplete_records.append(
                    {
                        **recovered,
                        "accepted_during_resume": True,
                    }
                )
                group_manifest["ingestion_batches"] = list(batch_records)
                group_manifest["discarded_ingest_attempts"] = list(retry_records)
                group_manifest["accepted_incomplete_ingest_telemetry"] = list(
                    accepted_incomplete_records
                )
                self._write_group_manifest(corpus_id, group_manifest)
            unresolved_next_batch = [
                record
                for record in retry_records
                if record.get("batch_index") == len(batch_records)
            ]
            if unresolved_next_batch:
                raise ValueError(
                    "Resume cannot accept a partially written batch unless its only "
                    "error is incomplete provider usage telemetry"
                )
        resumed_active_seconds = active_seconds
        operational_started = time.monotonic()

        for batch_index in range(len(batch_records), len(batches)):
            batch = batches[batch_index]
            offset = batch_index * batch_size
            final_batch = batch_index == len(batches) - 1
            group_manifest["status"] = "ingesting"
            group_manifest["ingestion_progress"] = {
                "completed_documents": offset,
                "total_documents": len(documents),
                "current_batch": batch_index + 1,
                "batch_count": len(batches),
                "batch_size": batch_size,
                "restart_count": restart_count,
            }
            self._write_group_manifest(corpus_id, group_manifest)

            snapshot_creation_seconds += self.snapshots.create(
                corpus_id, batch_index
            ).duration_seconds
            attempt = 0
            try:
                while True:
                    attempt_started = time.monotonic()
                    try:
                        result = self.bridge.ingest(
                            run.run_id,
                            batch,
                            document_offset=offset,
                            final_batch=final_batch,
                        )
                        break
                    except BridgeRequestError as exc:
                        failed_seconds = time.monotonic() - attempt_started
                        if _is_incomplete_ingest_telemetry(exc):
                            result = StageResult(
                                duration_seconds=failed_seconds,
                                usage=TokenUsage(),
                                payload={
                                    "status": "completed",
                                    "tokenUsageComplete": False,
                                    "telemetryError": exc.detail,
                                    "reviewSweepCompleted": final_batch,
                                    "embeddingDimensions": 1024,
                                },
                            )
                            accepted_incomplete_records.append(
                                {
                                    "batch_index": batch_index,
                                    "attempt": attempt + 1,
                                    "run_id": run.run_id,
                                    "duration_seconds": failed_seconds,
                                    "token_usage": None,
                                    "token_usage_complete": False,
                                    "error": exc.detail,
                                    "accepted_as_completed_batch": True,
                                    "excluded_from_primary_time_metrics": False,
                                }
                            )
                            group_manifest[
                                "accepted_incomplete_ingest_telemetry"
                            ] = list(accepted_incomplete_records)
                            self._write_group_manifest(corpus_id, group_manifest)
                            break
                        retryable = _is_retryable_ingest_error(exc)
                        retry_record = {
                            "batch_index": batch_index,
                            "attempt": attempt + 1,
                            "run_id": run.run_id,
                            "duration_seconds": failed_seconds,
                            "token_usage": None,
                            "token_usage_complete": False,
                            "retryable": retryable,
                            "error": exc.detail,
                            "excluded_from_primary_metrics": True,
                        }
                        retry_records.append(retry_record)
                        group_manifest["discarded_ingest_attempts"] = list(retry_records)
                        self._write_group_manifest(corpus_id, group_manifest)
                        if not retryable or attempt >= self.config.max_batch_retries:
                            raise
                        discarded_retry_seconds += failed_seconds
                        batch_retry_count += 1
                        attempt += 1
                        self.bridge.stop_service()
                        snapshot_restore_seconds += self.snapshots.restore(
                            corpus_id, batch_index
                        ).duration_seconds
                        self.bridge.restart_service(self.config.project_path)
                        restart_count += 1
                        retry_restart_count += 1
                        run = self.bridge.create_run(
                            corpus_id=corpus_id,
                            project_path=self.config.project_path,
                            continuation=True,
                        )
                        run_ids.append(run.run_id)
            finally:
                snapshot_cleanup_seconds += self.snapshots.delete_batch(
                    corpus_id, batch_index
                ).duration_seconds
            active_seconds += result.duration_seconds
            usage = usage + result.usage
            result_usage_complete = result.payload.get("tokenUsageComplete", True)
            usage_complete = usage_complete and result_usage_complete
            batch_record = {
                "batch_index": batch_index,
                "run_id": run.run_id,
                "document_offset": offset,
                "document_count": len(batch),
                "final_batch": final_batch,
                "attempt_count": attempt + 1,
                "retry_count": attempt,
                **_stage_dict(result),
            }
            batch_records.append(batch_record)
            group_manifest["ingestion_batches"] = list(batch_records)
            group_manifest["ingestion_progress"] = {
                "completed_documents": offset + len(batch),
                "total_documents": len(documents),
                "current_batch": batch_index + 1,
                "batch_count": len(batches),
                "batch_size": batch_size,
                "restart_count": restart_count,
            }
            self._write_group_manifest(corpus_id, group_manifest)

            if not final_batch and self.config.restart_between_ingest_batches:
                self.bridge.restart_service(self.config.project_path)
                restart_count += 1
                planned_restart_count += 1
                run = self.bridge.create_run(
                    corpus_id=corpus_id,
                    project_path=self.config.project_path,
                    continuation=True,
                )
                run_ids.append(run.run_id)

        operational_seconds = (
            resumed_active_seconds + time.monotonic() - operational_started
        )
        payload = {
            "status": "completed",
            "reviewSweepCompleted": True,
            "reviewSweepCount": 1,
            "embeddingDimensions": 1024,
            "batchSize": batch_size,
            "batchCount": len(batches),
            "restartCount": restart_count,
            "plannedRestartCount": planned_restart_count,
            "retryRestartCount": retry_restart_count,
            "batchRetryCount": batch_retry_count,
            "resumeRestartCount": resume_restart_count,
            "tokenUsageComplete": usage_complete,
            "knownTokenUsage": usage.as_dict(),
            "discardedRetryTimeSeconds": discarded_retry_seconds,
            "discardedRetryTokenUsage": None,
            "discardedRetryTokenUsageComplete": False if retry_records else True,
            "snapshotCreationTimeSeconds": snapshot_creation_seconds,
            "snapshotRestoreTimeSeconds": snapshot_restore_seconds,
            "snapshotCleanupTimeSeconds": snapshot_cleanup_seconds,
            "operationalWallClockSeconds": operational_seconds,
            "operationalWallClockComplete": not resume_ingest,
            "batches": batch_records,
            "discardedAttempts": retry_records,
            "acceptedIncompleteTelemetry": accepted_incomplete_records,
        }
        return StageResult(active_seconds, usage, payload), run.run_id, run_ids

    def _read_group_manifest(self, corpus_id: str) -> dict[str, Any]:
        path = self.config.output_dir / "groups" / corpus_id / "run.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError(f"Invalid group manifest: {path}")
        return value

    def _validate_resume_manifest(
        self,
        manifest: dict[str, Any],
        *,
        corpus_id: str,
        experiments: list[PreparedExperiment],
    ) -> None:
        if manifest.get("corpus_id") != corpus_id:
            raise ValueError("Resume manifest corpus does not match the selected corpus")
        if manifest.get("experiments") != [item.spec.id for item in experiments]:
            raise ValueError("Resume manifest experiments do not match the selection")
        if manifest.get("status") not in {
            "ingesting",
            "resuming_ingest",
            "ingested",
            "answering",
            "judging",
        }:
            raise ValueError(
                "Resume requires an interrupted ingestion, QA, or Judge manifest"
            )
        if manifest.get("config") != self.config.public_manifest():
            raise ValueError("Resume manifest benchmark configuration has changed")

    @staticmethod
    def _validate_completed_batches(
        records: list[dict[str, Any]], batches: list[list[Path]]
    ) -> None:
        if len(records) > len(batches):
            raise ValueError("Resume manifest has more completed batches than planned")
        for index, record in enumerate(records):
            expected_count = len(batches[index])
            if record.get("batch_index") != index or record.get(
                "document_count"
            ) != expected_count:
                raise ValueError(f"Resume manifest has an invalid batch {index + 1}")

    @staticmethod
    def _recover_completed_incomplete_telemetry_batch(
        *,
        batches: list[list[Path]],
        batch_records: list[dict[str, Any]],
        retry_records: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        next_index = len(batch_records)
        candidates = [
            record
            for record in retry_records
            if record.get("batch_index") == next_index
            and record.get("error")
            == "Provider usage telemetry was incomplete during ingestion"
        ]
        if not candidates:
            return None
        if next_index >= len(batches) or len(candidates) != 1:
            raise ValueError("Cannot unambiguously recover incomplete ingestion telemetry")
        attempt = candidates[0]
        retry_records.remove(attempt)
        batch_size = len(batches[next_index])
        return {
            "batch_index": next_index,
            "run_id": attempt["run_id"],
            "document_offset": sum(len(batch) for batch in batches[:next_index]),
            "document_count": batch_size,
            "final_batch": next_index == len(batches) - 1,
            "attempt_count": int(attempt.get("attempt", 1)),
            "retry_count": max(0, int(attempt.get("attempt", 1)) - 1),
            "duration_seconds": float(attempt["duration_seconds"]),
            "input_tokens": None,
            "output_tokens": None,
            "embedding_tokens": None,
            "total_tokens": None,
            "known_input_tokens": 0,
            "known_output_tokens": 0,
            "known_embedding_tokens": 0,
            "known_total_tokens": 0,
            "token_usage_complete": False,
            "telemetry_error": attempt["error"],
        }

    def _run_qa(
        self,
        experiment: PreparedExperiment,
        run_id: str,
        ingestion: StageResult,
        *,
        corpus_id: str,
        group_manifest: dict[str, Any],
        existing_records: list[dict[str, Any]] | None = None,
    ) -> tuple[list[dict[str, Any]], str]:
        records = list(existing_records or [])
        retry_audit_path = self._experiment_output(experiment.spec.id) / "qa_retry_audit.json"
        retry_audit: list[dict[str, Any]] = []
        if retry_audit_path.exists():
            saved_audit = json.loads(retry_audit_path.read_text(encoding="utf-8"))
            if not isinstance(saved_audit, list):
                raise ValueError(f"Invalid QA retry audit: {retry_audit_path}")
            retry_audit = saved_audit
        for index, qa in enumerate(experiment.qas[len(records) :], start=len(records)):
            prompt = self.answer_prompt.format(question=qa["question"])
            retry_count = 0
            while True:
                session_id = f"benchmark-{experiment.spec.id}-{index}-{uuid.uuid4().hex}"
                attempt_started = time.monotonic()
                try:
                    result = self.bridge.answer(
                        run_id,
                        prompt=prompt,
                        session_id=session_id,
                    )
                    break
                except BridgeRequestError as error:
                    failed_duration = time.monotonic() - attempt_started
                    if (
                        not _is_retryable_qa_error(error)
                        or retry_count >= self.config.max_qa_retries
                    ):
                        raise
                    retry_count += 1
                    audit_entry: dict[str, Any] = {
                        "sample_id": qa["id"],
                        "question_index": index,
                        "failed_attempt": retry_count,
                        "session_id": session_id,
                        "run_id": run_id,
                        "error": str(error),
                        "failed_attempt_wall_clock_seconds": failed_duration,
                        "failed_attempt_token_usage": None,
                        "failed_attempt_token_usage_complete": False,
                        "excluded_from_primary_qa_metrics": True,
                    }
                    recovery_started = time.monotonic()
                    try:
                        self.bridge.restart_service(self.config.project_path)
                        replacement = self.bridge.create_run(
                            corpus_id=corpus_id,
                            project_path=self.config.project_path,
                            continuation=True,
                        )
                    except BaseException as recovery_error:
                        audit_entry["recovery_error"] = str(recovery_error)
                        audit_entry["recovery_wall_clock_seconds"] = (
                            time.monotonic() - recovery_started
                        )
                        retry_audit.append(audit_entry)
                        _write_json(retry_audit_path, retry_audit)
                        raise
                    run_id = replacement.run_id
                    audit_entry["replacement_run_id"] = run_id
                    audit_entry["recovery_wall_clock_seconds"] = (
                        time.monotonic() - recovery_started
                    )
                    retry_audit.append(audit_entry)
                    _write_json(retry_audit_path, retry_audit)
                    run_ids = group_manifest.setdefault("run_ids", [])
                    if run_id not in run_ids:
                        run_ids.append(run_id)
                    group_manifest["run_id"] = run_id
                    group_manifest["qa_retry_count"] = int(
                        group_manifest.get("qa_retry_count", 0)
                    ) + 1
                    group_manifest["qa_retry_failed_time_excluded"] = True
                    group_manifest["qa_retry_failed_tokens_excluded"] = True
                    self._write_group_manifest(corpus_id, group_manifest)
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
                        "trace_log_path": result.payload.get("traceLogPath"),
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
            self._write_generated_answers(experiment, records)
            group_manifest["completed_answers"] = len(records)
            self._write_group_manifest(corpus_id, group_manifest)
        self._write_report(experiment, records, ingestion, deletion=None)
        return records, run_id

    def _load_existing_records(
        self, experiment: PreparedExperiment
    ) -> list[dict[str, Any]]:
        output = self._experiment_output(experiment.spec.id)
        candidates = [
            output / "qa_eval_detailed_results.json",
            output / "generated_answers.json",
        ]
        for path in candidates:
            if not path.exists():
                continue
            value = json.loads(path.read_text(encoding="utf-8"))
            records = value.get("results") if isinstance(value, dict) else None
            if not isinstance(records, list):
                raise ValueError(f"Invalid saved QA results: {path}")
            if len(records) > len(experiment.qas):
                raise ValueError(f"Saved QA results exceed experiment size: {path}")
            for index, record in enumerate(records):
                expected_id = experiment.qas[index].get("id")
                if not isinstance(record, dict) or record.get("sample_id") != expected_id:
                    raise ValueError(f"Saved QA results are misaligned at index {index}")
            return records
        return []

    def _write_generated_answers(
        self,
        experiment: PreparedExperiment,
        records: list[dict[str, Any]],
    ) -> None:
        output = self._experiment_output(experiment.spec.id)
        _write_json(
            output / "generated_answers.json",
            {
                "summary": {
                    "dataset": experiment.spec.id,
                    "total_queries": len(records),
                    "complete": len(records) == len(experiment.qas),
                },
                "results": records,
            },
        )

    def _run_judge(
        self,
        experiment: PreparedExperiment,
        records: list[dict[str, Any]],
        ingestion: StageResult,
    ) -> None:
        judge_usage = TokenUsage()
        judge_usage_complete = True
        telemetry_path = self._experiment_output(experiment.spec.id) / "judge_telemetry.json"
        if telemetry_path.exists():
            telemetry = json.loads(telemetry_path.read_text(encoding="utf-8"))
            judge_usage = _known_usage_from_stage_dict(telemetry)
            judge_usage_complete = telemetry.get("usage_complete", True)
        for record in records:
            if "llm_evaluation" in record:
                continue
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
            _write_json(
                output / "qa_eval_detailed_results.json",
                {
                    "complete": all(
                        "llm_evaluation" in candidate for candidate in records
                    ),
                    "results": records,
                },
            )
            _write_json(
                output / "judge_telemetry.json",
                {
                    "usage_complete": judge_usage_complete,
                    "completed_queries": sum(
                        "llm_evaluation" in candidate for candidate in records
                    ),
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
                "Operational Wall Clock Time Including Restarts (s)": (
                    ingestion.payload.get("operationalWallClockSeconds")
                ),
                "Operational Wall Clock Time Complete": ingestion.payload.get(
                    "operationalWallClockComplete", True
                ),
                "Total Source Documents": len(experiment.documents),
                "Token Usage Complete": ingestion.payload.get(
                    "tokenUsageComplete", True
                ),
                "Total Input Tokens": (
                    ingestion.usage.input_tokens
                    if ingestion.payload.get("tokenUsageComplete", True)
                    else None
                ),
                "Total Output Tokens": (
                    ingestion.usage.output_tokens
                    if ingestion.payload.get("tokenUsageComplete", True)
                    else None
                ),
                "Total Embedding Tokens": (
                    ingestion.usage.embedding_tokens
                    if ingestion.payload.get("tokenUsageComplete", True)
                    else None
                ),
                "Total Insertion Token Cost": (
                    ingestion.usage.total
                    if ingestion.payload.get("tokenUsageComplete", True)
                    else None
                ),
                "Known Input Tokens (Lower Bound)": ingestion.usage.input_tokens,
                "Known Output Tokens (Lower Bound)": ingestion.usage.output_tokens,
                "Known Embedding Tokens (Lower Bound)": (
                    ingestion.usage.embedding_tokens
                ),
                "Known Insertion Token Cost (Lower Bound)": ingestion.usage.total,
                "Includes Review Sweep": True,
                "Review Sweep Count": ingestion.payload.get("reviewSweepCount", 1),
                "Ingest Batch Size": ingestion.payload.get("batchSize", 0),
                "Ingest Batch Count": ingestion.payload.get("batchCount", 1),
                "WebKit Restart Count": ingestion.payload.get("restartCount", 0),
                "Planned WebKit Restart Count": ingestion.payload.get(
                    "plannedRestartCount", 0
                ),
                "Retry WebKit Restart Count": ingestion.payload.get(
                    "retryRestartCount", 0
                ),
                "Resume WebKit Restart Count": ingestion.payload.get(
                    "resumeRestartCount", 0
                ),
                "Batch Retry Count": ingestion.payload.get("batchRetryCount", 0),
                "Discarded Retry Time (s)": ingestion.payload.get(
                    "discardedRetryTimeSeconds", 0.0
                ),
                "Discarded Retry Token Usage": ingestion.payload.get(
                    "discardedRetryTokenUsage"
                ),
                "Discarded Retry Token Usage Complete": ingestion.payload.get(
                    "discardedRetryTokenUsageComplete", True
                ),
                "Snapshot Creation Time (s)": ingestion.payload.get(
                    "snapshotCreationTimeSeconds", 0.0
                ),
                "Snapshot Restore Time (s)": ingestion.payload.get(
                    "snapshotRestoreTimeSeconds", 0.0
                ),
                "Snapshot Cleanup Time (s)": ingestion.payload.get(
                    "snapshotCleanupTimeSeconds", 0.0
                ),
                "Snapshot Costs Included In Primary Metrics": False,
                "Snapshot Cleanup Included In Deletion Time": False,
                "Ingest Concurrency": 1,
            },
            "Query Efficiency (Average Per Query)": {
                "Average Retrieval Time (s)": qa_time / count if count else 0.0,
                "Average End-to-End Answer Time (s)": (
                    qa_time / count if count else 0.0
                ),
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
                "Searchable Data Deletion Time (s)": deletion.payload.get(
                    "searchableDeletionDurationSeconds", deletion.duration_seconds
                ),
                "Frontend Quiescence Time (s)": deletion.payload.get(
                    "frontendCleanupDurationSeconds", 0.0
                ),
                "Post-Deletion Cleanup and Recovery Time (s)": deletion.payload.get(
                    "postDeletionCleanupDurationSeconds", 0.0
                ),
                "Only Searchable Data Included In Primary Time": deletion.payload.get(
                    "onlySearchableDataIncludedInPrimaryTime", False
                ),
                "Timed Deletion Scope": deletion.payload.get("timedDeletionScope", []),
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
    usage_complete = stage.payload.get("tokenUsageComplete", True)
    known_usage = stage.usage.as_dict()
    value: dict[str, Any] = {"duration_seconds": stage.duration_seconds}
    if usage_complete:
        value.update(known_usage)
    else:
        value.update({name: None for name in known_usage})
        value.update({f"known_{name}": amount for name, amount in known_usage.items()})
    value["token_usage_complete"] = usage_complete
    wire_names = {
        "operationalWallClockSeconds": "operational_wall_clock_seconds",
        "operationalWallClockComplete": "operational_wall_clock_complete",
        "batchSize": "batch_size",
        "batchCount": "batch_count",
        "restartCount": "restart_count",
        "reviewSweepCount": "review_sweep_count",
        "plannedRestartCount": "planned_restart_count",
        "retryRestartCount": "retry_restart_count",
        "resumeRestartCount": "resume_restart_count",
        "batchRetryCount": "batch_retry_count",
        "discardedRetryTimeSeconds": "discarded_retry_time_seconds",
        "discardedRetryTokenUsage": "discarded_retry_token_usage",
        "discardedRetryTokenUsageComplete": "discarded_retry_token_usage_complete",
        "snapshotCreationTimeSeconds": "snapshot_creation_time_seconds",
        "snapshotRestoreTimeSeconds": "snapshot_restore_time_seconds",
        "snapshotCleanupTimeSeconds": "snapshot_cleanup_time_seconds",
    }
    for wire_name, report_name in wire_names.items():
        if wire_name in stage.payload:
            value[report_name] = stage.payload[wire_name]
    return value


def _known_usage_from_stage_dict(value: dict[str, Any]) -> TokenUsage:
    prefix = "" if value.get("token_usage_complete", True) else "known_"

    def token(field: str) -> int:
        candidate = value.get(f"{prefix}{field}", 0)
        return candidate if isinstance(candidate, int) and not isinstance(candidate, bool) else 0

    return TokenUsage(
        input_tokens=token("input_tokens"),
        output_tokens=token("output_tokens"),
        embedding_tokens=token("embedding_tokens"),
    )


def _stage_from_manifest_dict(value: Any) -> StageResult:
    if not isinstance(value, dict):
        raise ValueError("Resume manifest is missing completed ingestion metrics")
    duration = value.get("duration_seconds")
    if isinstance(duration, bool) or not isinstance(duration, (int, float)):
        raise ValueError("Resume manifest has invalid ingestion duration")
    report_to_wire = {
        "operational_wall_clock_seconds": "operationalWallClockSeconds",
        "batch_size": "batchSize",
        "batch_count": "batchCount",
        "restart_count": "restartCount",
        "review_sweep_count": "reviewSweepCount",
        "planned_restart_count": "plannedRestartCount",
        "retry_restart_count": "retryRestartCount",
        "resume_restart_count": "resumeRestartCount",
        "batch_retry_count": "batchRetryCount",
        "discarded_retry_time_seconds": "discardedRetryTimeSeconds",
        "discarded_retry_token_usage": "discardedRetryTokenUsage",
        "discarded_retry_token_usage_complete": "discardedRetryTokenUsageComplete",
        "snapshot_creation_time_seconds": "snapshotCreationTimeSeconds",
        "snapshot_restore_time_seconds": "snapshotRestoreTimeSeconds",
        "snapshot_cleanup_time_seconds": "snapshotCleanupTimeSeconds",
    }
    payload: dict[str, Any] = {
        "status": "completed",
        "tokenUsageComplete": value.get("token_usage_complete", True),
        "operationalWallClockComplete": value.get(
            "operational_wall_clock_complete",
            value.get("resume_restart_count", 0) == 0,
        ),
    }
    for report_name, wire_name in report_to_wire.items():
        if report_name in value:
            payload[wire_name] = value[report_name]
    return StageResult(
        duration_seconds=float(duration),
        usage=_known_usage_from_stage_dict(value),
        payload=payload,
    )


def _is_incomplete_ingest_telemetry(error: BridgeRequestError) -> bool:
    return (
        error.detail
        == "Provider usage telemetry was incomplete during ingestion"
    )


def _is_retryable_ingest_error(error: BridgeRequestError) -> bool:
    return error.retryable_transport_failure or (
        "truncated wiki file(s) could not be repaired" in error.detail.lower()
    )


def _is_retryable_qa_error(error: BridgeRequestError) -> bool:
    return error.retryable_transport_failure or (
        error.detail == "Provider usage telemetry was incomplete during QA"
    )


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)
