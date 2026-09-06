"""Four independently runnable rounds for the TencentDB LLM-Wiki baseline."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from ..io import load_jsonl
from ..schema import validate_documents, validate_qas
from ..specs import ExperimentSpec
from .client import MemoryKnowledgeClient
from .config import TencentDBConfig
from .ledger import CallLedger
from .llm import OpenAICompatibleClient
from .pdf import read_source_as_markdown, split_markdown_by_chapter


@dataclass(frozen=True)
class PreparedTencentExperiment:
    spec: ExperimentSpec
    root: Path
    documents: list[dict[str, Any]]
    qas: list[dict[str, Any]]
    corpus_fingerprint: str

    @classmethod
    def load(cls, spec: ExperimentSpec, data_dir: Path) -> "PreparedTencentExperiment":
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


class TencentDBRunner:
    def __init__(self, config: TencentDBConfig, *, answer_prompt_path: Path, judge_prompt_path: Path):
        self.config = config
        self.answer_prompt = answer_prompt_path.read_text(encoding="utf-8")
        self.judge_prompt = judge_prompt_path.read_text(encoding="utf-8")

    def run_group(self, experiments: list[PreparedTencentExperiment], *, round_name: str = "all", retry_failed_pdfs: bool = False) -> dict[str, Any]:
        if not experiments:
            raise ValueError("at least one experiment is required")
        fingerprints = {e.corpus_fingerprint for e in experiments}
        if len(fingerprints) != 1:
            raise ValueError("experiments in a group must have the same corpus")
        corpus_id = f"{experiments[0].spec.dataset}-{experiments[0].corpus_fingerprint[:16]}"
        group_dir = self.config.output_dir / corpus_id
        group_dir.mkdir(parents=True, exist_ok=True)
        ledger = CallLedger(group_dir / "calls.jsonl")
        manifest = self._load_json(group_dir / "manifest.json") or {
            "schema_version": "1.0",
            "corpus_id": corpus_id,
            "experiments": [e.spec.id for e in experiments],
            "config": self.config.manifest(tencentdb_commit=_resolve_tencentdb_commit()),
            "runtime": _runtime_manifest(),
            "status": "created",
        }
        manifest["experiments"] = sorted(
            set(manifest.get("experiments", [])) | {e.spec.id for e in experiments}
        )
        client = MemoryKnowledgeClient(self.config)
        wiki_id = manifest.get("wiki_id")
        if round_name in {"all", "ingest"} and not wiki_id:
            try:
                created = self._service(ledger, "ingest", "wiki.create", client.create_wiki, corpus_id)
                wiki_id = str(created.get("wiki_id") or "")
                if not wiki_id:
                    raise RuntimeError("MemoryKnowledge did not return wiki_id")
                manifest["wiki_id"] = wiki_id
            except Exception as exc:
                manifest["status"] = "create_failed"
                manifest.setdefault("round_errors", []).append({"round": "ingest", "operation": "wiki.create", "error": str(exc)})
                self._write_json(group_dir / "manifest.json", manifest)
                return manifest
        if not wiki_id and round_name != "ingest":
            manifest["status"] = "blocked_no_wiki"
            manifest.setdefault("round_errors", []).append({"round": round_name, "error": "manifest has no wiki_id; run the ingest round first"})
            self._write_json(group_dir / "manifest.json", manifest)
            return manifest
        manifest["status"] = f"running_{round_name}"
        self._write_json(group_dir / "manifest.json", manifest)

        if round_name in {"all", "ingest"}:
            self._run_ingest(experiments[0], wiki_id, group_dir, manifest, ledger, client, retry_failed_pdfs)
            manifest["status"] = "ingested_partial" if manifest.get("partial_ingest") else "ingested"
            manifest.setdefault("rounds", {})["ingest"] = {
                "status": "partial" if manifest.get("partial_ingest") else "success",
                "completed_at": time.time(),
            }
            self._write_json(group_dir / "manifest.json", manifest)
        if round_name in {"all", "qa", "judge"}:
            if round_name in {"all", "qa"}:
                for experiment in experiments:
                    try:
                        self._run_qa(experiment, wiki_id, group_dir, ledger, client)
                    except Exception as exc:
                        manifest.setdefault("round_errors", []).append({"round": "qa", "experiment": experiment.spec.id, "error": str(exc)})
                        self._write_failed_qa_rows(experiment, group_dir, str(exc))
            if round_name in {"all", "judge"}:
                for experiment in experiments:
                    try:
                        self._run_judge(experiment, group_dir, ledger)
                    except Exception as exc:
                        manifest.setdefault("round_errors", []).append({"round": "judge", "experiment": experiment.spec.id, "error": str(exc)})
                        self._write_failed_judge_rows(experiment, group_dir, str(exc))
        if round_name in {"all", "delete"}:
            started = time.perf_counter()
            try:
                delete_result = self._service(ledger, "delete", "wiki.delete", client.delete_wiki, wiki_id)
                failed_items = delete_result.get("failed") if isinstance(delete_result, dict) else None
                deletion_status = "partial" if failed_items else "success"
                error = None if not failed_items else f"delete failed for {len(failed_items)} item(s)"
            except Exception as exc:
                deletion_status = "failed"
                error = str(exc)
            elapsed = time.perf_counter() - started
            manifest["deletion"] = {"status": deletion_status, "time_seconds": elapsed, "error": error, "timed_until": "delete_api_return"}
            manifest["status"] = "completed" if deletion_status == "success" else "delete_failed"
            manifest.setdefault("rounds", {})["delete"] = {"status": deletion_status, "completed_at": time.time()}
            self._write_json(group_dir / "manifest.json", manifest)
            for experiment in experiments:
                self._write_report(experiment, group_dir, ledger, manifest)
        elif round_name in {"ingest", "qa", "judge"}:
            for experiment in experiments:
                self._write_report(experiment, group_dir, ledger, manifest)
        return manifest

    def _run_ingest(self, experiment: PreparedTencentExperiment, wiki_id: str, group_dir: Path, manifest: dict[str, Any], ledger: CallLedger, client: MemoryKnowledgeClient, retry_failed: bool) -> None:
        started = time.perf_counter()
        log_offset = manifest.get("ingest_log_offset")
        if log_offset is None and self.config.service_log_path is not None:
            try:
                log_offset = self.config.service_log_path.stat().st_size
            except OSError:
                log_offset = 0
            manifest["ingest_log_offset"] = log_offset
        prior = {str(x.get("source")): x for x in manifest.get("documents", [])}
        service_failures = self._service_failed_sources(int(manifest.get("ingest_log_offset", 0)))
        if prior and len(prior) == len(experiment.documents) and all(
            item.get("status") == "uploaded" for item in prior.values()
        ) and not service_failures:
            # A completed ingest round is idempotent. Do not perturb the
            # measured insertion wall clock on a no-op resume.
            manifest["documents"] = [prior[str(document["id"])] for document in experiment.documents]
            return
        if service_failures:
            for record in prior.values():
                if str(record.get("source", "")) in service_failures:
                    record["status"] = "ingest_failed"
        converted: list[dict[str, str]] = []
        records: list[dict[str, Any]] = []
        for document, path in zip(experiment.documents, experiment.document_paths):
            key = str(document["id"])
            old = prior.get(key)
            prior_status = old.get("status") if old else None
            if prior_status == "failed":
                # Manifests from the initial draft used a single "failed"
                # state. Treat it as a conversion failure and only retry when
                # explicitly requested for PDFs.
                old = {**old, "status": "conversion_failed"}
                prior_status = "conversion_failed"
            if prior_status in {"uploaded", "upload_failed", "conversion_failed", "ingest_failed"} and not (
                retry_failed and prior_status == "conversion_failed" and path.suffix.lower() == ".pdf"
            ):
                if prior_status == "ingest_failed":
                    pass
                else:
                    records.append(old)
                    continue
            if prior_status == "converted":
                # Compatibility with manifests produced by the first draft.
                records.append(prior[key])
                continue
            try:
                text = read_source_as_markdown(path)
                parts = split_markdown_by_chapter(text, max_bytes=self.config.raw_max_bytes)
                for index, part in enumerate(parts, 1):
                    filename = f"{key}.part-{index:04d}.md" if len(parts) > 1 else f"{key}.md"
                    converted.append({"filename": filename, "content": part, "source": key})
                record = {"source": key, "path": str(path), "status": "converted", "parts": len(parts), "uploaded_parts": 0, "error": None}
            except Exception as exc:
                record = {"source": key, "path": str(path), "status": "conversion_failed", "parts": 0, "uploaded_parts": 0, "error": str(exc)}
            records.append(record)
            manifest["documents"] = list(records)
            self._write_json(group_dir / "manifest.json", {**manifest, "status": "converting"})
        for offset in range(0, len(converted), 10):
            batch = converted[offset : offset + 10]
            try:
                self._service(
                    ledger,
                    "ingest",
                    "wiki.raw.write",
                    client.raw_write,
                    wiki_id,
                    [{"filename": item["filename"], "content": item["content"]} for item in batch],
                    experiment_id=experiment.spec.id,
                )
                by_source = {}
                for item in batch:
                    by_source[item["source"]] = by_source.get(item["source"], 0) + 1
                for record in records:
                    if record["source"] in by_source:
                        record["uploaded_parts"] = int(record.get("uploaded_parts", 0)) + by_source[record["source"]]
                        if record["uploaded_parts"] == record["parts"]:
                            record["status"] = "uploaded"
            except Exception as exc:
                # Keep going; later batches can still be uploaded and accounted for.
                manifest.setdefault("raw_write_failures", []).append({"offset": offset, "sources": sorted({item["source"] for item in batch}), "error": str(exc)})
                for record in records:
                    if record["source"] in {item["source"] for item in batch}:
                        record["status"] = "upload_failed"
            manifest["documents"] = list(records)
            self._write_json(group_dir / "manifest.json", {**manifest, "status": "uploading"})
        if converted:
            by_source_files = {}
            for item in converted:
                by_source_files.setdefault(item["source"], []).append(item)
            retry_sources = set(by_source_files)
            failed_sources_final: set[str] = set()
            attempts = []
            for attempt in range(1, self.config.max_ingest_attempts + 1):
                if attempt > 1:
                    retry_files = [item for source in sorted(retry_sources) for item in by_source_files[source]]
                    try:
                        for batch_offset in range(0, len(retry_files), 10):
                            retry_batch = retry_files[batch_offset : batch_offset + 10]
                            self._service(
                                ledger,
                                "ingest",
                                "wiki.raw.write.retry",
                                client.raw_write,
                                wiki_id,
                                [{"filename": x["filename"], "content": x["content"]} for x in retry_batch],
                                experiment_id=experiment.spec.id,
                                token_bearing=False,
                            )
                    except Exception as exc:
                        attempts.append({"attempt": attempt, "status": "raw_write_failed", "sources": sorted(retry_sources), "error": str(exc)})
                        continue
                try:
                    attempt_log_offset = self._service_log_size()
                    self._service(ledger, "ingest", "wiki.ingest", client.ingest, wiki_id, experiment_id=experiment.spec.id)
                    detail = self._service(ledger, "ingest", "wiki.wait_ready", client.wait_ready, wiki_id, experiment_id=experiment.spec.id)
                    manifest["wiki_status"] = detail.get("status")
                    failed_sources = self._service_failed_sources(attempt_log_offset)
                    retry_sources = retry_sources & failed_sources
                    failed_sources_final = set(retry_sources)
                    attempts.append({"attempt": attempt, "status": "ready" if detail.get("status") == "ready" else "not_ready", "failed_sources": sorted(retry_sources)})
                    if not retry_sources or attempt == self.config.max_ingest_attempts:
                        if detail.get("status") != "ready":
                            manifest["ingest_error"] = f"wiki ended in status {detail.get('status')}"
                        break
                except Exception as exc:
                    attempts.append({"attempt": attempt, "status": "failed", "sources": sorted(retry_sources), "error": str(exc)})
                    failed_sources_final = set(retry_sources)
            for record in records:
                if record["source"] in failed_sources_final:
                    record["status"] = "ingest_failed"
                    record["error"] = "MemoryKnowledge did not generate a legal wiki page after maximum attempts"
                elif record.get("status") == "ingest_failed":
                    record["status"] = "uploaded"
                    record["error"] = None
            manifest["ingest_attempts"] = attempts
        manifest["documents"] = records
        manifest["partial_ingest"] = any(r.get("status") != "uploaded" for r in records) or bool(manifest.get("ingest_error"))
        attempt_seconds = time.perf_counter() - started
        previous_seconds = float(manifest.get("ingest", {}).get("time_seconds", 0.0))
        manifest["ingest"] = {"time_seconds": previous_seconds + attempt_seconds, "last_attempt_seconds": attempt_seconds, "documents_total": len(records), "documents_failed": sum(r.get("status") != "uploaded" for r in records), "source": "raw_conversion_start_to_ready"}
        self._write_json(group_dir / "manifest.json", manifest)

    def _service_failed_sources(self, offset: int = 0) -> set[str]:
        """Read source-level generation failures when the service log is available."""
        path = self.config.service_log_path
        if path is None or not path.is_file():
            return set()
        import re
        failures: set[str] = set()
        try:
            with path.open("rb") as handle:
                handle.seek(offset)
                text = handle.read().decode("utf-8", errors="replace")
            for line in text.splitlines():
                if "runIngest 单源抽取失败" not in line and "未生成任何合法 wiki 页" not in line:
                    continue
                match = re.search(r'"source":"([^"]+)"', line)
                if match:
                    source = match.group(1)
                    failures.add(source[:-3] if source.endswith(".md") else source)
        except OSError:
            return set()
        return failures

    def _service_log_size(self) -> int:
        path = self.config.service_log_path
        if path is None:
            return 0
        try:
            return path.stat().st_size
        except OSError:
            return 0

    def _run_qa(self, experiment: PreparedTencentExperiment, wiki_id: str, group_dir: Path, ledger: CallLedger, client: MemoryKnowledgeClient) -> None:
        tools: list[dict[str, Any]] = []
        try:
            listed = self._service(
                ledger,
                "qa",
                "tools.list",
                client.tools_list,
                wiki_id,
                experiment_id=experiment.spec.id,
            )
            tools = [_tool_schema(item) for item in listed.get("tools", []) if isinstance(item, dict)]
        except Exception:
            tools = []
        llm = OpenAICompatibleClient(self.config, ledger)
        output = group_dir / f"{experiment.spec.id}.qa.jsonl"
        existing = {str(r.get("qa_id")): r for r in _read_jsonl(output)}

        def one(index_qa: tuple[int, dict[str, Any]]) -> dict[str, Any]:
            index, qa = index_qa
            qa_id = str(qa["id"])
            if qa_id in existing:
                return existing[qa_id]
            try:
                prompt = _substitute(
                    self.answer_prompt, {"question": str(qa["question"])}
                )
                answer, elapsed, turns = llm.answer_with_tools(
                    wiki_id=wiki_id,
                    question=prompt,
                    answer_prompt="",
                    tools=tools,
                    call_tool=lambda name, args: self._service(
                        ledger,
                        "qa",
                        f"tools.call.{name}",
                        client.tools_call,
                        wiki_id,
                        name,
                        args,
                        qa_id=qa_id,
                        experiment_id=experiment.spec.id,
                    ),
                    qa_id=qa_id,
                    experiment_id=experiment.spec.id,
                )
                return {
                    "qa_id": qa_id,
                    "question": qa["question"],
                    "answer": answer,
                    "gold_answers": qa["gold_answers"],
                    "duration_seconds": elapsed,
                    "loop_turns": turns,
                    "status": "success" if answer else "failed",
                    "retrieval_available": bool(tools),
                    "failure_reason": None if answer else (
                        "max_loop_turns_without_final_answer"
                        if turns >= self.config.max_loop_turns
                        else "no_final_text_answer"
                    ),
                }
            except Exception as exc:
                return {
                    "qa_id": qa_id,
                    "question": qa.get("question", ""),
                    "answer": "",
                    "gold_answers": qa.get("gold_answers", []),
                    "duration_seconds": 0.0,
                    "loop_turns": 0,
                    "status": "failed",
                    "retrieval_available": bool(tools),
                    "error": str(exc),
                    "failure_reason": "llm_or_agent_exception",
                }

        results: dict[int, dict[str, Any]] = {}
        with ThreadPoolExecutor(max_workers=self.config.qa_workers) as pool:
            futures = {pool.submit(one, item): item[0] for item in enumerate(experiment.qas)}
            for future in as_completed(futures):
                result = future.result()
                results[futures[future]] = result
                if str(result.get("qa_id")) not in existing:
                    _append_jsonl(output, result)
        self._write_json(group_dir / f"{experiment.spec.id}.qa.summary.json", {"count": len(results), "average_duration_seconds": sum(float(r.get("duration_seconds", 0)) for r in results.values()) / max(1, len(results)), "qa_failed": sum(r.get("status") != "success" for r in results.values())})

    def _run_judge(self, experiment: PreparedTencentExperiment, group_dir: Path, ledger: CallLedger) -> None:
        qa_rows = _read_jsonl(group_dir / f"{experiment.spec.id}.qa.jsonl")
        llm = OpenAICompatibleClient(self.config, ledger)
        output = group_dir / f"{experiment.spec.id}.judge.jsonl"
        # Cache entries are valid only for the exact answer that was judged.
        # This prevents stale scores surviving a later QA rerun.
        existing = {
            str(r.get("qa_id")): r
            for r in _read_jsonl(output)
            if r.get("answer_sha256")
        }

        def one(row: dict[str, Any]) -> dict[str, Any]:
            qa_id = str(row.get("qa_id"))
            answer = str(row.get("answer") or "")
            answer_hash = hashlib.sha256(answer.encode("utf-8")).hexdigest()
            if qa_id in existing and existing[qa_id].get("answer_sha256") == answer_hash:
                return existing[qa_id]
            # Failed/empty QA must never be delegated to the judge.  A model
            # judge can hallucinate a positive score for an empty response.
            if row.get("status") != "success" or not answer.strip():
                return {
                    "qa_id": qa_id,
                    "score": 0,
                    "normalized_accuracy": 0.0,
                    "reasoning": "QA failed or returned an empty answer; score forced to 0.",
                    "answer_sha256": answer_hash,
                    "qa_status": str(row.get("status", "")),
                }
            prompt = _substitute(self.judge_prompt, {"question": str(row.get("question", "")), "gold_answers_joined_by_pipe": " | ".join(row.get("gold_answers", [])), "generated_answer": str(row.get("answer", ""))})
            score = 0
            reasoning = "judge failed"
            try:
                payload = llm.complete(
                    messages=[{"role": "user", "content": prompt}],
                    phase="judge",
                    qa_id=qa_id,
                    experiment_id=experiment.spec.id,
                )
                content = _content(payload)
                parsed = json.loads(content)
                score = max(0, min(4, int(parsed.get("score", 0))))
                reasoning = str(parsed.get("reasoning", ""))
            except Exception as exc:
                reasoning = str(exc)
            return {"qa_id": qa_id, "score": score, "normalized_accuracy": score / 4.0, "reasoning": reasoning, "answer_sha256": answer_hash, "qa_status": str(row.get("status", ""))}
        rows_by_id: dict[str, dict[str, Any]] = dict(existing)
        with ThreadPoolExecutor(max_workers=self.config.judge_workers) as pool:
            futures = [pool.submit(one, row) for row in qa_rows]
            for future in as_completed(futures):
                result = future.result()
                qa_id = str(result.get("qa_id"))
                rows_by_id[qa_id] = result
                if not (
                    qa_id in existing
                    and existing[qa_id].get("answer_sha256") == result.get("answer_sha256")
                ):
                    _append_jsonl(output, result)
        rows = [
            rows_by_id[str(qa["id"])]
            for qa in experiment.qas
            if str(qa["id"]) in rows_by_id
        ]
        self._write_json(
            group_dir / f"{experiment.spec.id}.judge.summary.json",
            {
                "count": len(rows),
                "average_score_0_4": sum(float(r.get("score", 0.0)) for r in rows) / max(1, len(experiment.qas)),
                "average_normalized_accuracy": sum(float(r.get("normalized_accuracy", 0.0)) for r in rows) / max(1, len(experiment.qas)),
                "expected_count": len(experiment.qas),
                "missing_count": max(0, len(experiment.qas) - len(rows)),
            },
        )

    def _write_failed_qa_rows(self, experiment: PreparedTencentExperiment, group_dir: Path, error: str) -> None:
        output = group_dir / f"{experiment.spec.id}.qa.jsonl"
        existing = {str(r.get("qa_id")) for r in _read_jsonl(output)}
        for qa in experiment.qas:
            qa_id = str(qa["id"])
            if qa_id in existing:
                continue
            _append_jsonl(output, {"qa_id": qa_id, "question": qa["question"], "answer": "", "gold_answers": qa["gold_answers"], "duration_seconds": 0.0, "loop_turns": 0, "status": "failed", "error": error, "retrieval_available": False})
        rows = _read_jsonl(output)
        self._write_json(group_dir / f"{experiment.spec.id}.qa.summary.json", {"count": len({str(r.get('qa_id')) for r in rows}), "average_duration_seconds": sum(float(r.get("duration_seconds", 0.0)) for r in rows) / max(1, len(experiment.qas)), "qa_failed": sum(r.get("status") != "success" for r in rows), "expected_count": len(experiment.qas)})

    def _write_failed_judge_rows(self, experiment: PreparedTencentExperiment, group_dir: Path, error: str) -> None:
        output = group_dir / f"{experiment.spec.id}.judge.jsonl"
        existing = {str(r.get("qa_id")) for r in _read_jsonl(output)}
        for qa in experiment.qas:
            qa_id = str(qa["id"])
            if qa_id in existing:
                continue
            _append_jsonl(output, {"qa_id": qa_id, "score": 0, "normalized_accuracy": 0.0, "reasoning": error})
        rows = _read_jsonl(output)
        self._write_json(group_dir / f"{experiment.spec.id}.judge.summary.json", {"count": len({str(r.get('qa_id')) for r in rows}), "average_score_0_4": sum(float(r.get("score", 0.0)) for r in rows) / max(1, len(experiment.qas)), "average_normalized_accuracy": sum(float(r.get("normalized_accuracy", 0.0)) for r in rows) / max(1, len(experiment.qas)), "expected_count": len(experiment.qas), "missing_count": 0})

    def _write_report(self, experiment: PreparedTencentExperiment, group_dir: Path, ledger: CallLedger, manifest: dict[str, Any]) -> None:
        ingest = ledger.aggregate(phase="ingest")
        qa = ledger.aggregate(phase="qa", experiment_id=experiment.spec.id)
        judge = ledger.aggregate(phase="judge", experiment_id=experiment.spec.id)
        summary = self._load_json(group_dir / f"{experiment.spec.id}.qa.summary.json") or {}
        judged = self._load_json(group_dir / f"{experiment.spec.id}.judge.summary.json") or {}
        report = {
            "dataset": experiment.spec.id,
            "backend": "TencentDB-Agent-Memory",
            "partial_ingest": bool(manifest.get("partial_ingest")),
            "Insertion Efficiency (Total Dataset)": {"Total Insertion Time (s)": manifest.get("ingest", {}).get("time_seconds", 0), "Total Insertion Token Cost": ingest["total_tokens"], "Token Usage Incomplete Calls": ingest["incomplete_usage_calls"]},
            "Query Efficiency (Average Per Query)": {"Average QA End-to-End Answer Time (s)": summary.get("average_duration_seconds", 0), "Average QA Token Cost": qa["total_tokens"] / max(1, len(experiment.qas))},
            "Performance Metrics": {
                "Average Accuracy (Hit 0-4)": judged.get("average_score_0_4", 0),
                "Average Accuracy (normalization)": judged.get("average_normalized_accuracy", 0),
                "Normalized Accuracy (0-1)": judged.get("average_normalized_accuracy", 0),
            },
            "Deletion Efficiency (Total Dataset)": {"Deletion Time (s)": manifest.get("deletion", {}).get("time_seconds", 0), "Deletion Token Cost": ledger.aggregate(phase="delete")["total_tokens"]},
            "usage_audit": {"ingest": ingest, "qa": qa, "judge": judge},
        }
        self._write_json(group_dir / f"{experiment.spec.id}.benchmark_metrics_report.json", report)

    @staticmethod
    def _service(
        ledger: CallLedger,
        phase: str,
        operation: str,
        fn: Callable[..., dict[str, Any]],
        *args: Any,
        qa_id: str | None = None,
        experiment_id: str | None = None,
        token_bearing: bool | None = None,
    ) -> dict[str, Any]:
        started = time.time()
        if token_bearing is None:
            token_bearing = operation == "wiki.ingest"
        try:
            value = fn(*args)
            usage = value.get("usage") if isinstance(value, dict) else None
            complete = _valid_service_usage(usage) if token_bearing else True
            usage_values = usage if isinstance(usage, dict) else {}
            ledger.append(
                phase=phase,
                operation=operation,
                status="success",
                started_at=started,
                input_tokens=usage_values.get("input_tokens", usage_values.get("inputTokens", 0)) if token_bearing and complete else 0,
                output_tokens=usage_values.get("output_tokens", usage_values.get("outputTokens", 0)) if token_bearing and complete else 0,
                token_usage_complete=complete,
                qa_id=qa_id,
                experiment_id=experiment_id,
                metadata={"token_bearing": token_bearing, "usage_missing_or_invalid": token_bearing and not complete},
            )
            return value
        except Exception as exc:
            ledger.append(
                phase=phase,
                operation=operation,
                status="failed",
                started_at=started,
                token_usage_complete=not token_bearing,
                qa_id=qa_id,
                experiment_id=experiment_id,
                error=str(exc),
                metadata={"token_bearing": token_bearing, "usage_missing_or_invalid": token_bearing},
            )
            raise

    @staticmethod
    def _load_json(path: Path) -> dict[str, Any] | None:
        try:
            value = json.loads(path.read_text(encoding="utf-8")); return value if isinstance(value, dict) else None
        except (OSError, json.JSONDecodeError):
            return None

    @staticmethod
    def _write_json(path: Path, value: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp"); tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"); os.replace(tmp, path)


def _append_jsonl(path: Path, value: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n"); handle.flush(); os.fsync(handle.fileno())


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists(): return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            value = json.loads(line)
            if isinstance(value, dict): out.append(value)
        except json.JSONDecodeError: pass
    return out


def _tool_schema(item: dict[str, Any]) -> dict[str, Any]:
    properties: dict[str, Any] = {}
    required: list[str] = []
    params = item.get("params") if isinstance(item.get("params"), dict) else {}
    for name, spec in params.items():
        spec = spec if isinstance(spec, dict) else {}
        prop: dict[str, Any] = {"type": spec.get("type", "string")}
        if spec.get("description"): prop["description"] = spec["description"]
        if spec.get("enum"): prop["enum"] = spec["enum"]
        properties[str(name)] = prop
        if spec.get("required"): required.append(str(name))
    return {"type": "function", "function": {"name": str(item.get("name", "")), "description": str(item.get("description", "")), "parameters": {"type": "object", "properties": properties, "required": required}}}


def _substitute(template: str, values: dict[str, str]) -> str:
    for key, value in values.items(): template = template.replace("{" + key + "}", value)
    return template


def _content(payload: dict[str, Any]) -> str:
    try: return str(payload["choices"][0]["message"].get("content") or "")
    except (KeyError, IndexError, TypeError): return ""


def _valid_service_usage(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    keys = ("input_tokens", "output_tokens")
    aliases = ("inputTokens", "outputTokens")
    return all(
        isinstance(value.get(key, value.get(alias)), int)
        and not isinstance(value.get(key, value.get(alias)), bool)
        and value.get(key, value.get(alias)) >= 0
        for key, alias in zip(keys, aliases)
    )


def _runtime_manifest() -> dict[str, str]:
    result = {
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "implementation": platform.python_implementation(),
    }
    try:
        result["benchmark_commit"] = subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True,
            text=True, timeout=5,
        ).stdout.strip()
    except Exception:
        result["benchmark_commit"] = "unresolved"
    try:
        from importlib.metadata import version
        for package in ("requests", "PyYAML", "pdfplumber"):
            try:
                result[f"package_{package.lower()}"] = version(package)
            except Exception:
                result[f"package_{package.lower()}"] = "unavailable"
    except Exception:
        pass
    return result


def _resolve_tencentdb_commit() -> str:
    env = os.environ.get("TENCENTDB_COMMIT", "").strip()
    if env: return env
    try:
        result = subprocess.run(["git", "ls-remote", "https://github.com/TencentCloud/tencentdb-agent-memory.git", "refs/heads/feat/server_team"], check=True, capture_output=True, text=True, timeout=20)
        return result.stdout.split()[0]
    except Exception:
        return "unresolved"
