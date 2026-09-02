from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from ov_wiki_baseline_benchmark.nashsu_llm_wiki.config import BenchmarkConfig
from ov_wiki_baseline_benchmark.nashsu_llm_wiki.client import BridgeRequestError
from ov_wiki_baseline_benchmark.nashsu_llm_wiki.judge import JudgeResult
from ov_wiki_baseline_benchmark.nashsu_llm_wiki.models import (
    QaResult,
    QaTokenUsage,
    RunInfo,
    StageResult,
    TokenUsage,
)
from ov_wiki_baseline_benchmark.nashsu_llm_wiki.runner import (
    BenchmarkRunner,
    PreparedExperiment,
)
from ov_wiki_baseline_benchmark.specs import ExperimentSpec, repository_root


class FakeBridge:
    def __init__(
        self,
        *,
        fail_first_ingest: bool = False,
        first_ingest_error_detail: str = "Generation failed: error sending request for url",
        fail_first_qa: bool = False,
        incomplete_telemetry_offsets: set[int] | None = None,
    ) -> None:
        self.prompts: list[str] = []
        self.sessions: list[str] = []
        self.deleted = False
        self.project_paths: list[Path] = []
        self.ready_paths: list[Path] = []
        self.ingest_calls: list[tuple[int, bool, int]] = []
        self.restart_count = 0
        self.stop_count = 0
        self.failures_remaining = 1 if fail_first_ingest else 0
        self.first_ingest_error_detail = first_ingest_error_detail
        self.qa_failures_remaining = 1 if fail_first_qa else 0
        self.incomplete_telemetry_offsets = set(incomplete_telemetry_offsets or set())

    def wait_until_ready(self, project_path: Path) -> None:
        self.ready_paths.append(project_path)

    def create_run(
        self,
        *,
        corpus_id: str,
        project_path: Path,
        continuation: bool = False,
    ) -> RunInfo:
        self.project_paths.append(project_path)
        return RunInfo(
            f"run-{len(self.project_paths)}",
            {
                "template": "general",
                "outputLanguage": "English",
                "chunking": "official_default",
                "persistExtractedMarkdown": False,
                "fileSha256": {
                    path: "0" * 64
                    for path in (
                        "purpose.md",
                        "schema.md",
                        "wiki/index.md",
                        "wiki/overview.md",
                        "wiki/log.md",
                    )
                },
            },
        )

    def ingest(
        self,
        run_id: str,
        documents: list[Path],
        *,
        document_offset: int = 0,
        final_batch: bool = True,
    ) -> StageResult:
        self.ingest_calls.append((document_offset, final_batch, len(documents)))
        if self.failures_remaining:
            self.failures_remaining -= 1
            (self.project_paths[-1] / "partial-write.txt").write_text(
                "must be rolled back", encoding="utf-8"
            )
            raise BridgeRequestError(
                status_code=500,
                detail=self.first_ingest_error_detail,
            )
        if document_offset in self.incomplete_telemetry_offsets:
            self.incomplete_telemetry_offsets.remove(document_offset)
            return StageResult(
                5.0,
                TokenUsage(7, 1, 9),
                {"status": "completed", "tokenUsageComplete": False},
            )
        return StageResult(5.0, TokenUsage(10, 2, 30), {"status": "completed"})

    def restart_service(self, project_path: Path) -> None:
        self.restart_count += 1

    def stop_service(self) -> None:
        self.stop_count += 1

    def answer(self, run_id: str, *, prompt: str, session_id: str) -> QaResult:
        self.prompts.append(prompt)
        self.sessions.append(session_id)
        if self.qa_failures_remaining:
            self.qa_failures_remaining -= 1
            raise BridgeRequestError(status_code=None, detail="Read timed out")
        return QaResult(
            answer="Padella",
            session_id=session_id,
            duration_seconds=2.0,
            usage=QaTokenUsage(8, 1, 3, 0, 4),
            payload={"references": ["wiki/padella.md"], "trace": []},
        )

    def delete(self, run_id: str) -> StageResult:
        self.deleted = True
        return StageResult(
            0.5,
            TokenUsage(),
            {
                "status": "completed",
                "searchableDeletionDurationSeconds": 0.5,
                "frontendCleanupDurationSeconds": 0.1,
                "postDeletionCleanupDurationSeconds": 0.2,
                "onlySearchableDataIncludedInPrimaryTime": True,
                "timedDeletionScope": [
                    "wiki-pages",
                    "raw-source-search-data",
                    "lancedb-vectors",
                ],
            },
        )


class FakeJudge:
    def grade(self, question: str, gold_answers: list[str], answer: str) -> JudgeResult:
        return JudgeResult(4, "Correct.", "Generic_0-4", TokenUsage(8, 2), "")


class RunnerTests(unittest.TestCase):
    def test_skip_deletion_preserves_project_and_marks_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            experiment = _prepared_experiment(root)
            (root / "project").mkdir()
            bridge = FakeBridge()
            runner = BenchmarkRunner(
                _config(root),
                answer_prompt_path=repository_root()
                / "prompts"
                / "ov_wiki_bot_answer.txt",
                judge_prompt_path=repository_root()
                / "prompts"
                / "generic_llm_judge_user.txt",
                bridge=bridge,
                judge=FakeJudge(),
            )

            manifest = runner.run_group([experiment], skip_deletion=True)

            self.assertFalse(bridge.deleted)
            self.assertEqual(manifest["status"], "completed_without_deletion")
            self.assertTrue(manifest["deletion_skipped"])
            self.assertEqual(manifest["preserved_project_path"], str(root / "project"))

    def test_retryable_qa_failure_restarts_and_excludes_failed_attempt(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            experiment = _prepared_experiment(root)
            (root / "project").mkdir()
            config = _config(root)
            bridge = FakeBridge(fail_first_qa=True)
            runner = BenchmarkRunner(
                config,
                answer_prompt_path=repository_root() / "prompts" / "ov_wiki_bot_answer.txt",
                judge_prompt_path=repository_root() / "prompts" / "generic_llm_judge_user.txt",
                bridge=bridge,
                judge=FakeJudge(),
            )

            manifest = runner.run_group([experiment])

            self.assertEqual(manifest["qa_retry_count"], 1)
            self.assertTrue(manifest["qa_retry_failed_time_excluded"])
            self.assertTrue(manifest["qa_retry_failed_tokens_excluded"])
            self.assertEqual(bridge.restart_count, 2)  # one ingest boundary + one QA recovery
            audit = json.loads(
                (
                    root
                    / "output"
                    / "fixture"
                    / "wiki"
                    / "qa_retry_audit.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(len(audit), 1)
            self.assertTrue(audit[0]["excluded_from_primary_qa_metrics"])
            self.assertIsNone(audit[0]["failed_attempt_token_usage"])
            self.assertEqual(audit[0]["replacement_run_id"], "run-3")

    def test_full_group_writes_aligned_metrics_and_isolated_sessions(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            experiment = _prepared_experiment(root)
            (root / "project").mkdir()
            config = BenchmarkConfig(
                bridge_base_url="http://127.0.0.1:19828",
                bridge_token_env="TOKEN",
                ark_api_key_env="ARK_API_KEY",
                model="doubao-seed-2-0-lite-260428",
                model_provider="volcengine",
                model_base_url="https://ark.cn-beijing.volces.com/api/v3",
                embedding_model="doubao-embedding-vision-251215",
                embedding_provider="volcengine",
                embedding_base_url="https://ark.cn-beijing.volces.com/api/v3",
                embedding_dimensions=1024,
                embedding_input="multimodal",
                output_dir=root / "output",
                project_path=root / "project",
                startup_timeout_seconds=30,
                request_timeout_seconds=30,
                ingest_batch_size=1,
                max_batch_retries=2,
                restart_between_ingest_batches=True,
                service_restart_command=("fake-restart",),
                service_stop_command=("fake-stop",),
                snapshot_root=root / "snapshots",
            )
            bridge = FakeBridge()
            runner = BenchmarkRunner(
                config,
                answer_prompt_path=repository_root() / "prompts" / "ov_wiki_bot_answer.txt",
                judge_prompt_path=repository_root() / "prompts" / "generic_llm_judge_user.txt",
                bridge=bridge,
                judge=FakeJudge(),
            )
            runner.run_group([experiment])

            self.assertTrue(bridge.deleted)
            self.assertEqual(bridge.ready_paths, [root / "project"])
            self.assertEqual(bridge.project_paths, [root / "project", root / "project"])
            self.assertEqual(bridge.ingest_calls, [(0, False, 1), (1, True, 1)])
            self.assertEqual(bridge.restart_count, 1)
            self.assertEqual(len(set(bridge.sessions)), 2)
            self.assertEqual(
                bridge.prompts[0],
                "Answer this question as briefly as possible. Use only the information "
                "available in the database. Do not use any external source.\n\n"
                "Question: Who is king?\n",
            )
            report_path = root / "output" / "fixture" / "wiki" / "benchmark_metrics_report.json"
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(
                report["Insertion Efficiency (Total Dataset)"]["Total Insertion Token Cost"],
                84,
            )
            self.assertEqual(
                report["Query Efficiency (Average Per Query)"]["Average Retrieval Token Cost"],
                16,
            )
            self.assertEqual(report["Performance Metrics"]["Normalized Accuracy (0-1)"], 1)
            self.assertEqual(
                report["Deletion Efficiency (Total Dataset)"]["Total Deletion Token Cost"],
                0,
            )
            deletion_report = report["Deletion Efficiency (Total Dataset)"]
            self.assertEqual(deletion_report["Searchable Data Deletion Time (s)"], 0.5)
            self.assertEqual(deletion_report["Frontend Quiescence Time (s)"], 0.1)
            self.assertEqual(
                deletion_report["Post-Deletion Cleanup and Recovery Time (s)"], 0.2
            )
            self.assertTrue(
                deletion_report["Only Searchable Data Included In Primary Time"]
            )
            group_manifest = json.loads(
                (
                    root
                    / "output"
                    / "groups"
                    / f"fixture-{experiment.corpus_fingerprint[:16]}"
                    / "run.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(group_manifest["project_scaffold"]["template"], "general")
            self.assertEqual(group_manifest["project_scaffold"]["outputLanguage"], "English")

    def test_retry_restores_batch_snapshot_and_excludes_failed_attempt(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            experiment = _prepared_experiment(root)
            project = root / "project"
            project.mkdir()
            (project / "marker.txt").write_text("clean", encoding="utf-8")
            config = BenchmarkConfig(
                bridge_base_url="http://127.0.0.1:19828",
                bridge_token_env="TOKEN",
                ark_api_key_env="ARK_API_KEY",
                model="doubao-seed-2-0-lite-260428",
                model_provider="volcengine",
                model_base_url="https://ark.cn-beijing.volces.com/api/v3",
                embedding_model="doubao-embedding-vision-251215",
                embedding_provider="volcengine",
                embedding_base_url="https://ark.cn-beijing.volces.com/api/v3",
                embedding_dimensions=1024,
                embedding_input="multimodal",
                output_dir=root / "output",
                project_path=project,
                startup_timeout_seconds=30,
                request_timeout_seconds=30,
                ingest_batch_size=1,
                max_batch_retries=2,
                restart_between_ingest_batches=True,
                service_restart_command=("fake-restart",),
                service_stop_command=("fake-stop",),
                snapshot_root=root / "snapshots",
            )
            bridge = FakeBridge(fail_first_ingest=True)
            runner = BenchmarkRunner(
                config,
                answer_prompt_path=repository_root() / "prompts" / "ov_wiki_bot_answer.txt",
                judge_prompt_path=repository_root() / "prompts" / "generic_llm_judge_user.txt",
                bridge=bridge,
                judge=FakeJudge(),
            )
            runner.run_group([experiment])

            self.assertEqual(
                bridge.ingest_calls,
                [(0, False, 1), (0, False, 1), (1, True, 1)],
            )
            self.assertEqual(bridge.stop_count, 1)
            self.assertEqual(bridge.restart_count, 2)
            self.assertFalse((project / "partial-write.txt").exists())
            self.assertFalse(any((root / "snapshots").iterdir()))
            report = json.loads(
                (
                    root
                    / "output"
                    / "fixture"
                    / "wiki"
                    / "benchmark_metrics_report.json"
                ).read_text(encoding="utf-8")
            )
            insertion = report["Insertion Efficiency (Total Dataset)"]
            self.assertEqual(insertion["Total Insertion Time (s)"], 10.0)
            self.assertEqual(insertion["Total Insertion Token Cost"], 84)
            self.assertEqual(insertion["Batch Retry Count"], 1)
            self.assertIsNone(insertion["Discarded Retry Token Usage"])
            self.assertFalse(insertion["Snapshot Cleanup Included In Deletion Time"])

    def test_unrepaired_truncated_wiki_file_retries_the_entire_batch(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            experiment = _prepared_experiment(root)
            project = root / "project"
            project.mkdir()
            (project / "marker.txt").write_text("clean", encoding="utf-8")
            bridge = FakeBridge(
                fail_first_ingest=True,
                first_ingest_error_detail=(
                    "Benchmark ingestion failed for 1 source(s): Ingest incomplete: "
                    "1 truncated wiki file(s) could not be repaired: "
                    "wiki/concepts/incomplete.md"
                ),
            )
            runner = BenchmarkRunner(
                _config(root),
                answer_prompt_path=repository_root()
                / "prompts"
                / "ov_wiki_bot_answer.txt",
                judge_prompt_path=repository_root()
                / "prompts"
                / "generic_llm_judge_user.txt",
                bridge=bridge,
                judge=FakeJudge(),
            )

            manifest = runner.run_group(
                [experiment],
                skip_deletion=True,
            )

            self.assertEqual(
                bridge.ingest_calls,
                [(0, False, 1), (0, False, 1), (1, True, 1)],
            )
            self.assertEqual(bridge.stop_count, 1)
            self.assertEqual(bridge.restart_count, 2)
            self.assertFalse((project / "partial-write.txt").exists())
            self.assertEqual(manifest["status"], "completed_without_deletion")
            self.assertTrue(manifest["discarded_ingest_attempts"][0]["retryable"])

    def test_incomplete_ingest_usage_preserves_and_sums_known_tokens(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            experiment = _prepared_experiment(root)
            (root / "project").mkdir()
            config = _config(root)
            bridge = FakeBridge(incomplete_telemetry_offsets={0})
            runner = BenchmarkRunner(
                config,
                answer_prompt_path=repository_root()
                / "prompts"
                / "ov_wiki_bot_answer.txt",
                judge_prompt_path=repository_root()
                / "prompts"
                / "generic_llm_judge_user.txt",
                bridge=bridge,
                judge=FakeJudge(),
            )

            runner.run_group([experiment])

            self.assertEqual(bridge.ingest_calls, [(0, False, 1), (1, True, 1)])
            report = json.loads(
                (
                    root
                    / "output"
                    / "fixture"
                    / "wiki"
                    / "benchmark_metrics_report.json"
                ).read_text(encoding="utf-8")
            )
            insertion = report["Insertion Efficiency (Total Dataset)"]
            self.assertFalse(insertion["Token Usage Complete"])
            self.assertIsNone(insertion["Total Insertion Token Cost"])
            self.assertEqual(insertion["Known Input Tokens (Lower Bound)"], 17)
            self.assertEqual(insertion["Known Output Tokens (Lower Bound)"], 3)
            self.assertEqual(insertion["Known Embedding Tokens (Lower Bound)"], 39)
            self.assertEqual(insertion["Known Insertion Token Cost (Lower Bound)"], 59)

    def test_resume_recovers_immediately_preceding_incomplete_telemetry_batch(self) -> None:
        retry_records = [
            {
                "batch_index": 1,
                "attempt": 1,
                "run_id": "run-2",
                "duration_seconds": 12.5,
                "error": "Provider usage telemetry was incomplete during ingestion",
            }
        ]

        recovered = BenchmarkRunner._recover_completed_incomplete_telemetry_batch(
            batches=[[Path("a")], [Path("b")], [Path("c")]],
            batch_records=[
                {
                    "batch_index": 0,
                    "document_count": 1,
                    "duration_seconds": 4.0,
                }
            ],
            retry_records=retry_records,
        )

        self.assertIsNotNone(recovered)
        assert recovered is not None
        self.assertEqual(recovered["batch_index"], 1)
        self.assertEqual(recovered["document_offset"], 1)
        self.assertFalse(recovered["token_usage_complete"])
        self.assertEqual(retry_records, [])

    def test_resume_after_ingestion_reuses_saved_answers(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            experiment = _prepared_experiment(root)
            (root / "project").mkdir()
            config = _config(root)
            initial_bridge = FakeBridge()
            initial_runner = BenchmarkRunner(
                config,
                answer_prompt_path=repository_root()
                / "prompts"
                / "ov_wiki_bot_answer.txt",
                judge_prompt_path=repository_root()
                / "prompts"
                / "generic_llm_judge_user.txt",
                bridge=initial_bridge,
                judge=FakeJudge(),
            )
            initial_runner.run_group([experiment])
            group_path = (
                root
                / "output"
                / "groups"
                / f"fixture-{experiment.corpus_fingerprint[:16]}"
                / "run.json"
            )
            manifest = json.loads(group_path.read_text(encoding="utf-8"))
            manifest["status"] = "ingested"
            manifest.pop("deletion", None)
            group_path.write_text(json.dumps(manifest), encoding="utf-8")
            output = root / "output" / "fixture" / "wiki"
            (output / "qa_eval_detailed_results.json").unlink()
            (output / "judge_telemetry.json").unlink()

            resumed_bridge = FakeBridge()
            resumed_runner = BenchmarkRunner(
                config,
                answer_prompt_path=repository_root()
                / "prompts"
                / "ov_wiki_bot_answer.txt",
                judge_prompt_path=repository_root()
                / "prompts"
                / "generic_llm_judge_user.txt",
                bridge=resumed_bridge,
                judge=FakeJudge(),
            )
            resumed_runner.run_group([experiment], resume_ingest=True)

            self.assertEqual(resumed_bridge.ingest_calls, [])
            self.assertEqual(resumed_bridge.prompts, [])
            self.assertTrue(resumed_bridge.deleted)
            resumed_manifest = json.loads(group_path.read_text(encoding="utf-8"))
            self.assertEqual(resumed_manifest["status"], "completed")


def _prepared_experiment(root: Path) -> PreparedExperiment:
    corpus = root / "prepared" / "fixture" / "corpus"
    corpus.mkdir(parents=True)
    document_path = corpus / "document.txt"
    document_path.write_text("Padella is king.", encoding="utf-8")
    sha = hashlib.sha256(document_path.read_bytes()).hexdigest()
    second_path = corpus / "second.txt"
    second_path.write_text("The king is Padella.", encoding="utf-8")
    second_sha = hashlib.sha256(second_path.read_bytes()).hexdigest()
    documents: list[dict[str, Any]] = [
        {"path": "corpus/document.txt", "sha256": sha},
        {"path": "corpus/second.txt", "sha256": second_sha},
    ]
    qas = [
        {"id": "q1", "question": "Who is king?", "gold_answers": ["Padella"]},
        {"id": "q2", "question": "Name the king.", "gold_answers": ["Padella"]},
    ]
    spec = ExperimentSpec(
        id="fixture",
        dataset="fixture",
        raw_dataset="fixture",
        expected_qas=2,
        expected_documents=2,
        options={},
    )
    return PreparedExperiment(
        spec=spec,
        root=root / "prepared" / "fixture",
        documents=documents,
        qas=qas,
        corpus_fingerprint=sha,
    )


def _config(root: Path) -> BenchmarkConfig:
    return BenchmarkConfig(
        bridge_base_url="http://127.0.0.1:19828",
        bridge_token_env="TOKEN",
        ark_api_key_env="ARK_API_KEY",
        model="doubao-seed-2-0-lite-260428",
        model_provider="volcengine",
        model_base_url="https://ark.cn-beijing.volces.com/api/v3",
        embedding_model="doubao-embedding-vision-251215",
        embedding_provider="volcengine",
        embedding_base_url="https://ark.cn-beijing.volces.com/api/v3",
        embedding_dimensions=1024,
        embedding_input="multimodal",
        output_dir=root / "output",
        project_path=root / "project",
        startup_timeout_seconds=30,
        request_timeout_seconds=30,
        ingest_batch_size=1,
        max_batch_retries=2,
        restart_between_ingest_batches=True,
        service_restart_command=("fake-restart",),
        service_stop_command=("fake-stop",),
        snapshot_root=root / "snapshots",
    )


if __name__ == "__main__":
    unittest.main()
