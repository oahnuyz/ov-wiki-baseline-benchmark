from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from ov_wiki_baseline_benchmark.nashsu_llm_wiki.config import BenchmarkConfig
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
    def __init__(self) -> None:
        self.prompts: list[str] = []
        self.sessions: list[str] = []
        self.deleted = False
        self.project_paths: list[Path] = []
        self.ready_paths: list[Path] = []

    def wait_until_ready(self, project_path: Path) -> None:
        self.ready_paths.append(project_path)

    def create_run(self, *, corpus_id: str, project_path: Path) -> RunInfo:
        self.project_paths.append(project_path)
        return RunInfo(
            "run-1",
            {
                "template": "general",
                "outputLanguage": "auto",
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

    def ingest(self, run_id: str, documents: list[Path]) -> StageResult:
        return StageResult(5.0, TokenUsage(10, 2, 30), {"status": "completed"})

    def answer(self, run_id: str, *, prompt: str, session_id: str) -> QaResult:
        self.prompts.append(prompt)
        self.sessions.append(session_id)
        return QaResult(
            answer="Padella",
            session_id=session_id,
            duration_seconds=2.0,
            usage=QaTokenUsage(8, 1, 3, 0, 4),
            payload={"references": ["wiki/padella.md"], "trace": []},
        )

    def delete(self, run_id: str) -> StageResult:
        self.deleted = True
        return StageResult(0.5, TokenUsage(), {"status": "completed"})


class FakeJudge:
    def grade(self, question: str, gold_answers: list[str], answer: str) -> JudgeResult:
        return JudgeResult(4, "Correct.", "Generic_0-4", TokenUsage(8, 2), "")


class RunnerTests(unittest.TestCase):
    def test_full_group_writes_aligned_metrics_and_isolated_sessions(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            experiment = _prepared_experiment(root)
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
            self.assertEqual(bridge.project_paths, [root / "project"])
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
                42,
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
            self.assertEqual(group_manifest["project_scaffold"]["outputLanguage"], "auto")


def _prepared_experiment(root: Path) -> PreparedExperiment:
    corpus = root / "prepared" / "fixture" / "corpus"
    corpus.mkdir(parents=True)
    document_path = corpus / "document.txt"
    document_path.write_text("Padella is king.", encoding="utf-8")
    sha = hashlib.sha256(document_path.read_bytes()).hexdigest()
    document: dict[str, Any] = {
        "path": "corpus/document.txt",
        "sha256": sha,
    }
    qas = [
        {"id": "q1", "question": "Who is king?", "gold_answers": ["Padella"]},
        {"id": "q2", "question": "Name the king.", "gold_answers": ["Padella"]},
    ]
    spec = ExperimentSpec(
        id="fixture",
        dataset="fixture",
        raw_dataset="fixture",
        expected_qas=2,
        expected_documents=1,
        options={},
    )
    return PreparedExperiment(
        spec=spec,
        root=root / "prepared" / "fixture",
        documents=[document],
        qas=qas,
        corpus_fingerprint=sha,
    )


if __name__ == "__main__":
    unittest.main()
