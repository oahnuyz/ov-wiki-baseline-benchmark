from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from ov_wiki_baseline_benchmark.nashsu_llm_wiki.client import (
    BridgeRequestError,
    LlmWikiBridgeClient,
)


class ClientReadinessTests(unittest.TestCase):
    def _client(self) -> LlmWikiBridgeClient:
        config = SimpleNamespace(
            bridge_base_url="http://127.0.0.1:19828",
            startup_timeout_seconds=2,
            request_timeout_seconds=30,
            qa_timeout_seconds=7,
            service_restart_command=("restart-service",),
            service_stop_command=("stop-service",),
            bridge_token=lambda: "bridge-token",
            public_manifest=lambda: {"fixed": True},
        )
        return LlmWikiBridgeClient(config)  # type: ignore[arg-type]

    def test_wait_until_ready_requires_the_expected_project(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            project = Path(name) / "project"
            response = Mock(status_code=200)
            response.json.return_value = {
                "ready": True,
                "projectPath": str(project),
            }
            with patch("requests.get", return_value=response):
                self._client().wait_until_ready(project)

    def test_wait_until_ready_rejects_a_different_project(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            project = Path(name) / "project"
            response = Mock(status_code=200)
            response.json.return_value = {
                "ready": True,
                "projectPath": str(Path(name) / "other"),
            }
            with patch("requests.get", return_value=response):
                with self.assertRaisesRegex(RuntimeError, "expected"):
                    self._client().wait_until_ready(project)

    def test_ingest_sends_batch_boundary_and_requires_matching_sweep_state(self) -> None:
        client = self._client()
        response = {
            "status": "completed",
            "durationSeconds": 3.0,
            "usage": {"inputTokens": 4, "outputTokens": 2, "embeddingTokens": 6},
            "reviewSweepCompleted": False,
            "embeddingDimensions": 1024,
        }
        with patch.object(client, "_request", return_value=response) as request:
            result = client.ingest(
                "run-1",
                [Path("/data/a.pdf")],
                document_offset=25,
                final_batch=False,
            )
        self.assertEqual(result.usage.total, 12)
        self.assertEqual(
            request.call_args.args[2],
            {
                "documents": ["/data/a.pdf"],
                "documentOffset": 25,
                "finalBatch": False,
                "wait": True,
            },
        )

    def test_create_run_marks_restart_continuation_explicitly(self) -> None:
        client = self._client()
        response = {
            "runId": "run-2",
            "resolvedMaxContextSize": 204800,
            "projectScaffold": {
                "template": "general",
                "outputLanguage": "auto",
                "chunking": "official_default",
                "persistExtractedMarkdown": False,
                "continued": True,
                "fileSha256": {
                    path: "a" * 64
                    for path in (
                        "purpose.md",
                        "schema.md",
                        "wiki/index.md",
                        "wiki/overview.md",
                        "wiki/log.md",
                    )
                },
            },
        }
        with patch.object(client, "_request", return_value=response) as request:
            run = client.create_run(
                corpus_id="paperscope-fingerprint",
                project_path=Path("/project"),
                continuation=True,
            )
        self.assertEqual(run.run_id, "run-2")
        self.assertIs(request.call_args.args[2]["continuation"], True)

    def test_request_exposes_retryable_bridge_error_detail(self) -> None:
        response = Mock(
            ok=False,
            status_code=500,
            text='{"error":"Generation failed: error sending request for url"}',
            reason="Internal Server Error",
        )
        response.json.return_value = {
            "error": "Generation failed: error sending request for url"
        }
        with patch("requests.request", return_value=response):
            with self.assertRaises(BridgeRequestError) as raised:
                self._client()._request("POST", "/runs/1/ingest", {})
        self.assertTrue(raised.exception.retryable_transport_failure)

    def test_answer_uses_the_shorter_per_question_timeout(self) -> None:
        client = self._client()
        response = {
            "status": "completed",
            "answer": "Padella",
            "sessionId": "session-1",
            "durationSeconds": 1.0,
            "resolvedTopK": 5,
            "usage": {
                "inputTokens": 4,
                "outputTokens": 2,
                "embeddingTokens": 1,
                "agentInputTokens": 4,
                "agentOutputTokens": 2,
                "searchInputTokens": 0,
                "searchOutputTokens": 0,
            },
        }
        with patch.object(client, "_request", return_value=response) as request:
            client.answer("run-1", prompt="Question", session_id="session-1")
        self.assertEqual(request.call_args.kwargs["timeout_seconds"], 7)


if __name__ == "__main__":
    unittest.main()
