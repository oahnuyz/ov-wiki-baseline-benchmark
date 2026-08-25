from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from ov_wiki_baseline_benchmark.nashsu_llm_wiki.client import LlmWikiBridgeClient


class ClientReadinessTests(unittest.TestCase):
    def _client(self) -> LlmWikiBridgeClient:
        config = SimpleNamespace(
            bridge_base_url="http://127.0.0.1:19828",
            startup_timeout_seconds=2,
            request_timeout_seconds=30,
            bridge_token=lambda: "bridge-token",
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


if __name__ == "__main__":
    unittest.main()
