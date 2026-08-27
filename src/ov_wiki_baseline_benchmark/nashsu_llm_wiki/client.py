"""HTTP client for the benchmark-only LLM Wiki bridge endpoints."""

from __future__ import annotations

import subprocess
import time
from pathlib import Path
from typing import Any

import requests

from .config import BenchmarkConfig
from .models import QaResult, RunInfo, StageResult


class LlmWikiBridgeClient:
    def __init__(self, config: BenchmarkConfig) -> None:
        self.config = config
        self.base_url = f"{config.bridge_base_url}/api/v1/benchmark"
        self.headers = {
            "Authorization": f"Bearer {config.bridge_token()}",
            "Content-Type": "application/json",
        }

    def wait_until_ready(self, project_path: Path) -> None:
        deadline = time.monotonic() + self.config.startup_timeout_seconds
        expected = str(project_path.resolve())
        last_error = "benchmark frontend has not responded"
        while time.monotonic() < deadline:
            try:
                response = requests.get(
                    f"{self.base_url}/ready",
                    headers=self.headers,
                    timeout=min(10, self.config.startup_timeout_seconds),
                )
                if response.status_code == 200:
                    value = response.json()
                    if not isinstance(value, dict) or value.get("ready") is not True:
                        raise RuntimeError("Bridge readiness response is not ready")
                    actual = str(Path(str(value.get("projectPath", ""))).resolve())
                    if actual != expected:
                        raise RuntimeError(
                            f"Bridge opened {actual!r}, expected {expected!r}"
                        )
                    return
                if response.status_code != 503:
                    response.raise_for_status()
                last_error = response.text.strip() or f"HTTP {response.status_code}"
            except (requests.RequestException, ValueError) as exc:
                last_error = str(exc)
            time.sleep(1)
        raise RuntimeError(
            "LLM Wiki benchmark frontend did not become ready within "
            f"{self.config.startup_timeout_seconds}s: {last_error}"
        )

    def create_run(
        self,
        *,
        corpus_id: str,
        project_path: Path,
        continuation: bool = False,
    ) -> RunInfo:
        value = self._request(
            "POST",
            "/runs",
            {
                "corpusId": corpus_id,
                "projectPath": str(project_path),
                "continuation": continuation,
                "config": self.config.public_manifest(),
            },
        )
        return RunInfo.from_wire(value)

    def ingest(
        self,
        run_id: str,
        documents: list[Path],
        *,
        document_offset: int = 0,
        final_batch: bool = True,
    ) -> StageResult:
        value = self._request(
            "POST",
            f"/runs/{run_id}/ingest",
            {
                "documents": [str(path) for path in documents],
                "documentOffset": document_offset,
                "finalBatch": final_batch,
                "wait": True,
            },
        )
        stage = StageResult.from_wire(value, context="ingestion")
        if value.get("reviewSweepCompleted") is not final_batch:
            raise RuntimeError(
                "Ingestion response review sweep state does not match finalBatch"
            )
        if value.get("embeddingDimensions") != 1024:
            raise RuntimeError(
                "Embedding dimension mismatch: "
                f"expected 1024, got {value.get('embeddingDimensions')!r}"
            )
        return stage

    def restart_service(self, project_path: Path) -> None:
        command = self.config.service_restart_command
        if not command:
            raise RuntimeError("No service_restart_command is configured")
        try:
            subprocess.run(
                command,
                check=True,
                timeout=self.config.startup_timeout_seconds,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise RuntimeError(f"Failed to restart LLM Wiki service: {exc}") from exc
        self.wait_until_ready(project_path)

    def answer(self, run_id: str, *, prompt: str, session_id: str) -> QaResult:
        value = self._request(
            "POST",
            f"/runs/{run_id}/qa",
            {
                "message": prompt,
                "sessionId": session_id,
                "mode": "standard",
                "retrievalMode": "standard",
                "tools": {"wiki": True, "web": False, "anytxt": False},
                "history": [],
                "historyExplicit": True,
                "skills": [],
                "skillMode": "explicit",
                "persistSession": False,
            },
        )
        return QaResult.from_wire(value)

    def delete(self, run_id: str) -> StageResult:
        value = self._request("POST", f"/runs/{run_id}/delete", {"wait": True})
        stage = StageResult.from_wire(value, context="deletion")
        if stage.usage.total != 0:
            raise RuntimeError("Deletion unexpectedly consumed model or embedding tokens")
        return stage

    def _request(self, method: str, path: str, body: dict[str, Any]) -> dict[str, Any]:
        response = requests.request(
            method,
            f"{self.base_url}{path}",
            headers=self.headers,
            json=body,
            timeout=self.config.request_timeout_seconds,
        )
        response.raise_for_status()
        value = response.json()
        if not isinstance(value, dict):
            raise RuntimeError(f"Bridge returned non-object JSON for {path}")
        return value
