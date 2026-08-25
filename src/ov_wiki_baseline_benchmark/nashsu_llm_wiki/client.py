"""HTTP client for the benchmark-only LLM Wiki bridge endpoints."""

from __future__ import annotations

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

    def create_run(self, *, corpus_id: str, project_path: Path) -> RunInfo:
        value = self._request(
            "POST",
            "/runs",
            {
                "corpusId": corpus_id,
                "projectPath": str(project_path),
                "config": self.config.public_manifest(),
            },
        )
        return RunInfo.from_wire(value)

    def ingest(self, run_id: str, documents: list[Path]) -> StageResult:
        value = self._request(
            "POST",
            f"/runs/{run_id}/ingest",
            {"documents": [str(path) for path in documents], "wait": True},
        )
        stage = StageResult.from_wire(value, context="ingestion")
        if value.get("reviewSweepCompleted") is not True:
            raise RuntimeError("Ingestion response does not include a completed review sweep")
        if value.get("embeddingDimensions") != 1024:
            raise RuntimeError(
                "Embedding dimension mismatch: "
                f"expected 1024, got {value.get('embeddingDimensions')!r}"
            )
        return stage

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
