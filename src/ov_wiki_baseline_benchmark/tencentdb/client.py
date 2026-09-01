"""Small, defensive HTTP client for MemoryKnowledge v3."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import requests

from .config import TencentDBConfig


class TencentDBRequestError(RuntimeError):
    def __init__(self, message: str, *, status_code: int | None = None):
        self.status_code = status_code
        super().__init__(message)


class MemoryKnowledgeClient:
    def __init__(self, config: TencentDBConfig):
        self.config = config
        self.base_url = config.base_url.rstrip("/")
        self.headers = {"Content-Type": "application/json", "x-tdai-service-id": config.service_id}

    def create_wiki(self, name: str) -> dict[str, Any]:
        return self._post("/wiki/create", {"team_id": self.config.team_id, "name": name, "user_id": self.config.user_id})

    def get_wiki(self, wiki_id: str) -> dict[str, Any]:
        return self._post("/wiki/get", {"wiki_id": wiki_id})

    def raw_write(self, wiki_id: str, files: list[dict[str, str]]) -> dict[str, Any]:
        return self._post("/wiki/raw/write", {"team_id": self.config.team_id, "wiki_id": wiki_id, "user_id": self.config.user_id, "files": files})

    def ingest(self, wiki_id: str) -> dict[str, Any]:
        return self._post("/wiki/ingest", {"wiki_id": wiki_id, "user_id": self.config.user_id})

    def wait_ready(self, wiki_id: str) -> dict[str, Any]:
        deadline = time.monotonic() + self.config.ingest_timeout_seconds
        while time.monotonic() < deadline:
            detail = self.get_wiki(wiki_id)
            status = str(detail.get("status", ""))
            if status in {"ready", "failed"}:
                return detail
            time.sleep(self.config.poll_interval_seconds)
        raise TencentDBRequestError(f"Wiki ingest timed out after {self.config.ingest_timeout_seconds}s")

    def tools_list(self, wiki_id: str) -> dict[str, Any]:
        return self._post("/tools/list", {"knowledge_id": wiki_id})

    def tools_call(self, wiki_id: str, tool_name: str, params: dict[str, Any]) -> dict[str, Any]:
        return self._post("/tools/call", {"knowledge_id": wiki_id, "tool_name": tool_name, "params": params})

    def delete_wiki(self, wiki_id: str) -> dict[str, Any]:
        return self._post("/wiki/delete", {"wiki_ids": [wiki_id]})

    def _post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        try:
            response = requests.post(f"{self.base_url}{path}", headers=self.headers, json=body, timeout=self.config.request_timeout_seconds)
        except requests.RequestException as exc:
            raise TencentDBRequestError(str(exc)) from exc
        try:
            payload = response.json()
        except ValueError as exc:
            raise TencentDBRequestError(f"non-JSON response from {path}: {response.text[:300]}", status_code=response.status_code) from exc
        if not response.ok or not isinstance(payload, dict) or payload.get("code") not in (0, None):
            message = payload.get("message") if isinstance(payload, dict) else response.text
            raise TencentDBRequestError(str(message), status_code=response.status_code)
        data = payload.get("data", payload)
        return data if isinstance(data, dict) else {"value": data}
