"""Configuration for the TencentDB MemoryKnowledge baseline."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class TencentDBConfig:
    base_url: str
    service_id: str
    team_id: str
    user_id: str
    llm_model: str
    llm_base_url: str
    llm_api_key_env: str
    output_dir: Path
    request_timeout_seconds: int = 1200
    poll_interval_seconds: float = 2.0
    ingest_timeout_seconds: int = 86400
    qa_workers: int = 1
    judge_workers: int = 1
    max_loop_turns: int = 15
    raw_max_bytes: int = 512 * 1024

    @classmethod
    def from_yaml(cls, path: Path) -> "TencentDBConfig":
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError(f"TencentDB config must be a mapping: {path}")
        api = _mapping(raw, "memoryknowledge")
        llm = _mapping(raw, "llm")
        execution = _mapping(raw, "execution")
        paths = _mapping(raw, "paths")
        cfg = cls(
            base_url=_text(api, "base_url").rstrip("/"),
            service_id=_text(api, "service_id"),
            team_id=_text(api, "team_id"),
            user_id=_text(api, "user_id"),
            llm_model=_text(llm, "model"),
            llm_base_url=_text(llm, "base_url").rstrip("/"),
            llm_api_key_env=_text(llm, "api_key_env"),
            output_dir=Path(_text(paths, "output_dir")).expanduser().resolve(),
            request_timeout_seconds=_positive_int(execution, "request_timeout_seconds", 1200),
            poll_interval_seconds=float(execution.get("poll_interval_seconds", 2.0)),
            ingest_timeout_seconds=_positive_int(execution, "ingest_timeout_seconds", 86400),
            qa_workers=_positive_int(execution, "qa_workers", 1),
            judge_workers=_positive_int(execution, "judge_workers", 1),
            max_loop_turns=_positive_int(execution, "max_loop_turns", 15),
        )
        if cfg.max_loop_turns != 15:
            raise ValueError("TencentDB baseline fixes max_loop_turns to 15")
        if cfg.poll_interval_seconds <= 0:
            raise ValueError("poll_interval_seconds must be positive")
        return cfg

    def api_key(self) -> str:
        value = os.environ.get(self.llm_api_key_env, "").strip()
        if not value:
            raise RuntimeError(f"LLM API key environment variable is not set: {self.llm_api_key_env}")
        return value

    def manifest(self, *, tencentdb_commit: str | None = None) -> dict[str, Any]:
        return {
            "backend": "TencentDB-Agent-Memory",
            "knowledge_backend": "MemoryKnowledge/LLM-Wiki",
            "memoryknowledge_branch": "feat/server_team",
            "memoryknowledge_commit": tencentdb_commit,
            "api_base_url": self.base_url,
            "service_id": self.service_id,
            "team_id": self.team_id,
            "llm": {
                "model": self.llm_model,
                "base_url": self.llm_base_url,
                "temperature": 0,
                "provider": "volcengine",
                "thinking": "disabled",
            },
            "embedding": {"mode": "not_configured_by_baseline", "reason": "MemoryKnowledge default FTS5 retrieval"},
            "pdf": {
                "pipeline": "external_pdf_to_markdown_plus_memoryknowledge_ingest",
                "parser": "OpenViking PDFParser local-compatible pdfplumber",
                "openviking_commit": "447b30ef8511dcc82c07ede857a52150479ee77c",
                "images": "omitted_with_placeholders_not_uploaded",
                "conversion_time_in_ingest_time": True,
            },
            "raw_max_bytes": self.raw_max_bytes,
            "qa_workers": self.qa_workers,
            "judge_workers": self.judge_workers,
            "max_loop_turns": self.max_loop_turns,
            "retrieval": "TencentDB default parameters",
        }


def _mapping(parent: dict[str, Any], key: str) -> dict[str, Any]:
    value = parent.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Config field {key!r} must be a mapping")
    return value


def _text(parent: dict[str, Any], key: str) -> str:
    value = parent.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Config field {key!r} must be a non-empty string")
    return value.strip()


def _positive_int(parent: dict[str, Any], key: str, default: int) -> int:
    value = parent.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"Config field {key!r} must be a positive integer")
    return value
