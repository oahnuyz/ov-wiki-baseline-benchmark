"""Configuration contract for the Nashsu LLM Wiki baseline."""

from __future__ import annotations

import os
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class BenchmarkConfig:
    bridge_base_url: str
    bridge_token_env: str
    ark_api_key_env: str
    model: str
    model_provider: str
    model_base_url: str
    embedding_model: str
    embedding_provider: str
    embedding_base_url: str
    embedding_dimensions: int
    embedding_input: str
    output_dir: Path
    project_path: Path
    startup_timeout_seconds: int
    request_timeout_seconds: int
    qa_timeout_seconds: int = 600
    qa_max_agent_iterations: int = 20
    qa_max_retrieval_actions: int = 15
    max_qa_retries: int = 2
    ingest_batch_size: int = 25
    max_batch_retries: int = 2
    restart_between_ingest_batches: bool = True
    service_restart_command: tuple[str, ...] = ()
    service_stop_command: tuple[str, ...] = ()
    snapshot_root: Path = Path("/tmp/ov-wiki-benchmark-snapshots")

    @classmethod
    def from_yaml(cls, path: Path) -> "BenchmarkConfig":
        with path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
        if not isinstance(raw, dict):
            raise ValueError(f"Benchmark config must be a mapping: {path}")
        bridge = _mapping(raw, "bridge")
        model = _mapping(raw, "model")
        embedding = _mapping(raw, "embedding")
        paths = _mapping(raw, "paths")
        execution = _mapping(raw, "execution")
        _expect(model, "temperature", 0)
        _expect(model, "thinking", "disabled")
        _expect(execution, "mode", "standard")
        _expect(execution, "retrieval_mode", "standard")
        _expect(execution, "top_k", "official_default")
        _expect(execution, "max_context_size", "official_default")
        _expect(execution, "project_template", "general")
        _expect(execution, "output_language", "English")
        _expect(execution, "chunking", "official_default")
        _expect(execution, "persist_extracted_markdown", False)
        _expect(execution, "vector_retrieval", True)
        _expect(execution, "pdf_parser", "builtin")
        _expect(execution, "image_captioning", True)
        _expect(execution, "review_sweep_in_ingest", True)
        _expect(execution, "qa_workers", 1)
        _expect(execution, "judge_workers", 1)
        cfg = cls(
            bridge_base_url=_text(bridge, "base_url").rstrip("/"),
            bridge_token_env=_text(bridge, "token_env"),
            ark_api_key_env=_text(model, "api_key_env"),
            model=_text(model, "name"),
            model_provider=_text(model, "provider"),
            model_base_url=_text(model, "base_url").rstrip("/"),
            embedding_model=_text(embedding, "name"),
            embedding_provider=_text(embedding, "provider"),
            embedding_base_url=_text(embedding, "base_url").rstrip("/"),
            embedding_dimensions=_integer(embedding, "dimensions"),
            embedding_input=_text(embedding, "input"),
            output_dir=Path(_text(paths, "output_dir")).expanduser().resolve(),
            project_path=Path(_text(paths, "project_path")).expanduser().resolve(),
            startup_timeout_seconds=_integer(execution, "startup_timeout_seconds"),
            request_timeout_seconds=_integer(execution, "request_timeout_seconds"),
            qa_timeout_seconds=_optional_integer(execution, "qa_timeout_seconds", 600),
            qa_max_agent_iterations=_optional_integer(
                execution, "qa_max_agent_iterations", 20
            ),
            qa_max_retrieval_actions=_optional_integer(
                execution, "qa_max_retrieval_actions", 15
            ),
            max_qa_retries=_optional_integer(execution, "max_qa_retries", 2),
            ingest_batch_size=_optional_integer(execution, "ingest_batch_size", 25),
            max_batch_retries=_optional_integer(execution, "max_batch_retries", 2),
            restart_between_ingest_batches=_optional_boolean(
                execution, "restart_between_ingest_batches", True
            ),
            service_restart_command=_command(execution.get("service_restart_command")),
            service_stop_command=_command(execution.get("service_stop_command")),
            snapshot_root=Path(_text(paths, "snapshot_root")).expanduser().resolve(),
        )
        cfg.validate()
        return cfg

    def validate(self) -> None:
        expected = {
            "model": (self.model, "doubao-seed-2-0-lite-260428"),
            "model_provider": (self.model_provider, "volcengine"),
            "model_base_url": (
                self.model_base_url,
                "https://ark.cn-beijing.volces.com/api/v3",
            ),
            "embedding_model": (
                self.embedding_model,
                "doubao-embedding-vision-251215",
            ),
            "embedding_provider": (self.embedding_provider, "volcengine"),
            "embedding_base_url": (
                self.embedding_base_url,
                "https://ark.cn-beijing.volces.com/api/v3",
            ),
            "embedding_dimensions": (self.embedding_dimensions, 1024),
            "embedding_input": (self.embedding_input, "multimodal"),
        }
        mismatches = [
            f"{name}={actual!r}, expected {wanted!r}"
            for name, (actual, wanted) in expected.items()
            if actual != wanted
        ]
        if mismatches:
            raise ValueError("Invalid fixed benchmark config: " + "; ".join(mismatches))
        if self.startup_timeout_seconds <= 0 or self.request_timeout_seconds <= 0:
            raise ValueError("startup and request timeouts must be positive")
        if self.qa_timeout_seconds <= 0:
            raise ValueError("qa_timeout_seconds must be positive")
        if self.qa_max_agent_iterations != 20:
            raise ValueError("qa_max_agent_iterations must be 20 for this benchmark")
        if self.qa_max_retrieval_actions != 15:
            raise ValueError("qa_max_retrieval_actions must be 15 for this benchmark")
        if self.max_qa_retries < 0:
            raise ValueError("max_qa_retries must be non-negative")
        if self.ingest_batch_size <= 0:
            raise ValueError("ingest_batch_size must be positive")
        if self.max_batch_retries < 0:
            raise ValueError("max_batch_retries must be non-negative")
        if self.restart_between_ingest_batches and not self.service_restart_command:
            raise ValueError(
                "service_restart_command is required when "
                "restart_between_ingest_batches=true"
            )
        if not self.service_stop_command:
            raise ValueError("service_stop_command is required for snapshot restoration")
        if self.snapshot_root == self.project_path or self.snapshot_root in self.project_path.parents:
            raise ValueError("snapshot_root must be outside project_path")

    def bridge_token(self) -> str:
        return _required_env(self.bridge_token_env)

    def ark_api_key(self) -> str:
        return _required_env(self.ark_api_key_env)

    def public_manifest(self) -> dict[str, Any]:
        return {
            "llmWiki": {
                "version": "0.6.11",
                "commit": "e8082119649e6a8e1cf85eaf289adcabfdf39d4e",
                "mode": "standard",
                "retrievalMode": "standard",
                "topK": {"source": "official_default", "resolvedValue": 5},
                "maxContextSize": {
                    "source": "official_default",
                    "resolvedValue": 204800,
                    "unit": "characters",
                },
                "projectTemplate": "general",
                "outputLanguage": "English",
                "chunking": "official_default",
                "persistExtractedMarkdown": False,
                "vectorRetrieval": True,
                "imageCaptioning": True,
                "pdfParser": "builtin",
                "reviewSweepIncludedInIngest": True,
                "ingestConcurrency": 1,
                "ingestBatchSize": self.ingest_batch_size,
                "maxBatchRetries": self.max_batch_retries,
                "restartBetweenIngestBatches": self.restart_between_ingest_batches,
                "qaTimeoutSeconds": self.qa_timeout_seconds,
                "qaMaxAgentIterations": self.qa_max_agent_iterations,
                "qaMaxRetrievalActions": self.qa_max_retrieval_actions,
                "benchmarkRawSourceSearch": True,
                "maxQaRetries": self.max_qa_retries,
            },
            "model": {
                "name": self.model,
                "provider": self.model_provider,
                "baseUrl": self.model_base_url,
                "temperature": 0,
                "thinking": "disabled",
            },
            "embedding": {
                "name": self.embedding_model,
                "provider": self.embedding_provider,
                "baseUrl": self.embedding_base_url,
                "dimensions": self.embedding_dimensions,
                "input": self.embedding_input,
            },
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


def _integer(parent: dict[str, Any], key: str) -> int:
    value = parent.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"Config field {key!r} must be an integer")
    return value


def _optional_integer(parent: dict[str, Any], key: str, default: int) -> int:
    value = parent.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"Benchmark config field {key!r} must be an integer")
    return value


def _optional_boolean(parent: dict[str, Any], key: str, default: bool) -> bool:
    value = parent.get(key, default)
    if not isinstance(value, bool):
        raise ValueError(f"Benchmark config field {key!r} must be a boolean")
    return value


def _command(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        parts = shlex.split(value)
    elif isinstance(value, list) and all(isinstance(item, str) for item in value):
        parts = [item for item in value if item]
    else:
        raise ValueError(
            "Benchmark config field 'service_restart_command' must be a string "
            "or a list of strings"
        )
    if not parts:
        raise ValueError("service_restart_command must not be empty")
    return tuple(parts)


def _required_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"Required environment variable is not set: {name}")
    return value


def _expect(parent: dict[str, Any], key: str, expected: Any) -> None:
    actual = parent.get(key)
    if type(actual) is not type(expected) or actual != expected:
        raise ValueError(
            f"Fixed benchmark field {key!r} must be {expected!r}, got {actual!r}"
        )
