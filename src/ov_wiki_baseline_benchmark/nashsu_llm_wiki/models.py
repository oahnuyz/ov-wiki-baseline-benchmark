"""Strict wire and metric models for the LLM Wiki benchmark bridge."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


class TelemetryError(RuntimeError):
    """Raised when provider-reported token telemetry is absent or malformed."""


@dataclass(frozen=True)
class RunInfo:
    run_id: str
    project_scaffold: Mapping[str, Any]

    @classmethod
    def from_wire(cls, value: Any) -> "RunInfo":
        if not isinstance(value, Mapping):
            raise RuntimeError("Invalid bridge create-run response")
        run_id = value.get("runId")
        if not isinstance(run_id, str) or not run_id:
            raise RuntimeError("Bridge create-run response is missing runId")
        if value.get("resolvedMaxContextSize") != 204800:
            raise RuntimeError(
                "Expected official default maxContextSize=204800 characters, bridge "
                f"reported {value.get('resolvedMaxContextSize')!r}"
            )
        scaffold = value.get("projectScaffold")
        if not isinstance(scaffold, Mapping):
            raise RuntimeError("Bridge did not report the project scaffold")
        expected = {
            "template": "general",
            "outputLanguage": "English",
            "chunking": "official_default",
            "persistExtractedMarkdown": False,
        }
        for field, wanted in expected.items():
            if type(scaffold.get(field)) is not type(wanted) or scaffold.get(field) != wanted:
                raise RuntimeError(
                    f"Invalid project scaffold field {field!r}: {scaffold.get(field)!r}"
                )
        hashes = scaffold.get("fileSha256")
        required_files = {
            "purpose.md",
            "schema.md",
            "wiki/index.md",
            "wiki/overview.md",
            "wiki/log.md",
        }
        if not isinstance(hashes, Mapping) or not required_files.issubset(hashes):
            raise RuntimeError("Bridge project scaffold is missing file hashes")
        for path in required_files:
            digest = hashes[path]
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise RuntimeError(f"Invalid scaffold SHA-256 for {path}")
        return cls(run_id=run_id, project_scaffold=dict(scaffold))


def _nonnegative_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TelemetryError(f"Telemetry field {field!r} must be a non-negative integer")
    return value


@dataclass(frozen=True)
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    embedding_tokens: int = 0

    @classmethod
    def from_wire(cls, value: Any, *, context: str) -> "TokenUsage":
        if not isinstance(value, Mapping):
            raise TelemetryError(f"{context} token usage is missing")
        required = {"inputTokens", "outputTokens", "embeddingTokens"}
        missing = sorted(required - set(value))
        if missing:
            raise TelemetryError(
                f"{context} token usage is incomplete; missing fields: {missing}"
            )
        return cls(
            input_tokens=_nonnegative_int(value["inputTokens"], "inputTokens"),
            output_tokens=_nonnegative_int(value["outputTokens"], "outputTokens"),
            embedding_tokens=_nonnegative_int(
                value["embeddingTokens"], "embeddingTokens"
            ),
        )

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            embedding_tokens=self.embedding_tokens + other.embedding_tokens,
        )

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens + self.embedding_tokens

    def as_dict(self) -> dict[str, int]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "embedding_tokens": self.embedding_tokens,
            "total_tokens": self.total,
        }


@dataclass(frozen=True)
class QaTokenUsage:
    agent_input_tokens: int
    agent_output_tokens: int
    search_input_tokens: int
    search_output_tokens: int
    embedding_tokens: int

    @classmethod
    def from_wire(cls, value: Any) -> "QaTokenUsage":
        if not isinstance(value, Mapping):
            raise TelemetryError("QA token usage is missing")
        required = {
            "inputTokens",
            "outputTokens",
            "embeddingTokens",
            "agentInputTokens",
            "agentOutputTokens",
            "searchInputTokens",
            "searchOutputTokens",
        }
        missing = sorted(required - set(value))
        if missing:
            raise TelemetryError(
                f"QA token usage is incomplete; missing fields: {missing}"
            )
        usage = cls(
            agent_input_tokens=_nonnegative_int(
                value["agentInputTokens"], "agentInputTokens"
            ),
            agent_output_tokens=_nonnegative_int(
                value["agentOutputTokens"], "agentOutputTokens"
            ),
            search_input_tokens=_nonnegative_int(
                value["searchInputTokens"], "searchInputTokens"
            ),
            search_output_tokens=_nonnegative_int(
                value["searchOutputTokens"], "searchOutputTokens"
            ),
            embedding_tokens=_nonnegative_int(
                value["embeddingTokens"], "embeddingTokens"
            ),
        )
        declared_input = _nonnegative_int(value["inputTokens"], "inputTokens")
        declared_output = _nonnegative_int(value["outputTokens"], "outputTokens")
        if declared_input != usage.input_tokens or declared_output != usage.output_tokens:
            raise TelemetryError(
                "QA token totals do not equal Agent plus search token components"
            )
        return usage

    @property
    def input_tokens(self) -> int:
        return self.agent_input_tokens + self.search_input_tokens

    @property
    def output_tokens(self) -> int:
        return self.agent_output_tokens + self.search_output_tokens

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens + self.embedding_tokens


@dataclass(frozen=True)
class StageResult:
    duration_seconds: float
    usage: TokenUsage
    payload: Mapping[str, Any]

    @classmethod
    def from_wire(cls, value: Any, *, context: str) -> "StageResult":
        if not isinstance(value, Mapping):
            raise RuntimeError(f"Invalid {context} response")
        if value.get("status") != "completed":
            raise RuntimeError(
                f"{context} did not complete: {value.get('error') or value.get('status')}"
            )
        duration = value.get("durationSeconds")
        if isinstance(duration, bool) or not isinstance(duration, (int, float)) or duration < 0:
            raise RuntimeError(f"Invalid {context} durationSeconds")
        return cls(
            duration_seconds=float(duration),
            usage=TokenUsage.from_wire(value.get("usage"), context=context),
            payload=value,
        )


@dataclass(frozen=True)
class QaResult:
    answer: str
    session_id: str
    duration_seconds: float
    usage: QaTokenUsage
    payload: Mapping[str, Any]

    @classmethod
    def from_wire(cls, value: Any) -> "QaResult":
        if not isinstance(value, Mapping):
            raise RuntimeError("Invalid QA response")
        if value.get("status") != "completed":
            raise RuntimeError(
                f"QA did not complete: {value.get('error') or value.get('status')}"
            )
        duration = value.get("durationSeconds")
        if isinstance(duration, bool) or not isinstance(duration, (int, float)) or duration < 0:
            raise RuntimeError("Invalid QA durationSeconds")
        answer = value.get("answer")
        session_id = value.get("sessionId")
        if not isinstance(answer, str) or not answer.strip():
            raise RuntimeError("LLM Wiki returned an empty QA answer")
        if not isinstance(session_id, str) or not session_id.strip():
            raise RuntimeError("LLM Wiki QA response is missing sessionId")
        resolved_top_k = value.get("resolvedTopK")
        if resolved_top_k != 5:
            raise RuntimeError(
                f"Expected official default top_k=5, bridge reported {resolved_top_k!r}"
            )
        return cls(
            answer=answer,
            session_id=session_id,
            duration_seconds=float(duration),
            usage=QaTokenUsage.from_wire(value.get("usage")),
            payload=value,
        )
