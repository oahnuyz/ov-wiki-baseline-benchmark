"""Append-only call ledger; metrics are always recomputed from this file."""

from __future__ import annotations

import json
import threading
import time
import uuid
from pathlib import Path
from typing import Any


class CallLedger:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def append(
        self,
        *,
        phase: str,
        operation: str,
        status: str,
        started_at: float,
        ended_at: float | None = None,
        input_tokens: int = 0,
        output_tokens: int = 0,
        token_usage_complete: bool = False,
        qa_id: str | None = None,
        experiment_id: str | None = None,
        error: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        # Invalid/missing provider usage is deliberately normalized to zero.
        if not isinstance(input_tokens, int) or input_tokens < 0:
            input_tokens = 0
        if not isinstance(output_tokens, int) or output_tokens < 0:
            output_tokens = 0
        if not token_usage_complete:
            input_tokens = output_tokens = 0
        record = {
            "id": uuid.uuid4().hex,
            "phase": phase,
            "operation": operation,
            "status": status,
            "started_at": started_at,
            "ended_at": ended_at if ended_at is not None else time.time(),
            "duration_seconds": max(0.0, (ended_at if ended_at is not None else time.time()) - started_at),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "token_usage_complete": bool(token_usage_complete),
            "qa_id": qa_id,
            "experiment_id": experiment_id,
            "error": error,
            "metadata": metadata or {},
        }
        with self._lock:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                import os
                os.fsync(handle.fileno())
        return record

    def records(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        out = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            try:
                value = json.loads(line)
                if isinstance(value, dict):
                    out.append(value)
            except json.JSONDecodeError:
                continue
        return out

    def aggregate(self, *, phase: str | None = None, experiment_id: str | None = None) -> dict[str, Any]:
        rows = [
            r for r in self.records()
            if (phase is None or r.get("phase") == phase)
            and (experiment_id is None or r.get("experiment_id") == experiment_id)
        ]
        return {
            "calls": len(rows),
            "successful_calls": sum(r.get("status") == "success" for r in rows),
            "failed_calls": sum(r.get("status") != "success" for r in rows),
            "input_tokens": sum(int(r.get("input_tokens", 0) or 0) for r in rows),
            "output_tokens": sum(int(r.get("output_tokens", 0) or 0) for r in rows),
            "total_tokens": sum(int(r.get("total_tokens", 0) or 0) for r in rows),
            "incomplete_usage_calls": sum(
                not r.get("token_usage_complete", False)
                for r in rows
                if r.get("metadata", {}).get("token_bearing", True)
            ),
        }
