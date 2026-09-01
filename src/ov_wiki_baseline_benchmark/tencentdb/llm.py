"""OpenAI-compatible LLM calls and the fixed 15-turn Wiki tool loop."""

from __future__ import annotations

import json
import time
import uuid
from typing import Any, Callable

import requests

from .config import TencentDBConfig
from .ledger import CallLedger


class LlmCallError(RuntimeError):
    pass


class OpenAICompatibleClient:
    def __init__(self, config: TencentDBConfig, ledger: CallLedger):
        self.config = config
        self.ledger = ledger
        self.url = f"{config.llm_base_url.rstrip('/')}/chat/completions"
        self.headers = {"Authorization": f"Bearer {config.api_key()}", "Content-Type": "application/json"}

    def complete(
        self,
        *,
        messages: list[dict[str, Any]],
        phase: str,
        qa_id: str | None = None,
        experiment_id: str | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        started = time.time()
        body: dict[str, Any] = {
            "model": self.config.llm_model,
            "messages": messages,
            "temperature": 0,
            "thinking": {"type": "disabled"},
            "stream": False,
        }
        if tools:
            body["tools"] = tools
        try:
            response = requests.post(self.url, headers=self.headers, json=body, timeout=self.config.request_timeout_seconds)
            payload = response.json()
            if not response.ok:
                raise LlmCallError(f"HTTP {response.status_code}: {str(payload)[:500]}")
            if not isinstance(payload, dict) or not _first_message(payload):
                raise LlmCallError("LLM response is missing choices[0].message")
            usage = payload.get("usage") if isinstance(payload, dict) else None
            complete = _valid_usage(usage)
            self.ledger.append(phase=phase, operation="llm.chat.completions", status="success", started_at=started,
                               input_tokens=usage.get("prompt_tokens", 0) if complete else 0,
                               output_tokens=usage.get("completion_tokens", 0) if complete else 0,
                               token_usage_complete=complete, qa_id=qa_id,
                               experiment_id=experiment_id,
                               metadata={"usage_missing_or_invalid": not complete})
            return payload if isinstance(payload, dict) else {}
        except Exception as exc:
            self.ledger.append(phase=phase, operation="llm.chat.completions", status="failed", started_at=started,
                               token_usage_complete=False, qa_id=qa_id,
                               experiment_id=experiment_id, error=str(exc))
            raise

    def answer_with_tools(
        self,
        *,
        wiki_id: str,
        question: str,
        answer_prompt: str,
        tools: list[dict[str, Any]],
        call_tool: Callable[[str, dict[str, Any]], dict[str, Any]],
        qa_id: str,
        experiment_id: str | None = None,
    ) -> tuple[str, float, int]:
        started = time.perf_counter()
        messages: list[dict[str, Any]] = []
        if answer_prompt.strip():
            messages.append({"role": "system", "content": answer_prompt.strip()})
        messages.append({"role": "user", "content": question})
        turns = 0
        for turns in range(1, self.config.max_loop_turns + 1):
            try:
                payload = self.complete(
                    messages=messages,
                    phase="qa",
                    qa_id=qa_id,
                    experiment_id=experiment_id,
                    tools=tools,
                )
            except Exception as exc:
                return "", time.perf_counter() - started, turns
            message = _first_message(payload)
            if not message:
                return "", time.perf_counter() - started, turns
            tool_calls = message.get("tool_calls") or []
            content = message.get("content") if isinstance(message.get("content"), str) else ""
            if not tool_calls:
                return content.strip(), time.perf_counter() - started, turns
            messages.append({"role": "assistant", "content": content or None, "tool_calls": tool_calls})
            for tool_call in tool_calls:
                call_id = str(tool_call.get("id") or uuid.uuid4().hex)
                function = tool_call.get("function") or {}
                name = str(function.get("name") or "")
                try:
                    arguments = json.loads(function.get("arguments") or "{}")
                    if not isinstance(arguments, dict):
                        arguments = {}
                    observation = call_tool(name, arguments)
                    result_text = json.dumps(observation, ensure_ascii=False)
                except Exception as exc:
                    result_text = json.dumps({"error": str(exc)}, ensure_ascii=False)
                messages.append({"role": "tool", "tool_call_id": call_id, "name": name, "content": result_text})
        return "", time.perf_counter() - started, turns


def _first_message(payload: dict[str, Any]) -> dict[str, Any] | None:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        return None
    message = choices[0].get("message")
    return message if isinstance(message, dict) else None


def _valid_usage(value: Any) -> bool:
    return isinstance(value, dict) and all(isinstance(value.get(k), int) and not isinstance(value.get(k), bool) and value.get(k) >= 0 for k in ("prompt_tokens", "completion_tokens"))
