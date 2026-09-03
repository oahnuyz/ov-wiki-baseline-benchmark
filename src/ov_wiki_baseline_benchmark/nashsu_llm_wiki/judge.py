"""Generic 0-4 judge with the reference baseline's exact fallback behavior."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

from .models import TokenUsage


@dataclass(frozen=True)
class JudgeResult:
    score: int
    reasoning: str
    prompt_type: str
    usage: TokenUsage | None
    raw_content: str


class ArkJudge:
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        prompt_path: Path,
        timeout_seconds: int,
    ) -> None:
        self.api_key = api_key
        self.url = f"{base_url.rstrip('/')}/chat/completions"
        self.model = model
        self.prompt_template = prompt_path.read_text(encoding="utf-8")
        self.timeout_seconds = timeout_seconds

    def grade(self, question: str, gold_answers: list[str], answer: str) -> JudgeResult:
        # Match only the three approved placeholders so the literal JSON example
        # remains unchanged. re.sub does not reinterpret braces in inserted values.
        values = {
            "question": question,
            "gold_answers_joined_by_pipe": " | ".join(gold_answers),
            "generated_answer": answer,
        }
        prompt = re.sub(
            r"\{(question|gold_answers_joined_by_pipe|generated_answer)\}",
            lambda match: values[match.group(1)],
            self.prompt_template,
        )
        content = ""
        score = 0
        reasoning = "No reasoning provided."
        usage: TokenUsage | None = None
        try:
            response = requests.post(
                self.url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0,
                    "thinking": {"type": "disabled"},
                    "stream": False,
                },
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
            content = _response_content(payload)
            result = json.loads(content)
            score = max(0, min(4, int(result.get("score", 0))))
            reasoning = result.get("reasoning", "No reasoning provided.")
            usage = _optional_usage(payload.get("usage"))
        except Exception:
            text = (content or "").strip()
            reasoning = (
                f"Parse fallback from raw output: {text}"
                if text
                else "Parse failed or model invocation failed. Defaulted to 0."
            )
            match = re.search(r'"score"\s*:\s*([0-4])', text)
            if not match:
                match = re.search(r"\b([0-4])\b", text)
            score = int(match.group(1)) if match else 0
            score = max(0, min(4, score))
        return JudgeResult(
            score=score,
            reasoning=str(reasoning),
            prompt_type="Generic_0-4",
            usage=usage,
            raw_content=content,
        )


def _response_content(payload: Any) -> str:
    if not isinstance(payload, dict):
        return ""
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        return ""
    message = choices[0].get("message")
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    return content if isinstance(content, str) else ""


def _optional_usage(value: Any) -> TokenUsage | None:
    if not isinstance(value, dict):
        return None
    prompt = value.get("prompt_tokens")
    completion = value.get("completion_tokens")
    if (
        isinstance(prompt, int)
        and not isinstance(prompt, bool)
        and prompt >= 0
        and isinstance(completion, int)
        and not isinstance(completion, bool)
        and completion >= 0
    ):
        return TokenUsage(input_tokens=prompt, output_tokens=completion)
    return None
