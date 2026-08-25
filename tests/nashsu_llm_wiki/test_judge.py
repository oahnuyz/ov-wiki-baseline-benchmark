from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from ov_wiki_baseline_benchmark.nashsu_llm_wiki.judge import ArkJudge


class JudgeTests(unittest.TestCase):
    def _judge(self, root: Path) -> ArkJudge:
        prompt = root / "judge.txt"
        prompt.write_text(
            "Question: {question}\nGold Answers: {gold_answers_joined_by_pipe}\n"
            "Generated Answer: {generated_answer}\n",
            encoding="utf-8",
        )
        return ArkJudge(
            api_key="not-a-real-key",
            base_url="https://example.invalid/v3",
            model="model",
            prompt_path=prompt,
            timeout_seconds=1,
        )

    def test_json_like_regex_fallback_matches_reference(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            response = Mock()
            response.raise_for_status.return_value = None
            response.json.return_value = {
                "choices": [{"message": {"content": 'prefix "score": 3 suffix'}}]
            }
            with patch(
                "ov_wiki_baseline_benchmark.nashsu_llm_wiki.judge.requests.post",
                return_value=response,
            ):
                result = self._judge(Path(name)).grade("q", ["g"], "a")
        self.assertEqual(result.score, 3)
        self.assertIn("Parse fallback", result.reasoning)

    def test_invocation_failure_defaults_to_zero(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            with patch(
                "ov_wiki_baseline_benchmark.nashsu_llm_wiki.judge.requests.post",
                side_effect=RuntimeError("network"),
            ):
                result = self._judge(Path(name)).grade("q", ["g"], "a")
        self.assertEqual(result.score, 0)
        self.assertEqual(
            result.reasoning,
            "Parse failed or model invocation failed. Defaulted to 0.",
        )


if __name__ == "__main__":
    unittest.main()
