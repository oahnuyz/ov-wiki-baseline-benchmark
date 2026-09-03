from __future__ import annotations

import unittest

from ov_wiki_baseline_benchmark.nashsu_llm_wiki.metrics import (
    is_refusal,
    max_token_f1,
    normalize_answer,
    token_f1,
)
from ov_wiki_baseline_benchmark.nashsu_llm_wiki.models import (
    QaTokenUsage,
    RunInfo,
    TelemetryError,
    TokenUsage,
)


class MetricsAndTelemetryTests(unittest.TestCase):
    def test_create_run_requires_reproducible_general_scaffold_hashes(self) -> None:
        scaffold = {
            "template": "general",
            "outputLanguage": "English",
            "chunking": "official_default",
            "persistExtractedMarkdown": False,
            "fileSha256": {
                path: "a" * 64
                for path in (
                    "purpose.md",
                    "schema.md",
                    "wiki/index.md",
                    "wiki/overview.md",
                    "wiki/log.md",
                )
            },
        }
        info = RunInfo.from_wire(
            {
                "runId": "run-1",
                "resolvedMaxContextSize": 204800,
                "projectScaffold": scaffold,
            }
        )
        self.assertEqual(info.project_scaffold["fileSha256"]["purpose.md"], "a" * 64)

        scaffold["fileSha256"]["purpose.md"] = "not-a-hash"
        with self.assertRaises(RuntimeError):
            RunInfo.from_wire(
                {
                    "runId": "run-1",
                    "resolvedMaxContextSize": 204800,
                    "projectScaffold": scaffold,
                }
            )

    def test_reference_normalization_and_f1(self) -> None:
        self.assertEqual(normalize_answer("The King, and a Queen."), "king queen")
        self.assertAlmostEqual(token_f1("King Padella", "Padella"), 2 / 3)
        self.assertEqual(max_token_f1("second", ["first", "second"]), 1.0)

    def test_reference_refusal_phrases(self) -> None:
        self.assertTrue(is_refusal("There is no information in the database."))
        self.assertFalse(is_refusal("The answer is Padella."))

    def test_token_usage_requires_all_real_fields(self) -> None:
        usage = TokenUsage.from_wire(
            {"inputTokens": 10, "outputTokens": 2, "embeddingTokens": 3},
            context="QA",
        )
        self.assertEqual(usage.total, 15)
        with self.assertRaises(TelemetryError):
            TokenUsage.from_wire(
                {"inputTokens": 10, "outputTokens": 2}, context="QA"
            )

    def test_qa_usage_requires_consistent_agent_and_search_breakdown(self) -> None:
        usage = QaTokenUsage.from_wire(
            {
                "inputTokens": 11,
                "outputTokens": 2,
                "embeddingTokens": 4,
                "agentInputTokens": 8,
                "agentOutputTokens": 1,
                "searchInputTokens": 3,
                "searchOutputTokens": 1,
            }
        )
        self.assertEqual(usage.total, 17)
        with self.assertRaises(TelemetryError):
            QaTokenUsage.from_wire(
                {
                    "inputTokens": 12,
                    "outputTokens": 2,
                    "embeddingTokens": 4,
                    "agentInputTokens": 8,
                    "agentOutputTokens": 1,
                    "searchInputTokens": 3,
                    "searchOutputTokens": 1,
                }
            )


if __name__ == "__main__":
    unittest.main()
