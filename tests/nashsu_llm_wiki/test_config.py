from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ov_wiki_baseline_benchmark.nashsu_llm_wiki.config import BenchmarkConfig
from ov_wiki_baseline_benchmark.specs import repository_root


class ConfigTests(unittest.TestCase):
    def test_committed_config_resolves_pinned_official_defaults(self) -> None:
        config = BenchmarkConfig.from_yaml(
            repository_root() / "baseline_configs" / "nashsu_llm_wiki.yaml"
        )
        manifest = config.public_manifest()
        self.assertEqual(manifest["llmWiki"]["topK"]["resolvedValue"], 5)
        self.assertEqual(
            manifest["llmWiki"]["maxContextSize"]["resolvedValue"], 204800
        )
        self.assertEqual(manifest["llmWiki"]["projectTemplate"], "general")
        self.assertEqual(manifest["llmWiki"]["outputLanguage"], "auto")
        self.assertEqual(manifest["llmWiki"]["chunking"], "official_default")
        self.assertIs(manifest["llmWiki"]["persistExtractedMarkdown"], False)

    def test_config_rejects_parallel_qa(self) -> None:
        source = (
            repository_root() / "baseline_configs" / "nashsu_llm_wiki.yaml"
        ).read_text(encoding="utf-8")
        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / "bad.yaml"
            path.write_text(source.replace("qa_workers: 1", "qa_workers: 2"), encoding="utf-8")
            with self.assertRaises(ValueError):
                BenchmarkConfig.from_yaml(path)


if __name__ == "__main__":
    unittest.main()
