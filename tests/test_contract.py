from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ov_wiki_baseline_benchmark.io import sha256_file
from ov_wiki_baseline_benchmark.schema import (
    SCHEMA_VERSION,
    validate_documents,
    validate_qas,
)
from ov_wiki_baseline_benchmark.specs import load_specs


class ContractTests(unittest.TestCase):
    def test_thirteen_experiment_specs(self) -> None:
        specs = load_specs()
        self.assertEqual(len(specs), 13)
        self.assertEqual(specs["enterprise_rag_bench_selected_80"].expected_qas, 80)
        self.assertEqual(specs["mudabench_complex"].expected_documents, 589)

    def test_canonical_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            root = Path(temp_name)
            corpus_file = root / "corpus" / "document.txt"
            corpus_file.parent.mkdir(parents=True)
            corpus_file.write_text("evidence", encoding="utf-8")
            documents = [
                {
                    "schema_version": SCHEMA_VERSION,
                    "id": "doc:1",
                    "dataset": "fixture",
                    "source_id": "source-1",
                    "path": "corpus/document.txt",
                    "media_type": "text/plain",
                    "size_bytes": corpus_file.stat().st_size,
                    "sha256": sha256_file(corpus_file),
                    "metadata": {"original_record": {"id": "source-1"}},
                }
            ]
            qas = [
                {
                    "schema_version": SCHEMA_VERSION,
                    "id": "qa:1",
                    "dataset": "fixture",
                    "variant": "fixture",
                    "question": "Question?",
                    "gold_answers": ["Answer."],
                    "evidence": ["evidence"],
                    "category": "fixture",
                    "document_ids": ["doc:1"],
                    "metadata": {"original_record": {"question": "Question?"}},
                }
            ]
            validate_documents(root, documents, expected_count=1)
            validate_qas(qas, documents, expected_count=1)


if __name__ == "__main__":
    unittest.main()
