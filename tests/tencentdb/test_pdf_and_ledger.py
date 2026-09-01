from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from ov_wiki_baseline_benchmark.specs import ExperimentSpec
from ov_wiki_baseline_benchmark.tencentdb.config import TencentDBConfig
from ov_wiki_baseline_benchmark.tencentdb.ledger import CallLedger
from ov_wiki_baseline_benchmark.tencentdb.llm import OpenAICompatibleClient
from ov_wiki_baseline_benchmark.tencentdb.pdf import split_markdown_by_chapter
from ov_wiki_baseline_benchmark.tencentdb.runner import PreparedTencentExperiment, TencentDBRunner, _tool_schema


class TencentDBPdfLedgerTests(unittest.TestCase):
    def test_split_is_utf8_safe_and_chapter_aware(self) -> None:
        content = "# 第一章\n\n" + ("甲" * 40) + "\n\n# 第二章\n\n" + ("乙" * 40)
        parts = split_markdown_by_chapter(content, max_bytes=100)
        self.assertGreater(len(parts), 1)
        for part in parts:
            self.assertLessEqual(len(part.encode("utf-8")), 100)
            part.encode("utf-8").decode("utf-8")

    def test_split_never_exceeds_limit_with_trailing_newline(self) -> None:
        parts = split_markdown_by_chapter("a" * 200, max_bytes=100)
        self.assertEqual(len(parts), 2)
        self.assertTrue(all(len(part.encode("utf-8")) <= 100 for part in parts))

    def test_ledger_normalizes_bad_usage_and_aggregates_successes(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            ledger = CallLedger(Path(temp) / "calls.jsonl")
            ledger.append(phase="qa", operation="llm", status="success", started_at=1, ended_at=2, input_tokens=3, output_tokens=4, token_usage_complete=True)
            ledger.append(phase="qa", operation="llm", status="failed", started_at=2, ended_at=3, input_tokens=99, output_tokens=99, token_usage_complete=False, metadata={"token_bearing": True})
            ledger.append(phase="qa", operation="tools.call", status="success", started_at=3, ended_at=4, metadata={"token_bearing": False})
            aggregate = ledger.aggregate(phase="qa")
            self.assertEqual(aggregate["total_tokens"], 7)
            self.assertEqual(aggregate["incomplete_usage_calls"], 1)
            rows = [json.loads(line) for line in (Path(temp) / "calls.jsonl").read_text().splitlines()]
            self.assertEqual(rows[1]["total_tokens"], 0)

    def test_service_usage_audit_distinguishes_control_and_token_calls(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            ledger = CallLedger(Path(temp) / "calls.jsonl")
            TencentDBRunner._service(
                ledger, "ingest", "wiki.raw.write", lambda: {"items": []},
                experiment_id="a",
            )
            TencentDBRunner._service(
                ledger, "ingest", "wiki.ingest", lambda: {"status": "pending"},
                experiment_id="a",
            )
            TencentDBRunner._service(
                ledger, "qa", "tools.call.search",
                lambda: {"usage": {"input_tokens": 3, "output_tokens": 4}},
                qa_id="q", experiment_id="b",
            )
            self.assertEqual(ledger.aggregate(phase="ingest")["incomplete_usage_calls"], 1)
            self.assertEqual(ledger.aggregate(phase="qa", experiment_id="a")["total_tokens"], 0)
            self.assertEqual(ledger.aggregate(phase="qa", experiment_id="b")["total_tokens"], 0)

    def test_tool_schema_preserves_all_returned_tools_and_required_params(self) -> None:
        value = _tool_schema({
            "name": "read_page",
            "description": "read",
            "params": {"refs": {"type": "array", "required": True}},
        })
        self.assertEqual(value["function"]["name"], "read_page")
        self.assertEqual(value["function"]["parameters"]["required"], ["refs"])
        self.assertEqual(value["function"]["parameters"]["properties"]["refs"]["type"], "array")

    def test_judge_resume_does_not_duplicate_rows_and_keeps_denominator(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            prompt = root / "prompt.txt"
            prompt.write_text("{question} {gold_answers_joined_by_pipe} {generated_answer}")
            config = TencentDBConfig(
                base_url="http://unused", service_id="s", team_id="t", user_id="u",
                llm_model="m", llm_base_url="http://unused", llm_api_key_env="KEY",
                output_dir=root / "out",
            )
            runner = TencentDBRunner(config, answer_prompt_path=prompt, judge_prompt_path=prompt)
            spec = ExperimentSpec("exp", "dataset", "raw", 2, 1, {})
            experiment = PreparedTencentExperiment(
                spec, root, [{"id": "doc", "path": "doc.md", "sha256": "x"}],
                [{"id": "q1", "question": "one", "gold_answers": ["1"]}, {"id": "q2", "question": "two", "gold_answers": ["2"]}], "abc",
            )
            group = root / "group"
            group.mkdir()
            (group / "exp.qa.jsonl").write_text(
                '{"qa_id":"q1","question":"one","answer":"1","gold_answers":["1"]}\n'
                '{"qa_id":"q2","question":"two","answer":"2","gold_answers":["2"]}\n'
            )
            (group / "exp.judge.jsonl").write_text(
                '{"qa_id":"q1","score":4,"normalized_accuracy":1.0,"reasoning":"saved"}\n'
            )

            class FakeLLM:
                def __init__(self, config, ledger):
                    pass

                def complete(self, **kwargs):
                    return {"choices": [{"message": {"content": '{"score": 2, "reasoning": "new"}'}}]}

            with patch("ov_wiki_baseline_benchmark.tencentdb.runner.OpenAICompatibleClient", FakeLLM):
                runner._run_judge(experiment, group, CallLedger(group / "calls.jsonl"))
                runner._run_judge(experiment, group, CallLedger(group / "calls.jsonl"))
            rows = (group / "exp.judge.jsonl").read_text().splitlines()
            self.assertEqual(len(rows), 2)
            summary = runner._load_json(group / "exp.judge.summary.json")
            self.assertEqual(summary["count"], 2)
            self.assertEqual(summary["expected_count"], 2)

    def test_tool_failure_is_returned_to_agent_and_next_turn_can_finish(self) -> None:
        class FakeAgent(OpenAICompatibleClient):
            def __init__(self):
                self.config = SimpleNamespace(max_loop_turns=15)
                self.responses = [
                    {"choices": [{"message": {"content": "", "tool_calls": [{"id": "c1", "function": {"name": "search", "arguments": '{"query":"x"}'}}]}}]},
                    {"choices": [{"message": {"content": "final answer"}}]},
                ]

            def complete(self, **kwargs):
                return self.responses.pop(0)

        def fail_tool(name, arguments):
            raise RuntimeError("tool unavailable")

        answer, _duration, turns = FakeAgent().answer_with_tools(
            wiki_id="wiki-1", question="q", answer_prompt="", tools=[{"type": "function"}],
            call_tool=fail_tool, qa_id="q1", experiment_id="exp",
        )
        self.assertEqual(answer, "final answer")
        self.assertEqual(turns, 2)

    def test_failed_pdf_can_be_retried_without_reprocessing_uploaded_pdf(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            prompt = root / "prompt.txt"
            prompt.write_text("{question}")
            first = root / "first.pdf"
            second = root / "second.pdf"
            first.write_bytes(b"one")
            second.write_bytes(b"two")
            config = TencentDBConfig(
                base_url="http://unused", service_id="s", team_id="t", user_id="u",
                llm_model="m", llm_base_url="http://unused", llm_api_key_env="KEY",
                output_dir=root / "out",
            )
            runner = TencentDBRunner(config, answer_prompt_path=prompt, judge_prompt_path=prompt)
            experiment = PreparedTencentExperiment(
                ExperimentSpec("exp", "dataset", "raw", 0, 2, {}), root,
                [{"id": "one", "path": "first.pdf"}, {"id": "two", "path": "second.pdf"}],
                [], "abc",
            )

            class FakeClient:
                writes = []

                def raw_write(self, wiki_id, files):
                    self.writes.extend(item["filename"] for item in files)
                    return {"items": []}

                def ingest(self, wiki_id):
                    return {"status": "pending"}

                def wait_ready(self, wiki_id):
                    return {"status": "ready"}

            client = FakeClient()
            manifest = {}

            def first_conversion(path):
                if path == second:
                    raise RuntimeError("bad pdf")
                return "# First\ntext"

            group = root / "group"
            group.mkdir()
            with patch("ov_wiki_baseline_benchmark.tencentdb.runner.read_source_as_markdown", first_conversion):
                runner._run_ingest(experiment, "wiki-1", group, manifest, CallLedger(group / "calls.jsonl"), client, False)
            self.assertEqual([r["status"] for r in manifest["documents"]], ["uploaded", "conversion_failed"])
            self.assertTrue(manifest["partial_ingest"])

            retried_paths = []
            def retry_conversion(path):
                retried_paths.append(path)
                return "# Retried\ntext"

            with patch("ov_wiki_baseline_benchmark.tencentdb.runner.read_source_as_markdown", retry_conversion):
                runner._run_ingest(experiment, "wiki-1", group, manifest, CallLedger(group / "calls.jsonl"), client, True)
            self.assertEqual(retried_paths, [second])
            self.assertEqual([r["status"] for r in manifest["documents"]], ["uploaded", "uploaded"])
            self.assertFalse(manifest["partial_ingest"])


if __name__ == "__main__":
    unittest.main()
