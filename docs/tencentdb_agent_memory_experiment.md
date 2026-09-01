# TencentDB Agent Memory baseline

This baseline targets only TencentDB's `MemoryKnowledge / LLM-Wiki` on the
`feat/server_team` branch. The commit is resolved and recorded in each corpus
manifest when a run starts.

MemoryKnowledge is a knowledge-service API rather than a complete agent
runtime. It does not provide the outer LLM loop, prompt orchestration, or the
PDF-ingest entrypoint used here; the benchmark runner supplies those pieces and
uses the documented `tools/list` and `tools/call` interface for QA.

Before running the ingest round, the standalone MemoryKnowledge service must
be configured with its own internal Wiki LLM binding. For a direct deployment,
use the TencentDB service's `LLM_MODE=custom`, `LLM_MODEL`, `LLM_BASE_URL`, and
`LLM_API_KEY` settings (or configure an equivalent Panel binding). Keep those
values aligned with the benchmark model configuration. The benchmark's
`ARK_API_KEY` is used for external QA/judge calls and is not automatically
passed into the MemoryKnowledge service.

## Data path

PDF sources are converted before upload using the local-compatible OpenViking
PDF strategy (`pdfplumber`, pinned to OpenViking commit
`447b30ef8511dcc82c07ede857a52150479ee77c`). Text and structured tables are
retained as Markdown. The no-wiki baseline additionally sends extracted images
to its vision model for image summaries; that is deliberately not done here:
MemoryKnowledge's `raw/write` contract is UTF-8 text-only and this baseline
does not add a multimodal model path. Images therefore become
`[Image omitted: page N, image M]` placeholders. Conversion time is part of
insertion time. Converted Markdown larger than 512 KiB is packed by Markdown
chapters into multiple source files.

Wiki search uses the service's BM25/FTS5 path with TencentDB default retrieval
parameters. This baseline does not configure a separate embedding model. Any
internal LLM work performed asynchronously by `wiki/ingest` is counted only
when the service exposes usage; otherwise it is recorded as incomplete with
zero tokens.

Run rounds independently:

```bash
ov-wiki-tencentdb EXPERIMENT_ID --data-dir /path/to/data --round ingest
ov-wiki-tencentdb EXPERIMENT_ID --data-dir /path/to/data --round qa
ov-wiki-tencentdb EXPERIMENT_ID --data-dir /path/to/data --round judge
ov-wiki-tencentdb EXPERIMENT_ID --data-dir /path/to/data --round delete
```

For multiple variants sharing one corpus, pass all experiment IDs together;
ingest and delete happen once while QA/judge outputs remain per experiment.
Use `--retry-failed-pdfs` on a later ingest round to reconvert only PDFs whose
conversion failed. This is not a retry of a failed HTTP/API call.

## Fault tolerance and accounting

`calls.jsonl` is append-only and fsynced after every call. A failed call never
contributes tokens. A successful model response with missing or malformed usage
is recorded with zero tokens and `token_usage_complete=false`; all prior
successful records remain valid and aggregation is rebuilt from the ledger.
Individual conversion, upload, tool, QA, and judge failures do not abort the
remaining items. Partial ingestion is reported as `partial_ingest=true`; all
QAs remain in the denominator and a failed judge contributes normalized
accuracy 0.

QA uses the supplied answer prompt and a 15-turn loop. It first discovers all
Wiki tools through `tools/list`, then lets the LLM call any returned tool through
`tools/call`; each QA has an isolated message context. Judge calls use the
supplied generic 0--4 prompt and report `score / 4` normalized accuracy.

Deletion is timed from the delete request start through the delete API response.
The complete Wiki resource, raw files, pages, SQLite/FTS5 data, graph data and
in-memory engine registration are deleted by the MemoryKnowledge delete path.
Experiment outputs, call ledger and reports are retained.
