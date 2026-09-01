# LLM Wiki fork patches

These patches target the exact upstream revision used by this baseline:

- repository: `https://github.com/nashsu/llm_wiki.git`
- version: `0.6.11`
- commit: `e8082119649e6a8e1cf85eaf289adcabfdf39d4e`

They are intentionally stored as patches instead of vendoring a second copy of the
entire upstream repository.

Apply them only to a clean checkout at the pinned commit:

```bash
git rev-parse HEAD
git apply --check /path/to/patches/llm_wiki/0001-volcengine-thinking-and-dimension.patch
git apply /path/to/patches/llm_wiki/0001-volcengine-thinking-and-dimension.patch
git apply --check /path/to/patches/llm_wiki/0002-benchmark-bridge-and-telemetry.patch
git apply /path/to/patches/llm_wiki/0002-benchmark-bridge-and-telemetry.patch
git apply --check /path/to/patches/llm_wiki/0003-webkit-request-timeout-fallback.patch
git apply /path/to/patches/llm_wiki/0003-webkit-request-timeout-fallback.patch
git apply --check /path/to/patches/llm_wiki/0004-restart-safe-batched-ingest.patch
git apply /path/to/patches/llm_wiki/0004-restart-safe-batched-ingest.patch

git apply --check /path/to/patches/llm_wiki/0005-partial-ingest-usage-and-doubao-dimensions.patch
git apply /path/to/patches/llm_wiki/0005-partial-ingest-usage-and-doubao-dimensions.patch

git apply --check /path/to/patches/llm_wiki/0006-restore-empty-project-after-delete.patch
git apply /path/to/patches/llm_wiki/0006-restore-empty-project-after-delete.patch

git apply --check /path/to/patches/llm_wiki/0007-reactivate-empty-project-after-delete.patch
git apply /path/to/patches/llm_wiki/0007-reactivate-empty-project-after-delete.patch

git apply --check /path/to/patches/llm_wiki/0008-searchable-only-deletion-telemetry.patch
git apply /path/to/patches/llm_wiki/0008-searchable-only-deletion-telemetry.patch

git apply --check /path/to/patches/llm_wiki/0009-benchmark-qa-json-trace.patch
git apply /path/to/patches/llm_wiki/0009-benchmark-qa-json-trace.patch
```

`0001` makes the approved model settings enforceable on both the TypeScript ingest
path and Rust Agent path:

- `reasoning.mode=off` becomes Volcengine's `thinking: {"type":"disabled"}`;
- configured embedding dimensionality becomes a fail-closed response-length check;
- tests cover the Volcengine thinking request shape.

`0002` implements the benchmark-only loopback bridge and fail-closed provider
telemetry described in `bridge_contract.md`:

- atomic ingest timing through the queue-drain review sweep;
- provider-reported LLM and embedding token totals, with missing usage rejected;
- isolated `standard` QA using the omitted official `topK=5` default;
- deterministic restoration and hashing of the unedited General project
  scaffold, official `auto` output language, default chunking, and disabled
  parsed-Markdown persistence before every corpus;
- benchmark-owned project cleanup with zero model-token cost;
- fixed model/routing, built-in PDF parsing, captions, vector retrieval, and
  disabled web/AnyTXT/skills.
- environment-driven benchmark headless startup: the main window stays hidden,
  the dedicated General project is initialized/opened without user interaction,
  and an authenticated readiness endpoint prevents startup races.

`0003` keeps the existing per-model-call timeout active on older WebKitGTK
runtimes that do not expose `AbortSignal.timeout`. This prevents a dropped
provider response from leaving an ingest task permanently stuck in
`processing`; the queue can receive the timeout error and use its normal retry
policy.

`0004` makes long corpus ingestion restart-safe without changing ingest concurrency:

- each non-final batch drains completely without running the review sweep;
- the runner may restart the headless WebKit service and open an explicitly
  validated continuation run against the same corpus knowledge base;
- the final batch performs exactly one review sweep;
- deletion removes staging directories created by every continuation run.

`0005` preserves all provider-reported ingest usage even when some calls omit
usage, sends the configured 1024-dimensional request to Doubao multimodal
embeddings, and excludes reusable-project recovery work from deletion timing.

`0006` restores the empty General scaffold after that deletion timer has stopped,
so a subsequent WebKit restart can reopen the dedicated project without adding
project recovery time to the deletion metric.

`0007` reopens the recovered empty project in the frontend after deletion. The
runner still restarts the service at the next formal experiment boundary because
WebKit file-watch events can subsequently clear the in-memory active project.

`0008` narrows primary deletion timing to Wiki/graph pages, non-hidden raw-source
search data, and LanceDB vectors. Frontend quiescence plus non-searchable cleanup
and project recovery are reported separately for audit.

`0009` requests strict JSON objects for benchmark Agent decisions and writes each
QA Agent event to a flushed JSONL trace as it happens. Generation start/end events
make a slow or interrupted question diagnosable before the HTTP response finishes.

The benchmark runner does not accept the stock LLM Wiki API as a benchmark bridge.
Stock `AgentUsage` reports character counts, not provider tokens, and the stock API
does not expose an atomic ingest-plus-review-sweep operation. Apply the patches to
obtain the bridge specified in [`bridge_contract.md`](bridge_contract.md); the runner
never substitutes character estimates. Apply all nine patches in order.
