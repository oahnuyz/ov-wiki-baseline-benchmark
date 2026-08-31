# Nashsu LLM Wiki benchmark bridge contract

This is the boundary between the Python benchmark runner and the pinned LLM Wiki
fork. It is benchmark-only control and telemetry: it must not change retrieval,
ranking, prompt content, Wiki generation, review resolution, or answer generation.

All routes are under `/api/v1/benchmark`, bind to loopback only, and require the
same `LLM_WIKI_API_TOKEN` authentication as paid/mutating stock API routes.

## Headless bootstrap

The Tauri process still runs the official WebView-side ingest pipeline, but the
window is hidden and no interactive desktop action is required. Startup requires:

```text
LLM_WIKI_BENCHMARK_HEADLESS=1
LLM_WIKI_BENCHMARK_PROJECT_PATH=/absolute/dedicated/project
LLM_WIKI_API_TOKEN=<random bridge token>
ARK_API_KEY=<Volcengine credential>
```

The frontend initializes an empty target with the official General scaffold,
opens it, installs the bridge listener, and marks the exact project ready. It
rejects a non-empty invalid directory and never returns either secret through
the readiness API.

## `GET /ready`

Returns HTTP 503 until both the frontend listener and exact dedicated project
are ready. A successful authenticated response is:

```json
{
  "ready": true,
  "projectPath": "/absolute/dedicated/project"
}
```

The Python runner waits for this response before creating the first run. Startup
and project-open time are outside all measured experiment stages.

## Required invariants

- LLM Wiki `0.6.11` at commit `e8082119649e6a8e1cf85eaf289adcabfdf39d4e`.
- `mode=standard`, `retrievalMode=standard`.
- Request omits `topK`; the backend reports the resolved official default `5`.
- `maxContextSize` is not overridden; the backend reports the official default
  `204800` characters.
- The dedicated project is restored before each corpus to the pinned, unedited
  General template with `outputLanguage=auto`, official default chunking, and
  `persistExtractedMarkdown=false`. The bridge reports SHA-256 for every
  scaffold file and the runner records them in the group manifest.
- Web search, AnyTXT, and skills are disabled.
- Every QA has an explicit empty history, a unique session, and
  `persistSession=false`.
- Main model, ingest model, caption model, and QA model are
  `doubao-seed-2-0-lite-260428` through Volcengine Ark.
- All model paths send `thinking.type=disabled`.
- Built-in PDF parsing and image captioning are enabled; MinerU is disabled.
- Vector retrieval is enabled with `doubao-embedding-vision-251215` and every
  returned vector is exactly 1024 dimensions.
- Token fields come from provider `usage`. Missing usage fails the operation.
- No retries are added by the bridge.

## Token object

Every completed stage returns these common fields, including zero-valued fields:

```json
{
  "inputTokens": 0,
  "outputTokens": 0,
  "embeddingTokens": 0
}
```

QA additionally requires `agentInputTokens`, `agentOutputTokens`,
`searchInputTokens`, and `searchOutputTokens`. Those four fields are omitted for
ingest and deletion. For QA, `inputTokens` must equal `agentInputTokens +
searchInputTokens`, and `outputTokens` must equal `agentOutputTokens +
searchOutputTokens`.

For ingest this is the sum of parsing-related model calls, image captions, Wiki
generation, review sweep, and page embeddings. For QA it is the complete Agent
chain, including search/route model calls and retrieval query embeddings. Deletion
must return three zeroes.

## `POST /runs`

Starts a benchmark run against the dedicated project opened by headless bootstrap
(or manually in normal desktop mode). It validates the complete fixed config and
resets only the in-memory telemetry accumulator.

Request:

```json
{
  "corpusId": "paperscope_summary-...",
  "projectPath": "/dedicated/benchmark/project",
  "continuation": false,
  "config": {}
}
```

Response:

```json
{
  "runId": "uuid",
  "resolvedMaxContextSize": 204800,
  "projectScaffold": {
    "template": "general",
    "outputLanguage": "auto",
    "chunking": "official_default",
    "persistExtractedMarkdown": false,
    "fileSha256": {
      "purpose.md": "...",
      "schema.md": "...",
      "wiki/index.md": "...",
      "wiki/overview.md": "...",
      "wiki/log.md": "..."
    }
  }
}
```

The route must reject a project mismatch and must never switch or delete an
unrelated project. A fresh run (`continuation=false`) rejects a project whose
`wiki/` contains data pages. After an intentional service restart, the runner may
create a continuation run (`continuation=true`) for the same corpus; this preserves
the existing knowledge base and validates that the protected General scaffold was
not manually edited.

## `POST /runs/{runId}/ingest`

The runner supplies canonical prepared corpus files in batches, together with a
zero-based `documentOffset` and `finalBatch`. Copying or hard-linking a batch into
`raw/sources` occurs before its timed region. Each batch ends after all of its tasks
and embeddings drain; only `finalBatch=true` runs the review sweep. Across the
complete corpus, the active insertion duration ends only after:

1. every source ingest task succeeds;
2. image extraction and captioning finish;
3. all generated pages are embedded;
4. the queue-drain review sweep finishes successfully;
5. all provider usage has been accumulated.

Response:

```json
{
  "status": "completed",
  "durationSeconds": 12.3,
  "usage": {
    "inputTokens": 1,
    "outputTokens": 2,
    "embeddingTokens": 3,
    "agentInputTokens": 1,
    "agentOutputTokens": 2,
    "searchInputTokens": 0,
    "searchOutputTokens": 0
  },
  "tokenUsageComplete": false,
  "llmTokenUsageComplete": false,
  "embeddingTokenUsageComplete": true,
  "reviewSweepCompleted": true,
  "finalBatch": true,
  "documentOffset": 75,
  "documentCount": 18,
  "embeddingDimensions": 1024
}
```

One failed source, caption, embedding, or sweep fails the relevant request. A
successfully completed batch always returns and accumulates every provider usage
value that was observed. If one or more successful provider calls omit usage,
`tokenUsageComplete` is false and the returned totals are a known lower bound;
the batch is not discarded and its known totals are never replaced with zero.
The benchmark totals provider usage and active duration over every batch. Service
restart time is recorded separately as operational wall-clock overhead.

## `POST /runs/{runId}/qa`

Accepts the stock Agent request fields plus the fully rendered approved prompt.
The benchmark endpoint measures from Agent invocation through final answer and
usage collection. It must not hydrate persisted history.

Response:

```json
{
  "status": "completed",
  "answer": "...",
  "sessionId": "...",
  "durationSeconds": 1.2,
  "resolvedTopK": 5,
  "usage": {
    "inputTokens": 1,
    "outputTokens": 2,
    "embeddingTokens": 3
  },
  "references": [],
  "trace": []
}
```

## `POST /runs/{runId}/delete`

Deletes all data created for the current corpus. The primary timer covers only
retrieval-visible state: generated Wiki pages and their derived graph candidates,
non-hidden raw sources addressable by `source.search`, and LanceDB page/chunk
vectors. Frontend queue/review/lint quiescence happens before the timer. Media,
hidden staging, parsed/caption/ingest/review/lint caches, remaining metadata,
project reactivation, run-registry cleanup, and service recovery happen after it.
Those two excluded phases are reported separately. The endpoint must reject any
target outside the run's exact dedicated project path.

Response:

```json
{
  "status": "completed",
  "durationSeconds": 0.5,
  "searchableDeletionDurationSeconds": 0.5,
  "frontendCleanupDurationSeconds": 0.1,
  "postDeletionCleanupDurationSeconds": 0.2,
  "onlySearchableDataIncludedInPrimaryTime": true,
  "timedDeletionScope": [
    "wiki-pages-and-derived-graph-data",
    "non-hidden-raw-source-search-data",
    "lancedb-page-and-chunk-vectors"
  ],
  "usage": {
    "inputTokens": 0,
    "outputTokens": 0,
    "embeddingTokens": 0
  }
}
```
