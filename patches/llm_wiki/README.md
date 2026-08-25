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

The benchmark runner does not accept the stock LLM Wiki API as a benchmark bridge.
Stock `AgentUsage` reports character counts, not provider tokens, and the stock API
does not expose an atomic ingest-plus-review-sweep operation. Apply both patches to
obtain the bridge specified in [`bridge_contract.md`](bridge_contract.md); the runner
never substitutes character estimates.
