# OV-Wiki Baseline Benchmark

This repository prepares the six datasets and thirteen fixed experiment variants used by the OV-Wiki baseline evaluation. The `main` branch is baseline-neutral: it downloads pinned upstream snapshots, validates them, applies shared gold-answer transformations, and emits one canonical corpus and QA format. Baseline-specific branches should only add ingestion, retrieval, generation, and evaluation integration.

## Supported experiment variants

| Experiment ID | QA | Documents |
|---|---:|---:|
| `paperscope_summary_57_trend` | 117 | 57 PDFs |
| `paperscope_summary_57_gap` | 119 | 57 PDFs |
| `paperscope_summary_57_results_comparison` | 116 | 57 PDFs |
| `paperscope_summary_93_trend` | 117 | 93 PDFs |
| `paperscope_summary_93_gap` | 119 | 93 PDFs |
| `paperscope_summary_93_results_comparison` | 116 | 93 PDFs |
| `mdaqa_first_100` | 100 | 143 PDFs |
| `wildgraphbench_summary_all` | 339 | 3,894 TXT files |
| `wildgraphbench_summary_health` | 55 | 509 TXT files |
| `scholarqa_multi_valid_101` | 101 | 413 TXT files |
| `mudabench_simple` | 166 | 589 PDFs |
| `mudabench_complex` | 166 | 589 PDFs |
| `enterprise_rag_bench_selected_80` | 80 | 323 TXT files |

Shared raw caches and hard links avoid storing the same PaperScope or MuDABench corpus multiple times.

## Environment

Python 3.10 or newer is required. With `uv`:

```bash
uv sync
```

List all fixed variants:

```bash
uv run ov-wiki-data list
```

Prepare one variant:

```bash
uv run ov-wiki-data prepare enterprise_rag_bench_selected_80
```

Prepare several variants in one invocation:

```bash
uv run ov-wiki-data prepare \
  mudabench_simple \
  mudabench_complex
```

Prepare all thirteen variants:

```bash
uv run ov-wiki-data prepare --all
```

Use an existing raw cache without downloading:

```bash
uv run ov-wiki-data prepare mdaqa_first_100 --skip-download
```

Verify already prepared data:

```bash
uv run ov-wiki-data verify --all
```

Use `--data-dir /some/path` to place large downloads outside the repository. The default is `data/`.

PaperScope PDFs may require an interactive temporary OpenReview login. Credentials are read from the terminal and are not written to disk.

## Output layout

Each experiment variant has an independent prepared directory:

```text
data/
├── raw/
│   ├── shared download caches
│   └── verified dataset-native snapshots
└── prepared/
    └── <experiment_id>/
        ├── qa.jsonl
        ├── documents.jsonl
        ├── dataset_info.json
        └── corpus/
```

### Canonical QA schema

Every line of `qa.jsonl` has the same structure:

```json
{
  "schema_version": "1.0",
  "id": "dataset-specific unique physical QA ID",
  "dataset": "dataset key",
  "variant": "experiment ID",
  "question": "question text",
  "gold_answers": ["complete normalized gold answer"],
  "evidence": ["evidence or answer fact"],
  "category": "question category",
  "document_ids": ["canonical document ID"],
  "metadata": {
    "original_record": {}
  }
}
```

`metadata.original_record` retains the complete upstream QA record. `gold_answers` is always a non-empty list. `evidence` and `document_ids` may be empty when the upstream dataset does not provide a resolvable mapping.

### Canonical document schema

Every line of `documents.jsonl` describes one physical corpus file:

```json
{
  "schema_version": "1.0",
  "id": "canonical physical document ID",
  "dataset": "dataset key",
  "source_id": "upstream logical document ID",
  "path": "corpus/relative/path.txt",
  "media_type": "text/plain",
  "size_bytes": 1234,
  "sha256": "...",
  "metadata": {
    "original_record": {}
  }
}
```

The EnterpriseRAG-Bench conflict case intentionally has two physical `id` values sharing one `source_id`.

## Shared gold-answer transformations

- **PaperScope Summary:** `answer` becomes the sole gold answer; each QA retains its prompt type and supporting paper IDs.
- **MDA-QA:** `answer` becomes the sole gold answer; `support` is mapped to canonical paper IDs.
- **WildGraphBench Summary:** all `gold_statements` are joined as one bullet-list gold answer. Original statements and reference URLs remain in `metadata.original_record`.
- **ScholarQA-Multi:** the original expert answer is followed by the zero-based citation-number-to-title reference key. Context text becomes evidence.
- **MuDABench:** `final_answer` becomes the sole gold answer and `source_answer` becomes evidence. The two exact duplicate rows remain separate physical QA records.
- **EnterpriseRAG-Bench:** `gold_answer` becomes the sole gold answer and `answer_facts` becomes evidence. Repeated logical document IDs are mapped to separate physical files.

## OV-Wiki bot answer prompt

The current OV-Wiki `vikingbot` path sends one shared instruction rather than the dataset adapters' `build_prompt()` methods. The machine-readable template is [`prompts/ov_wiki_bot_answer.txt`](prompts/ov_wiki_bot_answer.txt).

```text
Answer this question as briefly as possible. Use only the information available in the database. Do not use any external source. Always use OpenViking tools first. Search first, then read the results to answer. Use the default OpenViking search scope; do not force a specific target_uri unless needed. Search results may come from original resources or wiki nodes. If wiki node documents are relevant, read them and use them as evidence together with original resources when useful.

Question: {question}
```

## Generic 0-4 LLM judge

All six datasets use the generic judge rather than the LoCoMo-specific binary judge. Machine-readable templates:

- [`prompts/generic_llm_judge_system.txt`](prompts/generic_llm_judge_system.txt)
- [`prompts/generic_llm_judge_user.txt`](prompts/generic_llm_judge_user.txt)

System prompt:

```text
You are an expert evaluator scoring how well an AI-generated answer matches a gold standard (ground truth).
```

User prompt:

```text
Please score the Generated Answer against the Gold Answers on a scale of 0 to 4.

[Evaluation Rubric]
- Score 4 (Perfect): Fully and accurately captures the core meaning and key facts of any of the Gold Answers. Additional relevant explanation or context is acceptable and does NOT reduce the score, as long as it is consistent with and does not contradict the Gold Answers. Minor differences in wording, capitalization, punctuation, or phrasing are acceptable if the core meaning is preserved.
- Score 3 (Good): Correctly captures the main answer and most key facts, but has minor issues such as slight imprecision, small omissions of non-critical details, or wording that is somewhat vague or ambiguous. The overall answer is still clearly correct.
- Score 2 (Partial): Partially correct, but missing at least one important fact, condition, or detail needed for a fully correct answer. The answer is related to the correct topic, but is incomplete or insufficient.
- Score 1 (Poor): Mostly incorrect, seriously incomplete, or only weakly related to the Gold Answers.
- Score 0 (Wrong): Incorrect, contradictory to the Gold Answers, or contains fabricated / hallucinated core content.

Important Notes:
- Gold answers are multiple possible correct answers separated by " | ". The generated answer only needs to match any one of them.
- The gold answers may be concise, but the generated answer can be longer and include additional explanations - this is acceptable for Score 4 as long as the core information is correct.
- Do NOT penalize for additional relevant information that doesn't contradict the gold answers. Examples of acceptable extra information: titles ("King Padella" vs "Padella"), locations ("Paflagonia" vs "the capital of Paflagonia"), or additional context that supports the answer.
- Only penalize for actual incorrect information, missing key facts, or contradictions.
- Ignore minor differences in capitalization (e.g., "CRIM TARTARY" vs "Crim Tartary") or punctuation (e.g., with or without a period at the end).

Question: {question}
Gold Answers: {gold_answers_joined_by_pipe}
Generated Answer: {generated_answer}

First, briefly explain the rating in 1 sentence. Then output the integer score.
Respond ONLY with a JSON object: {"score": 0 to 4, "reasoning": "string"}
```

## Metric alignment

- Token-level F1 is computed against every item in `gold_answers`, taking the maximum score.
- Accuracy sends all gold answers to the generic judge and uses its integer score from 0 to 4.
- If both the generated answer and at least one gold answer are recognized as refusal/unanswerable responses, F1 is set to `1.0` and Accuracy to `4`.

## Baseline branch workflow

Keep dataset download, normalization, schemas, configs, and prompts on `main`. Create one branch per baseline:

```bash
git switch main
git switch -c baseline/<baseline-name>
```

A baseline branch should consume `qa.jsonl`, `documents.jsonl`, and `corpus/`, and add only its own ingestion, retrieval, answer generation, metric execution, and logging integration. Merge updates from `main` into long-lived baseline branches when the shared data layer changes.
