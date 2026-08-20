# OV-Wiki Baseline Benchmark

[中文说明](README_zh.md)

This repository provides a baseline-neutral data preparation layer for the OV-Wiki benchmark. It downloads and verifies six upstream datasets, prepares thirteen fixed experiment variants, normalizes their corpora and QA records, and preserves the prompts used by the OV-Wiki experiment for consistent generation and evaluation.

The `main` branch contains only shared dataset preparation. A baseline-specific branch can consume the prepared files and add its own ingestion, retrieval, generation, evaluation, and logging code.

## 1. Common commands

### Environment setup

Python 3.10 or newer is required. Install the project dependencies with `uv`:

```bash
uv sync
```

The examples below use the `ov-wiki-data` command registered by this project.

### List all experiment variants

```bash
uv run ov-wiki-data list
```

### Download and prepare data

Prepare one experiment:

```bash
uv run ov-wiki-data prepare enterprise_rag_bench_selected_80
```

Prepare several experiments in one invocation:

```bash
uv run ov-wiki-data prepare \
  mudabench_simple \
  mudabench_complex
```

Prepare all thirteen experiments:

```bash
uv run ov-wiki-data prepare --all
```

Use a custom data directory for large downloads and prepared artifacts:

```bash
uv run ov-wiki-data prepare mdaqa_first_100 \
  --data-dir /path/to/benchmark-data
```

By default, data is written to `data/` in the repository.

### Reuse or refresh raw data

Prepare canonical outputs from an already downloaded and verified raw dataset:

```bash
uv run ov-wiki-data prepare mdaqa_first_100 --skip-download
```

Force the downloader to refresh the upstream artifacts before preparation:

```bash
uv run ov-wiki-data prepare mdaqa_first_100 --force-download
```

`--skip-download` and `--force-download` are mutually exclusive in intent and should not be used together.

PaperScope PDF downloads may require an interactive temporary OpenReview login. Credentials are read from the terminal and are not written to disk.

### Verify prepared data

Verify one experiment:

```bash
uv run ov-wiki-data verify scholarqa_multi_valid_101
```

Verify all prepared experiments:

```bash
uv run ov-wiki-data verify --all
```

Verification checks the expected QA and document counts, schema fields, unique IDs and paths, referenced document IDs, file sizes, and SHA-256 checksums.

### Run contract tests

```bash
uv run python -m unittest discover -s tests -v
```

### Prepared output layout

Each experiment has an independent output directory. Shared raw caches are reused when multiple experiments need the same upstream corpus.

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

`qa.jsonl` contains normalized QA records. `documents.jsonl` is the corpus manifest, while `corpus/` contains the corresponding PDF or TXT files. `dataset_info.json` records the experiment identity, counts, and source-preparation information.

## 2. Prompt alignment

The repository keeps machine-readable prompt templates under [`prompts/`](prompts/). Baseline branches should load these templates instead of maintaining private copies, so that answer generation and accuracy evaluation remain aligned with the OV-Wiki experiment.

This repository prepares data and prompt contracts; a baseline branch is responsible for calling its model and evaluator.

### Answer-generation prompt

Source of truth: [`prompts/ov_wiki_bot_answer.txt`](prompts/ov_wiki_bot_answer.txt)

The template is aligned with the OV-Wiki `vikingbot` experiment path. `{question}` is replaced with the current canonical QA record's `question` field.

```text
Answer this question as briefly as possible. Use only the information available in the database. Do not use any external source. Always use OpenViking tools first. Search first, then read the results to answer. Use the default OpenViking search scope; do not force a specific target_uri unless needed. Search results may come from original resources or wiki nodes. If wiki node documents are relevant, read them and use them as evidence together with original resources when useful.

Question: {question}
```

### Generic 0–4 LLM judge

All thirteen experiments use the same generic LLM judge contract. The machine-readable sources of truth are:

- System prompt: [`prompts/generic_llm_judge_system.txt`](prompts/generic_llm_judge_system.txt)
- User prompt: [`prompts/generic_llm_judge_user.txt`](prompts/generic_llm_judge_user.txt)

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

The evaluator fills the placeholders as follows:

- `{question}`: `qa.jsonl` → `question`
- `{gold_answers_joined_by_pipe}`: all values in `gold_answers`, joined with ` | `
- `{generated_answer}`: the baseline model's final answer

For metric alignment, token-level F1 is computed against every value in `gold_answers` and takes the maximum. Accuracy is the generic judge's integer score from 0 to 4. If both the generated answer and at least one gold answer are recognized as refusal or unanswerable responses, F1 is `1.0` and Accuracy is `4`.

## 3. Supported experiment variants

The project supports six datasets and thirteen fixed experiment configurations.

| Experiment ID | Dataset and scope | QA | Corpus |
|---|---|---:|---:|
| `paperscope_summary_57_trend` | PaperScope Summary, `trend`, 57-paper corpus | 117 | 57 PDFs |
| `paperscope_summary_57_gap` | PaperScope Summary, `gap`, 57-paper corpus | 119 | 57 PDFs |
| `paperscope_summary_57_results_comparison` | PaperScope Summary, `results_comparison`, 57-paper corpus | 116 | 57 PDFs |
| `paperscope_summary_93_trend` | PaperScope Summary, `trend`, 93-paper corpus | 117 | 93 PDFs |
| `paperscope_summary_93_gap` | PaperScope Summary, `gap`, 93-paper corpus | 119 | 93 PDFs |
| `paperscope_summary_93_results_comparison` | PaperScope Summary, `results_comparison`, 93-paper corpus | 116 | 93 PDFs |
| `mdaqa_first_100` | First 100 MDA-QA records | 100 | 143 PDFs |
| `wildgraphbench_summary_all` | All WildGraphBench Summary topics | 339 | 3,894 TXT files |
| `wildgraphbench_summary_health` | WildGraphBench Summary, Health only | 55 | 509 TXT files |
| `scholarqa_multi_valid_101` | Valid ScholarQA-Multi records | 101 | 413 merged TXT files |
| `mudabench_simple` | MuDABench Simple QA, complete corpus | 166 | 589 PDFs |
| `mudabench_complex` | MuDABench Complex QA, complete corpus | 166 | 589 PDFs |
| `enterprise_rag_bench_selected_80` | Project Related, Conflicting Info, and Completeness | 80 | 323 TXT files |

The experiment IDs are fixed by the YAML files in [`configs/`](configs/). Each configuration declares the dataset handler, raw snapshot name, expected counts, and experiment-specific selection options.

### Shared gold-answer normalization

- **PaperScope Summary:** `answer` becomes the sole gold answer. Prompt type and supporting paper IDs remain available.
- **MDA-QA:** `answer` becomes the sole gold answer. `support` is mapped to canonical paper IDs.
- **WildGraphBench Summary:** `gold_statements` are joined into one bullet-list gold answer. Original statements and reference URLs are preserved in `metadata.original_record`.
- **ScholarQA-Multi:** the original expert answer is retained and followed by a zero-based citation-number-to-title key. Context text becomes evidence.
- **MuDABench:** `final_answer` becomes the gold answer and `source_answer` becomes evidence. The two fully duplicated QA rows remain separate physical records.
- **EnterpriseRAG-Bench:** `gold_answer` becomes the gold answer and `answer_facts` becomes evidence. Repeated logical documents are represented by separate physical document IDs.

## 4. Code module guide

### End-to-end flow

```text
configs/*.yaml
    ↓
specs.py loads and validates the experiment definition
    ↓
cli.py selects list / prepare / verify
    ↓
runner.py dispatches the dataset downloader
    ↓
datasets/<dataset>.py downloads and verifies the upstream snapshot
    ↓
normalize.py creates the canonical QA, manifest, and corpus
    ↓
schema.py validates the prepared experiment
```

### Top-level files and directories

| Path | Responsibility |
|---|---|
| `configs/` | Thirteen declarative experiment specifications. Counts and selection options live here instead of in the CLI. |
| `prompts/` | Machine-readable answer-generation and LLM-judge templates shared by baseline branches. |
| `src/ov_wiki_baseline_benchmark/` | Installable Python package containing the common preparation pipeline. |
| `tests/` | Contract tests for the thirteen configurations and canonical schemas. |
| `pyproject.toml` | Python version, dependencies, packaging metadata, and the `ov-wiki-data` console entry point. |
| `.gitignore` | Excludes environments, caches, generated data, and local editor artifacts. |

### Core Python modules

| Module | Responsibility |
|---|---|
| `__main__.py` | Enables `python -m ov_wiki_baseline_benchmark`. |
| `cli.py` | Parses `list`, `prepare`, and `verify`, validates experiment selection, and prints results. |
| `specs.py` | Loads all YAML files into `ExperimentSpec` objects and validates the fixed thirteen-config contract. |
| `runner.py` | Maps each dataset key to its downloader, manages raw/prepared paths, and coordinates download → normalize → verify. |
| `normalize.py` | Converts dataset-native records into shared `qa.jsonl`, `documents.jsonl`, `dataset_info.json`, and `corpus/`; it also applies gold-answer transformations. |
| `schema.py` | Enforces schema version, required fields, unique IDs/paths, valid document references, sizes, and checksums. |
| `io.py` | Shared JSON/JSONL I/O, safe relative paths, checksums, hard-link/copy behavior, and atomic directory replacement. |

### Dataset modules

| Module | Responsibility |
|---|---|
| `datasets/paperscope_summary.py` | Selects PaperScope Summary QA by type and prepares the shared 57- or 93-PDF corpus. |
| `datasets/mdaqa.py` | Selects the first 100 MDA-QA records and downloads the 143 referenced arXiv PDFs. |
| `datasets/wildgraphbench.py` | Prepares all Summary topics or the Health-only subset from reference-page TXT documents. |
| `datasets/scholarqa_multi.py` | Removes records with invalid citation indices, merges official citation contexts into 413 TXT documents, and retains 101 valid QA records. |
| `datasets/mudabench.py` | Prepares the full 589-PDF corpus and independently selects Simple or Complex QA records. |
| `datasets/enterprise_rag_bench.py` | Downloads the official archive but extracts only the 323 physical documents required by the three selected categories. |

### Canonical data contract

Each line of `qa.jsonl` contains:

```json
{
  "schema_version": "1.0",
  "id": "unique physical QA ID",
  "dataset": "dataset key",
  "variant": "experiment ID",
  "question": "question text",
  "gold_answers": ["normalized gold answer"],
  "evidence": ["optional evidence or answer fact"],
  "category": "question category",
  "document_ids": ["canonical document ID"],
  "metadata": {
    "original_record": {}
  }
}
```

Each line of `documents.jsonl` contains:

```json
{
  "schema_version": "1.0",
  "id": "unique physical document ID",
  "dataset": "dataset key",
  "source_id": "upstream logical document ID",
  "path": "corpus/relative/path.pdf",
  "media_type": "application/pdf",
  "size_bytes": 1234,
  "sha256": "...",
  "metadata": {
    "original_record": {}
  }
}
```

`metadata.original_record` preserves the complete upstream record. `gold_answers` is always non-empty. `evidence` or `document_ids` may be empty when the upstream dataset does not provide a resolvable mapping.

## Baseline branch workflow

Keep shared dataset logic, schemas, configurations, and prompts on `main`. Create a separate branch for each baseline:

```bash
git switch main
git switch -c baseline/<baseline-name>
```

The baseline branch should consume `qa.jsonl`, `documents.jsonl`, and `corpus/`, then add only the baseline-specific ingestion, retrieval, generation, evaluation, and logging integration.
