# OV-Wiki Baseline Benchmark

[English README](README.md)

本项目为 OV-Wiki benchmark 提供一套与具体 baseline 无关的数据准备公共层。它负责下载并校验六个上游数据集，准备 13 组固定实验，将不同数据集的语料与 QA 统一成标准格式，并保存与 OV-Wiki 实验对齐的回答和评测 prompt。

`main` 分支只维护公共的数据准备能力。接入某个 baseline 时，可以从 `main` 创建独立分支，在统一数据之上增加该 baseline 的入库、检索、回答生成、评测与日志逻辑。

## 一、常用命令

### 1. 准备运行环境

项目要求 Python 3.10 或更高版本。使用 `uv` 安装项目依赖：

```bash
uv sync
```

后续示例使用本项目注册的 `ov-wiki-data` 命令。

### 2. 查看全部实验配置

```bash
uv run ov-wiki-data list
```

该命令会列出 13 个固定的实验 ID，以及每组实验预期包含的 QA 和文档数量。

### 3. 下载并准备数据

准备一组实验：

```bash
uv run ov-wiki-data prepare enterprise_rag_bench_selected_80
```

一次准备多组实验：

```bash
uv run ov-wiki-data prepare \
  mudabench_simple \
  mudabench_complex
```

准备全部 13 组实验：

```bash
uv run ov-wiki-data prepare --all
```

将体积较大的下载文件和实验数据放到自定义目录：

```bash
uv run ov-wiki-data prepare mdaqa_first_100 \
  --data-dir /path/to/benchmark-data
```

如果不指定 `--data-dir`，默认写入项目根目录下的 `data/`。

### 4. 复用或重新下载原始数据

已有下载并校验通过的原始数据时，跳过下载，仅重新生成统一格式的数据：

```bash
uv run ov-wiki-data prepare mdaqa_first_100 --skip-download
```

强制重新获取上游文件，再进行数据准备：

```bash
uv run ov-wiki-data prepare mdaqa_first_100 --force-download
```

`--skip-download` 和 `--force-download` 的语义相反，不应同时使用。

PaperScope 的部分 PDF 可能要求临时登录 OpenReview。账号和密码从终端交互读取，不会写入磁盘。

### 5. 校验已准备的数据

校验一组实验：

```bash
uv run ov-wiki-data verify scholarqa_multi_valid_101
```

校验全部已经准备的实验：

```bash
uv run ov-wiki-data verify --all
```

校验内容包括：QA 和文档数量、统一字段、ID 与路径唯一性、QA 引用的文档 ID、文件大小以及 SHA-256 校验值。

### 6. 运行数据契约测试

```bash
uv run python -m unittest discover -s tests -v
```

### 7. 数据输出位置

每组实验都有独立的标准化输出目录。多组实验依赖相同上游语料时，会复用原始缓存。

```text
data/
├── raw/
│   ├── 共享下载缓存
│   └── 按上游数据结构整理并校验后的原始数据
└── prepared/
    └── <experiment_id>/
        ├── qa.jsonl
        ├── documents.jsonl
        ├── dataset_info.json
        └── corpus/
```

- `qa.jsonl`：统一格式的实验 QA。
- `documents.jsonl`：语料文件清单以及文档 ID、路径、哈希等信息。
- `corpus/`：实际用于 baseline 入库的 PDF 或 TXT 文件。
- `dataset_info.json`：实验名称、数据量及来源准备信息。

## 二、Prompt 对齐

机器可读的 prompt 统一保存在 [`prompts/`](prompts/) 中。各 baseline 分支应直接加载这些模板，不要在自己的代码中维护另一份副本，以免回答生成方式或 Accuracy 评测标准发生偏移。

本项目的 `main` 分支只准备数据和 prompt 契约；具体的模型调用与评测执行由各 baseline 分支实现。

### 1. 回答生成 Prompt

唯一机器模板：[`prompts/ov_wiki_bot_answer.txt`](prompts/ov_wiki_bot_answer.txt)

该模板与 OV-Wiki 当前 `vikingbot` 实验路径使用的指令保持一致。执行实验时，将 `{question}` 替换为 `qa.jsonl` 当前记录的 `question` 字段。

```text
Answer this question as briefly as possible. Use only the information available in the database. Do not use any external source.

Question: {question}
```

### Nashsu LLM Wiki baseline 分支

`nashsu_llm_wiki` 分支增加了面向 LLM Wiki `0.6.11`（固定 commit
`e8082119649e6a8e1cf85eaf289adcabfdf39d4e`）的严格三轮实验编排器。

编排器按标准化 corpus 哈希自动分组：共享同一 corpus 的变体只入库一次，
分别执行 QA/Judge，最后只删除一次。所有计入指标的 token 必须来自 provider
返回的真实 usage；任何缺失字段都会令实验失败，不使用字符数估算。LLM Wiki
fork 需要实现的接口见
[`patches/llm_wiki/bridge_contract.md`](patches/llm_wiki/bridge_contract.md)，实际实现位于
[`patches/llm_wiki/`](patches/llm_wiki/) 的四个顺序补丁中。`paths.project_path`
指向一个专用 LLM Wiki 项目。benchmark headless 模式下，隐藏 WebView 会自动初始化并打开
该项目；runner 在开始前等待鉴权 readiness 接口，不同 corpus 组之间会完整清理并复用项目。
入库使用项目外的批次快照：可重试网络错误会恢复并整批重跑，失败尝试与快照维护成本不进入
主要入库/删除指标，而是作为审计字段单独报告。

### 2. 通用 0–4 分 LLM Judge

13 组实验统一采用同一套 LLM Judge 契约。机器可读的唯一模板为 [`prompts/generic_llm_judge_user.txt`](prompts/generic_llm_judge_user.txt)。

```text
You are an expert evaluator scoring how well an AI-generated answer matches a gold standard (ground truth).

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

占位符的填充规则为：

- `{question}`：来自 `qa.jsonl` 的 `question`。
- `{gold_answers_joined_by_pipe}`：将 `gold_answers` 中的所有答案用 ` | ` 连接。
- `{generated_answer}`：baseline 最终生成的回答。

指标对齐规则：token-level F1 分别与 `gold_answers` 中的每个答案计算，取最大值；Accuracy 使用通用 Judge 返回的 0–4 整数分数。如果生成答案和至少一个 gold answer 都被判定为拒答或问题不可回答，则 F1 记为 `1.0`，Accuracy 记为 `4`。

## 三、支持的 13 组数据集实验

当前公共层覆盖六个数据集，共 13 份固定实验配置。

| 实验 ID | 数据集及范围 | QA 数 | 语料库 |
|---|---|---:|---:|
| `paperscope_summary_57_trend` | PaperScope Summary，`trend`，57 篇论文语料 | 117 | 57 个 PDF |
| `paperscope_summary_57_gap` | PaperScope Summary，`gap`，57 篇论文语料 | 119 | 57 个 PDF |
| `paperscope_summary_57_results_comparison` | PaperScope Summary，`results_comparison`，57 篇论文语料 | 116 | 57 个 PDF |
| `paperscope_summary_93_trend` | PaperScope Summary，`trend`，93 篇论文语料 | 117 | 93 个 PDF |
| `paperscope_summary_93_gap` | PaperScope Summary，`gap`，93 篇论文语料 | 119 | 93 个 PDF |
| `paperscope_summary_93_results_comparison` | PaperScope Summary，`results_comparison`，93 篇论文语料 | 116 | 93 个 PDF |
| `mdaqa_first_100` | MDA-QA 前 100 条 QA | 100 | 143 个 PDF |
| `wildgraphbench_summary_all` | WildGraphBench 全部主题的 Summary QA | 339 | 3,894 个 TXT |
| `wildgraphbench_summary_health` | WildGraphBench Health 主题 Summary QA | 55 | 509 个 TXT |
| `scholarqa_multi_valid_101` | ScholarQA-Multi 中引用编号有效的 QA | 101 | 413 个合并 TXT |
| `mudabench_simple` | MuDABench Simple QA，完整语料库 | 166 | 589 个 PDF |
| `mudabench_complex` | MuDABench Complex QA，完整语料库 | 166 | 589 个 PDF |
| `enterprise_rag_bench_selected_80` | Project Related、Conflicting Info、Completeness 三类 | 80 | 323 个 TXT |

这些实验 ID 由 [`configs/`](configs/) 中的 YAML 文件固定定义。每份配置声明数据集处理器、原始数据目录名、预期 QA/文档数量以及该实验特有的筛选参数。

### 各数据集的 Gold Answer 公共处理

- **PaperScope Summary**：将 `answer` 作为唯一 gold answer，并保留问题类型和支撑论文 ID。
- **MDA-QA**：将 `answer` 作为唯一 gold answer，将 `support` 映射为统一论文 ID。
- **WildGraphBench Summary**：将全部 `gold_statements` 合并为一个项目符号列表答案；原始 statements 和引用 URL 保存在 `metadata.original_record`。
- **ScholarQA-Multi**：保留原始专家答案，并在答案后附加从零开始的“引用编号 → 文献标题”映射；context 文本作为 evidence。
- **MuDABench**：将 `final_answer` 作为 gold answer，将 `source_answer` 作为 evidence；数据中两条完全重复的 QA 仍作为不同物理记录保留。
- **EnterpriseRAG-Bench**：将 `gold_answer` 作为 gold answer，将 `answer_facts` 作为 evidence；重复出现的逻辑文档会映射成不同的物理文档 ID。

## 四、代码模块作用

### 1. 整体执行流程

```text
configs/*.yaml
    ↓
specs.py 读取并校验实验定义
    ↓
cli.py 处理 list / prepare / verify 命令
    ↓
runner.py 将实验分发给相应数据集下载器
    ↓
datasets/<dataset>.py 下载并校验上游数据
    ↓
normalize.py 生成统一 QA、文档清单和语料库
    ↓
schema.py 校验最终实验数据
```

### 2. 顶层目录和文件

| 路径 | 作用 |
|---|---|
| `configs/` | 13 份声明式实验配置；实验数量和筛选范围不硬编码在 CLI 中。 |
| `prompts/` | 各 baseline 共同使用的回答生成与 LLM Judge 机器模板。 |
| `src/ov_wiki_baseline_benchmark/` | 可安装的 Python 包，包含完整公共数据准备流程。 |
| `tests/` | 校验 13 份配置和统一数据 schema 的契约测试。 |
| `pyproject.toml` | Python 版本、项目依赖、打包信息以及 `ov-wiki-data` 命令入口。 |
| `.gitignore` | 排除虚拟环境、缓存、生成的数据和本地编辑器文件。 |

### 3. 核心 Python 模块

| 模块 | 作用 |
|---|---|
| `__main__.py` | 支持执行 `python -m ov_wiki_baseline_benchmark`。 |
| `cli.py` | 解析 `list`、`prepare`、`verify`，校验实验选择并输出执行结果。 |
| `specs.py` | 将 YAML 配置加载为 `ExperimentSpec`，并校验固定 13 份配置的契约。 |
| `runner.py` | 将数据集 key 映射到下载器，管理 raw/prepared 路径，协调“下载 → 标准化 → 校验”。 |
| `normalize.py` | 将数据集原生结构转换为统一的 `qa.jsonl`、`documents.jsonl`、`dataset_info.json` 和 `corpus/`，同时完成 gold answer 处理。 |
| `schema.py` | 校验 schema 版本、必填字段、ID/路径唯一性、文档引用、文件大小和哈希。 |
| `io.py` | 提供 JSON/JSONL 读写、安全相对路径、SHA-256、硬链接/复制和目录原子替换等公共能力。 |

### 4. 数据集处理模块

| 模块 | 作用 |
|---|---|
| `datasets/paperscope_summary.py` | 按问题类型选择 PaperScope Summary QA，并准备共享的 57 或 93 篇 PDF 语料。 |
| `datasets/mdaqa.py` | 选择 MDA-QA 前 100 条记录，并从 arXiv 下载涉及的 143 篇 PDF。 |
| `datasets/wildgraphbench.py` | 从 reference-page TXT 文档中准备全主题或仅 Health 主题的 Summary 实验。 |
| `datasets/scholarqa_multi.py` | 排除引用编号越界的记录，将官方引用 contexts 合并为 413 个 TXT，并保留 101 条有效 QA。 |
| `datasets/mudabench.py` | 准备完整的 589 篇 PDF，并分别筛选 Simple 或 Complex QA。 |
| `datasets/enterprise_rag_bench.py` | 下载官方完整压缩包，但只提取三个目标类别所需的 323 个物理文档。 |

### 5. 统一数据契约

`qa.jsonl` 每行结构如下：

```json
{
  "schema_version": "1.0",
  "id": "唯一物理 QA ID",
  "dataset": "数据集 key",
  "variant": "实验 ID",
  "question": "问题文本",
  "gold_answers": ["标准化 gold answer"],
  "evidence": ["可选 evidence 或 answer fact"],
  "category": "问题类别",
  "document_ids": ["统一文档 ID"],
  "metadata": {
    "original_record": {}
  }
}
```

`documents.jsonl` 每行结构如下：

```json
{
  "schema_version": "1.0",
  "id": "唯一物理文档 ID",
  "dataset": "数据集 key",
  "source_id": "上游逻辑文档 ID",
  "path": "corpus/relative/path.pdf",
  "media_type": "application/pdf",
  "size_bytes": 1234,
  "sha256": "...",
  "metadata": {
    "original_record": {}
  }
}
```

`metadata.original_record` 完整保留上游记录。`gold_answers` 必须非空；如果上游数据没有提供可解析的证据或 QA—文档映射，`evidence` 或 `document_ids` 可以为空。

## Baseline 分支开发方式

公共下载、标准化、schema、配置和 prompt 始终维护在 `main`。每接入一个 baseline，从 `main` 创建独立分支：

```bash
git switch main
git switch -c baseline/<baseline-name>
```

baseline 分支读取 `qa.jsonl`、`documents.jsonl` 和 `corpus/`，只增加该 baseline 所需的入库、检索、回答生成、评测与日志逻辑。
