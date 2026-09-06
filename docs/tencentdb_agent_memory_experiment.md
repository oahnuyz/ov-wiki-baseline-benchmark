# TencentDB-Agent-Memory baseline 复现说明

本文档说明如何复现 `TencentDB-Agent-Memory` baseline。实验代码的来源是本仓库
`TencentDB-Agent-Memory` 分支；实验后端是 TencentDB/MemoryKnowledge 的
`feat/server_team` 分支在实验开始时固定的 commit。

## 1. 版本固定

| 组件 | 固定版本 |
| --- | --- |
| benchmark 仓库分支 | `TencentDB-Agent-Memory` |
| benchmark 代码 commit | 以本分支当前 HEAD 为准；首次复现实验前执行 `git rev-parse HEAD` 并记录 |
| TencentDB/MemoryKnowledge 分支 | `feat/server_team` |
| TencentDB/MemoryKnowledge commit | `3efcd317b84146d6a08518ac0f7ee7c8a8d200ec` |
| OpenViking PDF parser 参考 commit | `447b30ef8511dcc82c07ede857a52150479ee77c` |

获取 benchmark 代码：

```bash
git clone https://github.com/oahnuyz/ov-wiki-baseline-benchmark.git
cd ov-wiki-baseline-benchmark
git checkout TencentDB-Agent-Memory
git rev-parse HEAD
```

获取并固定 TencentDB/MemoryKnowledge：

```bash
git clone https://github.com/TencentCloud/tencentdb-agent-memory.git
cd tencentdb-agent-memory
git fetch origin feat/server_team
git checkout --detach 3efcd317b84146d6a08518ac0f7ee7c8a8d200ec
git rev-parse HEAD
```

不得使用未记录的浮动分支或最新代码替代上述 commit。

## 2. Python 环境

要求 Python 3.10 或更高版本。建议使用项目虚拟环境，不要向系统 Python 安装依赖：

```bash
cd ov-wiki-baseline-benchmark
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install pdfplumber
```

`pdfplumber` 是 TencentDB baseline 的外部 PDF 转 Markdown 转换器所需依赖；其余
Python 依赖由 `pyproject.toml` 声明。当前仓库没有提交 lockfile，因此复现实验时
应同时记录 `python --version`、`pip freeze`、操作系统和服务端运行时信息。

安装后可运行公共契约测试：

```bash
python -m unittest discover -s tests -v
```

## 3. MemoryKnowledge 服务

先按 TencentDB/MemoryKnowledge 上游项目的部署说明启动服务，并确认服务监听地址
与 `baseline_configs/tencentdb_agent_memory.yaml` 一致：

```text
http://127.0.0.1:8421/v3
```

配置中的 `service_id`、`team_id` 和 `user_id` 必须与服务端允许的身份一致。运行者
还应记录服务启动命令、Node.js 版本、依赖 lockfile 版本、服务日志路径和实际
commit。benchmark 通过以下接口调用服务：

```text
/wiki/create
/wiki/raw/write
/wiki/ingest
/wiki/get
/tools/list
/tools/call
/wiki/delete
```

## 4. 模型和密钥

配置文件已固定模型值，与 no-wiki baseline 对齐：

```yaml
model: doubao-seed-2-0-lite-260428
base_url: https://ark.cn-beijing.volces.com/api/v3
api_key_env: ARK_API_KEY
```

运行前在当前 shell 注入密钥；不要把密钥写入仓库、配置文件、日志或实验结果：

```bash
export ARK_API_KEY='<由运行者安全提供的 API key>'
```

如果服务端 ingest 使用独立的 LLM 凭据，也必须在服务启动环境中配置同一实验所需
的 provider/model，并将配置摘要（不含密钥）写入运行记录。

## 5. 数据集准备和指纹

公共准备数据位于 `data/prepared/<experiment_id>/`，也可以通过 `--data-dir` 指向
服务器上的共享数据目录。PaperScope 93 篇 PDF 的三个 QA 变体为：

```text
paperscope_summary_93_trend                 117 QA / 93 PDFs
paperscope_summary_93_gap                   119 QA / 93 PDFs
paperscope_summary_93_results_comparison   116 QA / 93 PDFs
```

若尚未准备数据：

```bash
python -m ov_wiki_baseline_benchmark.cli prepare \
  paperscope_summary_93_trend \
  paperscope_summary_93_gap \
  paperscope_summary_93_results_comparison \
  --data-dir /path/to/ov-wiki-benchmark-data
python -m ov_wiki_baseline_benchmark.cli verify --all \
  --data-dir /path/to/ov-wiki-benchmark-data
```

实际 CLI 参数以 `python -m ov_wiki_baseline_benchmark.cli --help` 为准；也可以直接
复用已经准备好的 `prepared` 目录。正式运行前应保存每个实验的
`dataset_info.json`、`documents.jsonl`、`qa.jsonl` 的 SHA-256，以及文档数量和 QA
数量。

## 6. TencentDB baseline 的入库定义

本 baseline 的正式定义是：

```text
PDF --外部 pdfplumber 转 Markdown--> raw/write
    --MemoryKnowledge wiki.ingest--> Wiki source/entity/concept 页面和 FTS5 索引
```

具体规则：

- PDF 文本和表格由外部 `pdfplumber` 转换；图片不单独上传，只保留文本占位符。
- 转换失败的 PDF 标记失败并继续处理其他文档。
- UTF-8 Markdown 超过 512 KiB 时按章节切分；该阈值来自配置默认值。
- PDF 转换时间计入入库时间；计时从首个原始文档开始转换起，到 `wiki.get` 返回
  `ready` 或 `failed` 为止。
- 同一 corpus 的多个 QA 变体只入库一次。入库完成后分别运行 QA 和 judge，最后
  只执行一次 delete。
- `wiki.ingest` 是异步批量任务启动 API；一次调用可能处理已写入的全部文档。
- QA 检索使用 TencentDB 默认工具和参数，底层为 SQLite FTS5 倒排索引，并用
  `bm25()` 排序；不配置 embedding。

## 7. 运行四个轮次

配置文件：`baseline_configs/tencentdb_agent_memory.yaml`。在仓库根目录执行：

```bash
# 入库轮：创建 Wiki、PDF 转换、raw/write、ingest、等待 ready
python -m ov_wiki_baseline_benchmark.tencentdb.cli \
  paperscope_summary_93_trend \
  paperscope_summary_93_gap \
  paperscope_summary_93_results_comparison \
  --data-dir /path/to/ov-wiki-benchmark-data \
  --config baseline_configs/tencentdb_agent_memory.yaml \
  --round ingest \
  --service-log /path/to/tencentdb-agent-memory-service.log

# QA 轮：每个 QA 使用独立、无历史上下文的完整 Agent loop
python -m ov_wiki_baseline_benchmark.tencentdb.cli \
  paperscope_summary_93_trend paperscope_summary_93_gap paperscope_summary_93_results_comparison \
  --data-dir /path/to/ov-wiki-benchmark-data \
  --config baseline_configs/tencentdb_agent_memory.yaml \
  --round qa \
  --service-log /path/to/tencentdb-agent-memory-service.log

# 评测轮：只对成功且非空的 QA 答案调用 judge；失败/空答案强制 0 分
python -m ov_wiki_baseline_benchmark.tencentdb.cli \
  paperscope_summary_93_trend paperscope_summary_93_gap paperscope_summary_93_results_comparison \
  --data-dir /path/to/ov-wiki-benchmark-data \
  --config baseline_configs/tencentdb_agent_memory.yaml \
  --round judge \
  --service-log /path/to/tencentdb-agent-memory-service.log

# 删除轮：删除完整 Wiki 资源和内存注册；保留结果、账本和报告
python -m ov_wiki_baseline_benchmark.tencentdb.cli \
  paperscope_summary_93_trend \
  --data-dir /path/to/ov-wiki-benchmark-data \
  --config baseline_configs/tencentdb_agent_memory.yaml \
  --round delete \
  --service-log /path/to/tencentdb-agent-memory-service.log
```

删除轮的计时截止于删除 API 返回，不包含实验结果文件、调用账本、报告或环境
恢复操作。除非明确要执行删除轮，否则不要运行 `--round delete`。

## 8. 异常、重试和恢复

- 单个 API 调用失败不会使已完成的其他文档或 QA 的指标失效；失败记录写入
  `calls.jsonl`、manifest 或对应 JSONL 结果文件。
- API 未返回合法 token usage 时，该调用 token 记为 0，并标记
  `token_usage_complete=false`；失败调用不计 token。
- 入库按服务日志识别未生成合法 Wiki 页的文档，最多使用配置允许的 3 次 ingest
  attempt。当前实验约定不做失败调用级别的单独补偿重试；失败文档可以在后续
  单独运行入库轮时重新处理。
- QA 每条记录独立保存；服务中断后可重新运行 QA/judge，已有成功结果会被复用。
- judge 缓存绑定 `answer_sha256`，因此新答案不会错误复用旧评分；失败或空答案
  不会提交给 judge LLM，直接记录 0 分。
- 不要删除已有 Wiki 或结果来“修复”单条失败，除非实验协议明确要求重新开始
  corpus；应保留调用账本、manifest 和失败原因。

## 9. 输出和指标

输出根目录默认为：

```text
Output/tencentdb/<corpus-id>/
```

主要文件包括：

- `manifest.json`：Wiki ID、版本、运行时、入库状态、文档状态和时间；
- `calls.jsonl`：服务/API/LLM 调用账本及 token 完整性标记；
- `<experiment>.qa.jsonl` 与 `.qa.summary.json`：逐 QA 答案、状态、耗时和 loop turns；
- `<experiment>.judge.jsonl` 与 `.judge.summary.json`：逐 QA judge 得分和归一化 accuracy；
- `<experiment>.benchmark_metrics_report.json`：最终汇总报告。

报告指标口径：

- 入库时间：外部 PDF 转换开始至 `wiki.get` 终态；累计重试耗时也会记录在累计值中；
- 入库 token：入库阶段所有成功且有合法 usage 的调用之和；MemoryKnowledge 异步
  内部调用若没有向外层 API 暴露 usage，只能标记为不完整/0，不能伪造精确值；
- QA 平均端到端时间：从发出问题到收到最终文本答案；
- QA 平均 token：问答阶段所有成功 LLM 调用 token 总和除以 QA 数；
- accuracy：judge 输出的 0–4 分及其除以 4 的 normalized accuracy；不记录 F1/Recall；
- 删除时间：删除 API 调用开始至返回；删除 token 按 0 记录。

正式报告应同时保留：代码 commit、TencentDB commit、配置文件、数据指纹、运行时
信息、服务日志路径、模型名称和结果目录。这样可以区分“代码可复现”和“同一模型/API
服务下的数值可重复”。

## 10. 当前已知限制

- 上游服务和模型 API 是外部依赖；API、模型版本、服务负载或限流变化可能导致耗时、
  token 和 accuracy 不完全相同。
- MemoryKnowledge 的异步 ingest API 不保证向 benchmark 返回内部抽取 LLM 的完整
  token usage，因此入库 token 可能只有外层可观测下界/0 标记。
- PDF 图片不上传，当前实验不进行多模态图片解析；表格按 Markdown 文本处理。
- QA 使用最多 15 个 Agent loop turns，不采用额外的 token 截断、摘要、分页或检索
  结果裁剪策略。
