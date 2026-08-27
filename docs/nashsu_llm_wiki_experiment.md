# Nashsu LLM Wiki baseline：系统原理、实验流程与指标定义

## 1. 文档范围与实现版本

本文描述 `ov-wiki-baseline-benchmark` 仓库 `nashsu_llm_wiki` 分支当前实现的实验流程，
以及该流程所调用的 Nashsu LLM Wiki 系统内部行为。这里的“Nashsu 系统”指
[`nashsu/llm_wiki`](https://github.com/nashsu/llm_wiki) 项目。

实验固定使用：

- LLM Wiki 版本：`0.6.11`
- 上游 commit：`e8082119649e6a8e1cf85eaf289adcabfdf39d4e`
- 补丁顺序：
  1. `patches/llm_wiki/0001-volcengine-thinking-and-dimension.patch`
  2. `patches/llm_wiki/0002-benchmark-bridge-and-telemetry.patch`
  3. `patches/llm_wiki/0003-webkit-request-timeout-fallback.patch`
  4. `patches/llm_wiki/0004-restart-safe-batched-ingest.patch`
- Python 实验入口：`ov-wiki-nashsu`
- 固定配置：`baseline_configs/nashsu_llm_wiki.yaml`

`0001` 负责 Volcengine 关闭深度思考的请求格式以及 1024 维 embedding 的强校验；
`0002` 增加只用于 benchmark 的本地控制接口、真实 token telemetry、完整入库等待、
独立 QA 和定向清理能力；`0003` 为旧 WebKitGTK 补齐单次 provider 请求 timeout；`0004`
增加分批 drain、受校验的跨重启续批和跨 run staging 清理。补丁不替换 Nashsu 的知识生成、
Agent 或检索算法。

## 2. 两类 token：bridge token 与 Ark API key

### 2.1 `LLM_WIKI_API_TOKEN`（bridge token）

bridge token 是 **LLM Wiki 本地 HTTP API 的访问令牌**。它用于鉴权，不用于模型推理：

- runner 请求 `http://127.0.0.1:19828/api/v1/benchmark/...` 时发送
  `Authorization: Bearer <token>`；
- LLM Wiki 使用相同的 `LLM_WIKI_API_TOKEN` 校验请求；
- 它防止本机其他进程未经许可触发付费入库、QA 或删除；
- benchmark 补丁还把这些接口限制为 loopback，只允许本机访问；
- 它不能调用 Volcengine Ark，不应与 Ark API key 混用；
- 它必须是随机秘密，不能提交到 Git、结果文件或日志。

示例生成方式如下。生成出的同一个值必须同时提供给 LLM Wiki 进程和 Python runner：

```bash
openssl rand -hex 32
export LLM_WIKI_API_TOKEN='<上一步输出的随机值>'
```

### 2.2 `ARK_API_KEY`

`ARK_API_KEY` 是 Volcengine Ark 的模型服务凭据，作用与 bridge token 完全不同：

- Nashsu 的入库、图片 caption、Agent QA 使用 Ark key；
- Python Judge 也使用 Ark key；
- 当前固定模型的 LLM、caption、QA 和 Judge 均走
  `https://ark.cn-beijing.volces.com/api/v3/chat/completions`；
- embedding 走同一 Ark base 下的多模态 embedding 接口；
- key 只应存在于环境变量或 LLM Wiki 的本机设置中，不能进入仓库。

## 3. 系统与实验的整体关系

Nashsu LLM Wiki 不是“把原文直接切块后回答”的单层 RAG。它先把原始 corpus 转换为一个
带来源、页面、交叉链接、review 项和向量索引的 Wiki 知识库；QA 时，Rust Agent 再通过
Wiki 混合检索工具寻找证据并生成回答。

```text
prepared 标准数据
  ├─ documents.jsonl ──> corpus/PDF 或 TXT
  └─ qa.jsonl ─────────> question + gold answers
             │
             ▼
      [入库轮：每个共享 corpus 一次]
  内置解析 → 图片提取/caption → 两阶段 LLM 知识生成
  → Wiki 页面/链接/review → chunk embedding/LanceDB → review sweep
             │
             ▼
      [QA 轮：每个实验独立执行]
  Standard Agent → wiki.search / wiki.read_page
  → 关键词 + 向量 + 一跳图扩展 → LLM 最终回答
             │
             ▼
      [评测轮：每个回答独立执行]
  question + gold answers + generated answer → 0–4 Judge
             │
             ▼
      [删除轮：每个共享 corpus 一次]
  清除本轮 Wiki、解析产物、review、caption cache、向量与索引
```

## 4. prepared 实验输入

每个实验目录为：

```text
prepared/<experiment_id>/
├── qa.jsonl
├── documents.jsonl
├── dataset_info.json
└── corpus/
    └── 实际 PDF 或 TXT 文件
```

### 4.1 `documents.jsonl`

每行是一个标准文档记录。runner 会校验：

| 字段 | 含义与约束 |
|---|---|
| `schema_version` | 固定为 `1.0` |
| `id` | 实验内唯一文档 ID，非空 |
| `source_id` | 上游数据中的来源 ID，非空 |
| `path` | 相对实验目录的安全路径，通常指向 `corpus/` |
| `media_type` | 仅接受 `application/pdf` 或 `text/plain` |
| `size_bytes` | 必须与实际文件大小一致 |
| `sha256` | 必须与实际文件 SHA-256 一致 |
| `metadata` | 数据集特有元数据对象 |

`id` 和 `path` 都必须唯一；文件不存在、大小不符或哈希不符时，实验在付费调用前失败。

### 4.2 `qa.jsonl`

每行是一个标准 QA 记录：

| 字段 | 含义与约束 |
|---|---|
| `schema_version` | 固定为 `1.0` |
| `id` | 实验内唯一 QA ID |
| `question` | 非空问题文本 |
| `gold_answers` | 一个或多个非空参考答案 |
| `evidence` | 标准化证据文本列表，可以为空 |
| `document_ids` | 相关文档 ID 列表；每个 ID 必须存在于 `documents.jsonl` |
| `metadata` | 必须是对象，并保留 `original_record` |

当前 Accuracy Judge 使用 `question`、`gold_answers` 和模型生成的 answer；`evidence` 与
`document_ids` 会写入逐题结果，但不直接进入 Judge prompt。

### 4.3 共享 corpus 分组

runner 将一个实验全部文档的 `sha256` 排序后串联，再计算 corpus fingerprint。fingerprint
相同的实验被视为共享 corpus：

- 只执行一次入库；
- 每个实验分别执行自己的 QA 和 Judge；
- 全部变体完成后只执行一次删除。

这用于 PaperScope、MuDA 等共享文档但 QA 变体不同的实验。fingerprint 当前只基于文档内容
哈希，不基于文件路径或元数据。

## 5. 固定模块与参数

### 5.1 已固定参数

| 模块 | 可选项 | 本实验选择 | 说明 |
|---|---|---|---|
| Agent mode | `fast` / `standard` / `deep` / `local_first` | `standard` | 使用 Nashsu 官方 Standard Agent loop |
| retrieval mode | `standard` / `smart` / `faithful` | `standard` | 不启用 Smart 的证据缺口策略，也不限制为 raw-source-only |
| Wiki 工具 | 开/关 | 开 | 允许 `wiki.search`、`wiki.read_page` |
| Web search | 开/关 | 关 | 防止外部知识源进入回答 |
| AnyTXT | 开/关 | 关 | 防止检索项目外文本 |
| Skills | 自动/显式/关闭 | 关闭 | `skills=[]`，避免技能改变工具选择和 prompt |
| `top_k` | 1–10 | 不传参数，解析为官方默认 `5` | bridge 同时校验返回值为 5 |
| `maxContextSize` | 用户可配置 | 不覆盖官方默认，要求当前值为 `204800` 字符 | 不等同于 204800 tokens |
| 项目模板 | Research / Reading / Personal / Business / General | 未经编辑的官方 `General` | 每个 corpus 入库计时前恢复并记录文件 SHA-256 |
| 输出语言 | `auto` 或固定语言 | 官方 `auto` | 按源内容自动选择页面输出语言 |
| 主 LLM | 多 provider/model | `doubao-seed-2-0-lite-260428` | provider 记为 Volcengine，LLM Wiki 内部使用 custom OpenAI-compatible wire |
| temperature | 模型采样参数 | `0` | benchmark telemetry 激活时强制覆盖入库与 QA 调用 |
| thinking | 开/关/不同预算 | `disabled` | 向 Ark 发送 `thinking: {"type":"disabled"}` |
| streaming | 开/关 | 关 | 非流式响应才能稳定取得完整 provider usage |
| ingest worker | 1–5 | `1` | corpus 文档串行处理，避免上下文和计时相互污染 |
| 入库批大小 | 正整数 | `25` 篇 | 每批保持同一知识库，批间重启 WebKit 服务以释放进程 mappings |
| 批间服务重启 | 开/关 | 开 | 非最终批完成后自动重启；最终批完成后不重启，直接进入 QA |
| 网络失败批次重试 | 非负整数 | 最多 `2` 次 | Nashsu 内部重试耗尽后，恢复批前快照并整批重跑 |
| 批次快照目录 | 项目外路径 | `~/nashsu-llm-wiki-baseline/snapshots` | 不参与检索，正常/失败/中止时清理 |
| QA worker | 可并行 | `1` | 逐题串行 |
| Judge worker | 可并行 | `1` | 逐题串行，与参考 baseline 对齐 |
| PDF 解析 | 内置 / MinerU Cloud / MinerU Local | 内置 | MinerU 强制关闭 |
| 图片 caption | 开/关、主模型/独立模型 | 开，复用主 LLM | caption 并发数固定为 4 |
| 向量检索 | 开/关 | 开 | 页面入库后自动建立 LanceDB chunk 索引 |
| embedding | 多 provider/model | `doubao-embedding-vision-251215` | multimodal input，返回维度必须为 1024 |
| embedding 并发 | 可配置 | `1` | 便于顺序 telemetry 和稳定限流 |
| embedding batch size | 可配置 | `1` | Doubao vision embedding 本身不走 OpenAI 文本 batch |
| chunk 参数 | 项目覆盖 / 官方默认 | 官方默认 | 清除 `maxChunkChars` / `overlapChunkChars` 项目覆盖 |
| parsed Markdown 副本 | 开/关 | 关 | 固定 `persistExtractedMarkdown=false` |
| headless 启动 | 可见桌面 / 隐藏 WebView | 隐藏 WebView | Xvfb 中保留官方前端入库链路，不需要人工点击 |
| startup timeout | 可配置 | `300` 秒 | runner 等待 bridge 和专用项目就绪；不计入实验时间 |
| bridge timeout | 可配置 | `129600` 秒（36 小时） | 对单批入库或单次 QA/Judge HTTP 请求的上限，不是整个 corpus 的总时限 |

### 5.2 Agent mode 与 retrieval mode 是两个独立参数

`mode` 控制 Agent loop 的总决策预算；`retrievalMode` 控制检索策略。此前用户所说的
“使用 standard”在本实验中明确落实为两个相互独立且都采用官方默认值的选择：

```text
mode = standard
retrievalMode = standard
```

Agent mode 的可选值为：

| Agent mode | 无 skills 时的最多 Agent 轮次 | `retrievalMode=standard` 时的最多检索动作 | 主要区别 |
|---|---:|---:|---|
| `fast` | 4 | 2 | 较小的工具与回答预算 |
| `standard` | 8 | 4 | 官方默认 Agent loop |
| `deep` | 12 | 8 | 更大预算，并更积极纳入原始来源或已开启的外部工具 |
| `local_first` | 8 | 4 | 固定版本中预算与 Standard 相同；当前源码未显示另一套实质性检索算法 |

Retrieval mode 的可选值为：

| Retrieval mode | `mode=standard` 时的最多检索动作 | 行为 |
|---|---:|---|
| `standard` | 4 | 模型在预算内自行决定搜什么、读什么和何时回答 |
| `smart` | 4 | 每次观察后识别尚缺证据，修改查询，抑制近似重复；连续两次没有新增证据时强制结束检索 |
| `faithful` | 3 | 只以 raw source excerpt 或显式附件为证据，排除生成 Wiki、overview/schema、Web 和 AnyTXT |

因此 Smart 与 Standard 的差异不是 `top_k` 或本实验组合下的检索次数上限。Smart 给同一组
检索工具增加“证据缺口循环”、近似查询归一化和无进展提前停止；Standard 只阻止参数完全
相同的重复动作，把检索路径交给模型自由规划。本实验保持官方默认 Standard，不启用 Smart。

### 5.3 Nashsu Standard Agent loop 的具体含义

在固定版本中，只要后端 LLM 可用，Standard 模式进入 **模型驱动的 Agent loop**，而不是
固定执行一次检索再回答：

- 最多 8 次 Agent 迭代；
- `retrieval_mode=standard` 且没有 skills 时，最多 4 个检索工具动作；
- 每一轮 LLM 输出紧凑 JSON，选择一个工具动作或直接输出 final answer；
- 可用的内部检索工具主要是 `wiki.search` 与 `wiki.read_page`；
- 工具结果作为 observation 回到下一轮；
- 达到检索预算后进入强制 final 阶段，不再允许新工具调用；
- `topK` 未指定时，每次搜索默认取 5，允许范围为 1–10。

因此，Standard 表示“官方标准 Agent 行为和预算”，不表示“一定恰好检索一次”。保守 router
只识别明显的 web/raw/graph/write 意图，普通问题通常交给模型自己判断是否调用
`wiki.search`。

## 6. 实验前置状态

1. `paths.project_path` 必须指向专用项目；headless 模式会自动初始化空目录并打开它。
2. 该项目不能与日常个人 Wiki 混用，因为删除轮会删除项目的 Wiki 和 `.llm-wiki` 运行数据。
3. 同一专用项目在不同 corpus 组之间复用；每组结束后先完整删除本组数据。
4. runner 校验 bridge 返回的 `maxContextSize=204800`、`top_k=5` 和 embedding 维度 1024。
5. 环境中必须提供 `LLM_WIKI_API_TOKEN`、`ARK_API_KEY`、
   `LLM_WIKI_BENCHMARK_HEADLESS=1` 和精确的 `LLM_WIKI_BENCHMARK_PROJECT_PATH`。
6. 真实实验前必须确认专用项目没有旧的 `wiki/`、向量、review 或 staged corpus。

服务器不需要可见桌面，也不需要人工操作 LLM Wiki 窗口。Tauri 仍在 Xvfb 中创建隐藏 WebView，
因为官方入库队列、两阶段页面生成和 review sweep 位于 TypeScript 前端。Rust API 与前端执行
双就绪握手：只有 bridge listener 已注册且环境变量指定的项目已打开时，鉴权的
`GET /api/v1/benchmark/ready` 才返回 200；否则返回 503。runner 最多等待 300 秒。Xvfb
启动、WebView 初始化、空 General 项目创建和打开均发生在任何实验计时之前。

若项目目录为空，headless bootstrap 只写入官方 General 脚手架后打开项目；若目录非空但
不是有效 LLM Wiki 项目，则直接失败，不猜测、不覆盖。Ark key 和 bridge token 不出现在
readiness 响应、manifest 或日志中。

本实验采用 **未经人工编辑的官方 General 项目模板**，不添加任何与数据集、问题或答案有关
的先验信息。General 模板只提供 Nashsu 开箱即用的通用项目脚手架。每个新 corpus 从字节
一致的脚手架开始；bridge 计算五个脚手架文件的 SHA-256，runner 将结果写入 group run
manifest。

四个文件分为“固定项目配置”和“运行中知识状态”两类：

| 文件 | 创建时来源 | 入库时用途 | 是否随 corpus 自动变化 | QA 时是否直接加入项目上下文 |
|---|---|---|---|---|
| `purpose.md` | General 模板 | 告诉分析/生成阶段项目目标和关注范围 | 否 | 否 |
| `schema.md` | General 模板 | 规定页面类型、目录、frontmatter 与链接结构 | 否 | 是 |
| `wiki/index.md` | 创建项目时生成空分类框架 | 告知后续文档已有页面，帮助连接、合并和减少重复 | 是，每份文档写完后程序确定性追加 | 否 |
| `wiki/overview.md` | 创建项目时生成空 overview 占位 | 作为高层 Wiki 上下文传给生成阶段 | 当前自动入库不会更新 | 是 |

`purpose.md` 的 General 默认内容只有空白目标、问题、范围和 `TBD`；`schema.md` 是通用组织
规则而不是 corpus 事实。保留二者表示测试 Nashsu 官方默认系统，而不是人为给系统提前输入
领域知识。`index.md` 初始没有知识条目，随后完全由已入库 corpus 产生。`overview.md` 虽由
项目创建器生成占位文件，但固定版本的入库 prompt 明确禁止模型生成它，当前入库调用链也
没有自动汇总更新逻辑；未经人工编辑时它保持空白占位，不构成数据集先验。

## 7. 第一轮：入库轮

### 7.1 计时前：校验与 staging

runner 先校验 prepared 数据和共享 corpus fingerprint。bridge 创建 run 时验证：

- readiness 已确认当前打开项目与 `project_path` 完全一致；
- 固定版本、模型、检索、caption、PDF 和 embedding manifest 完全一致；
- 项目没有被另一个 benchmark run 占用。

每个 corpus 文件按 prepared 清单顺序分批复制到：

```text
<project>/raw/sources/.benchmark-<run_id>/
```

**文件复制不计入入库时间。** 每一批 staging 完成后才启动该批计时和 token telemetry。
默认每批 25 篇、并发仍为 1。非最终批只等待本批队列完全 drain，不运行 review sweep；
随后 runner 重启整个 headless 服务、等待 readiness，并以显式 `continuation=true` 创建同一
corpus 的续批 run。续批接口会校验 `purpose.md`、`schema.md` 和 `overview.md` 仍是未经人工
编辑的 General 默认文件，并保留前批生成的 Wiki 页面、索引、向量库、review 和 ingest cache。
因此这是进程资源的分段释放，不是知识库清空或断点跳过。

每批开始前，runner 在项目目录之外创建完整项目快照。如果 Nashsu 内部最多 3 次任务重试
仍因网络发送错误、连接中断或 timeout 耗尽，runner 会停止整个服务、恢复到该批开始前的
快照、重启服务并整批重跑，最多额外重跑 2 次。配置、解析、维度、telemetry 等非网络错误
不自动重跑。恢复整个批次可保证失败尝试留下的 caption cache、Wiki 文件、review、索引或
LanceDB 写入不会被下一次尝试复用。

### 7.2 串行队列与解析

所有文档进入 Nashsu 的持久化 ingest queue，worker 固定为 1。每个任务执行：

1. 根据文件类型读取源内容；
2. PDF 使用内置 Rust 解析链路，不调用 MinerU；
3. 同时读取 `schema.md`、`purpose.md`、`wiki/index.md`、`wiki/overview.md`；
4. 计算 SHA-256 ingest cache。全新专用项目应为 cache miss；未变文档在非 benchmark 重入时
   可以跳过主要知识生成，但仍可能重新执行图片链路。

PDF 解析发生在入库计时区间内。可见的 parsed Markdown 副本是否写入 `raw/parsed` 由项目的
source-watch 设置决定，但真正的解析工作无论是否保留可见副本都会发生。

### 7.3 图片提取与 caption

Nashsu 从 PDF、PPTX、DOCX 等源中抽取内嵌图片，保存到：

```text
<project>/wiki/media/<source-slug>/
```

本实验开启 multimodal caption，并复用主 LLM：

1. 读取图片字节并发送给 `doubao-seed-2-0-lite-260428`；
2. 生成事实性 caption；
3. 将空 alt 文本 `![](path)` 改写为 `![caption](path)`；
4. 把带语义的图片引用注入 source summary；
5. 后续 Wiki 页面 embedding 会包含 caption 文本，因此图表内容可以参与向量检索。

caption cache 以“图片字节 SHA-256 + 输出语言”为 key，存放在
`.llm-wiki/image-caption-cache.json`，相同图片可跨文档复用 caption。caption 调用、cache I/O
和图片处理都在入库计时内；staging 文件复制不在其中。

### 7.4 两阶段知识生成

Nashsu 对每个源执行核心两阶段 LLM 入库。Markdown 页面不是等所有源完成后统一生成：在
worker=1 下，每份文档依次完成“解析 → caption → analysis → generation → 写页/合并 →
更新 index → embedding”，然后才处理下一份文档。因此后面的文档能够读取前面文档已经
生成的页面和更新后的 index；这一点跨服务重启和跨批次保持不变。全部文档都完成后才运行
一次 review sweep。

#### 阶段 1：Analysis

LLM 阅读源内容以及 purpose/schema/index 等上下文，生成结构化分析，内容包括：

- 关键实体、概念和论点；
- 与现有 Wiki 的连接；
- 潜在矛盾或张力；
- Wiki 页面组织建议。

若源内容超过基于 `maxContextSize` 和稳定项目上下文计算的 source budget，系统先分块分析，
再汇总成长文档 analysis，并使用 checkpoint 支持恢复。

#### 阶段 2：Generation

LLM 同时看到阶段 1 analysis 与源上下文，输出受约束的 `FILE`/`REVIEW` blocks：

- `wiki/sources/<source-slug>.md` 来源摘要；
- entity、concept 等知识页面；
- YAML frontmatter，包括 `type`、`title`、`sources[]`；
- `[[wikilinks]]` 交叉引用；
- `wiki/log.md` 的新日志条目；
- 可能需要人工处理的 review 项。

Generation prompt 明确禁止模型生成 `wiki/index.md` 和 `wiki/overview.md`。`index.md` 由程序
根据本次实际写入的内容页面确定性更新；`overview.md` 在当前固定版本中不会随自动入库更新。

系统对文件路径做项目边界检查，并在写入时处理合并、截断和结构错误：

- 截断的 FILE block 会触发一次完整 repair 生成；
- `wiki/index.md` 还会执行确定性更新；
- 若模型漏掉 `wiki/log.md`，系统写入确定性 log entry；
- 若模型漏掉 source summary，系统生成 fallback summary；
- 未修复的截断或硬写入失败使该 ingest task 失败，不会标记为成功 cache。

必要时系统还执行独立的 review suggestion LLM 阶段，为真正的知识缺口生成高价值 review。

### 7.5 页面分块与向量入库

启用 embedding 后，每个成功写入的可检索 Wiki 页面会执行：

1. 去除 YAML frontmatter；
2. 按 Markdown heading、段落、行、句子、空白递归切分；
3. fenced code block 和单张 Markdown table 不被切断；
4. 给每个 chunk 带上 heading breadcrumb；
5. embedding 输入为 `page title + heading path + chunk text`；
6. 调用 `doubao-embedding-vision-251215`；
7. 强制校验每个返回向量恰好 1024 维；
8. 将 chunk text、heading path 和向量写入项目内嵌 LanceDB。

官方未覆盖时的 chunker 默认值为：目标约 1000 字符、硬上限 1500、最小 200、相邻 overlap
200。代码允许项目设置改变目标长度和 overlap；当前 bridge 固定 embedding concurrency=1、
batch size=1，但尚未显式清空已有项目的 chunk 长度覆盖值。

`index.md`、`log.md`、`overview.md` 作为聚合页不作为普通页面 embedding 目标。

### 7.6 Review sweep

只有最终批的所有 corpus 任务结束、队列 drain 后，入库轮才执行唯一一次 review sweep。
中间批次不会执行 sweep。sweep 分两级：

1. **规则级清理**：例如 missing-page 对应页面已存在，或 duplicate 涉及页面已变化；
2. **LLM 语义判断**：对剩余 review 保守判断是否已被新 Wiki 内容解决。

LLM sweep 每批最多 40 个 review，最多 5 批；prompt 最多列出 300 个 Wiki 页面。如果一批
没有解决任何 review，会提前停止后续批次。

**入库结束时间点是最终 review sweep 完成并收集完 provider usage 之后。** 主指标
`Total Insertion Time` 是各批 active duration 之和，包含解析、图片提取/caption、知识生成、
文件写入、页面 embedding 和最终完整 review sweep，但不包含 staging 与批间服务重启。
报告另列 `Operational Wall Clock Time Including Restarts`，表示从第一批开始到最终批完成的
实际墙钟时间，包含批间重启与 readiness 等待。这样既保持原 baseline 的入库计时边界，又
透明记录为规避 WebKit mappings 上限而付出的运行开销。

若某批发生自动回滚，主指标只累加该批最终成功尝试的 `durationSeconds` 和真实 provider
usage；被回滚尝试的时间和 token 不进入 `Total Insertion Time` 或
`Total Insertion Token Cost`。其墙钟耗时、错误、run ID、usage 是否可得会写入
`discarded_ingest_attempts`。网络错误没有 provider 响应时，失败尝试 token 记为
`null / usage_complete=false`，不估算为 0。快照创建、恢复和清理时间也只进入审计字段。

### 7.7 入库 token

入库 usage 累加计时区间内所有 provider 返回的：

- LLM input tokens：analysis、generation、caption、repair、review suggestion、review sweep；
- LLM output tokens：上述调用的 completion tokens；
- embedding tokens：所有页面 chunk embedding 调用。

不使用字符数估算。任何已发出的模型/embedding 请求如果没有可解析的 usage，整个入库请求
标记为 telemetry incomplete，并使 benchmark 失败。Nashsu 原生 ingest queue 仍保留最多
3 次任务重试和 embedding oversize 缩半重试；这些内部重试属于一次批次尝试，若最终成功，
其时间和可报告 usage 均计入该成功尝试。如果内部重试最终因可重试网络错误耗尽，则由外层
快照机制回滚整批，并从主要指标中排除整个失败批次尝试。

## 8. 第二轮：QA 问答轮

### 8.1 QA 隔离与 prompt

每道题严格渲染以下 prompt，不追加 baseline 私有指令：

```text
Answer this question as briefly as possible. Use only the information available in the database. Do not use any external source.

Question: {question}
```

每个 QA 使用新的随机 `sessionId`，并发送：

- `history=[]`
- `historyExplicit=true`
- `persistSession=false`
- `skills=[]`
- `tools={wiki: true, web: false, anytxt: false}`
- `mode=standard`
- `retrievalMode=standard`
- 不发送 `topK`

因此不会读取上一题对话，也不会把本题写入持久化聊天历史。所有题串行运行。

### 8.2 Standard Agent 如何决定检索

Agent 先构造项目上下文和工具说明，然后由同一主 LLM 在最多 8 次循环内选择动作：

- `wiki.search(query, topK=5)`：对生成 Wiki 做混合检索；
- `wiki.read_page(path)`：读取已找到的完整 Wiki 页面；
- `final(answer)`：结束并回答。

Web、AnyTXT 和 skills 不可用，写 Wiki 也不是本轮授权行为。Standard retrieval budget 为 4；
达到预算后系统强制进入 final-only prompt。

“最多 8 次循环”统计所有模型决策，不只是检索。一次循环只能选择一个工具动作或 final。
例如 `wiki.search → wiki.read_page → final` 是 3 轮、2 个检索动作。检索预算统计
`wiki.search`、`wiki.read_page`、`source.search`、`graph.search`、`web.search` 和
`anytxt.search`；`top_k=5` 只表示一次 `wiki.search` 最多返回 5 个结果，不表示最多搜索
5 次或最多读取 5 页。

Agent 理论上还支持 `user.ask`、`wiki.write_page`、`workspace.write_file`、
`workspace.append_file`、`skill.read_file` 和 `shell.exec` 等非检索动作，格式错误后的动作修复
也可能消耗一轮。当前 QA 的 skills 为空，Web/AnyTXT 关闭，用户 prompt 只要求回答而不要求
写入，因此正常轨迹应为若干检索动作后 final，或模型直接 final。按照保留 Nashsu 官方
Agent 行为的决定，benchmark **不增加额外的动作 allowlist 或硬拦截**；实际 trace 会进入
逐题结果，异常动作应在 smoke test 和结果审计中可见。

需要特别说明：固定版本的 Standard Agent **允许模型直接选择 final**。当前 bridge 检查会话
隔离、模式、工具开关和 `top_k`，但尚未要求 trace 中必须出现一次成功的 `wiki.search`。
因此 QA prompt 虽要求只用 database，代码层目前没有对“零检索直接回答”执行 fail-closed。

### 8.3 `wiki.search` 的混合检索

当 Agent 调用 `wiki.search` 时，Nashsu 执行以下流程。

#### A. 关键词检索

- 扫描 `wiki/` 下最多 10,000 个 Markdown 文件；
- query 小写化、移除常见英文停用词，并进行英文 token/CJK 处理；
- 文件名完全匹配加 200 分；
- title 包含完整 query phrase 加 50 分；
- content 中每次 phrase 出现加 20 分，最多计算 10 次；
- title token 命中权重为 5，content token 命中权重为 1。

#### B. 向量检索

- 先对 Agent 生成的 search query 调用同一个 Doubao embedding 模型；
- 强制校验 query vector 为 1024 维；
- LanceDB 先取 `max(topK × 3, 30)` 个 chunk；
- 按 page ID 聚合 chunk；
- 页面向量分数为最高 chunk 分数，加上其余 chunk 分数和的 0.3 倍，但 tail contribution
  不允许把总分推过 1；
- 向量结果即使没有关键词命中，也可以新增候选页面。

#### C. 关键词与向量融合

两路排序使用 Reciprocal Rank Fusion：

```text
RRF(page) = Σ 1 / (60 + rank_in_channel)
```

即关键词和向量各贡献一个排名项，`RRF_K=60`。最终结果保留原始 vector score 供 trace
和诊断使用。

#### D. 一跳 Wiki 图扩展

系统从初始排名的前若干页面提取 `[[wikilinks]]`，构造双向邻接关系，再加入一跳邻居：

- 最多使用 20 个 seed；
- 图候选分数按 seed rank 的 `1/(rank+1)` 累加；
- 最终 top-K 中给图邻居预留约 15%–30% 配额；
- 向量覆盖越充分，图配额越接近 15%；向量越稀疏，越接近 30%；
- 若没有可用图候选，配额自动归还给关键词/向量结果。

由于本实验 `topK=5`，一次 `wiki.search` 最终最多向 Agent 返回 5 个结果。结果包含 title、
path、snippet、融合 score、可选 vector score、图片引用和图关系信息。Agent 可以继续通过
`wiki.read_page` 读取页面全文。

### 8.4 上下文与最终回答

Agent prompt 会组合：

- 用户的完整 QA prompt；
- schema/overview 等项目上下文；
- 显式空历史；
- 已检索 references 与 tool observations；
- Standard Agent 协议和引用要求。

上下文通过 `maxContextSize=204800` **字符**预算裁剪。模型最后输出 JSON
`{"action":"final","answer":"..."}`，bridge 提取 answer。主模型、temperature、thinking 和
非流式设置与入库轮一致。

### 8.5 QA 时间与 token

单题 `durationSeconds` 从 Rust Agent 调用前开始，到最终 answer 和 provider usage 收集完成后
结束，包含：

- Agent 规划/循环的所有 LLM 调用；
- query embedding；
- 关键词、LanceDB 和图检索；
- `wiki.read_page` 等工具调用；
- 最终回答生成。

因此报告字段虽然为 `Average Retrieval Time`，实际含义是 **平均端到端 QA 回答时间**，
不是纯搜索函数耗时。

QA token 分为：

- `agentInputTokens` / `agentOutputTokens`：Agent loop 和 final 的真实 LLM usage；
- `searchInputTokens` / `searchOutputTokens`：为兼容参考 baseline 保留的独立搜索 LLM 槽位；
  Nashsu 当前混合检索没有独立 search LLM，因此这两个值为 0；
- `embeddingTokens`：检索 query embedding usage。

## 9. 第三轮：评测轮

### 9.1 Judge 输入与固定参数

每个完成的 QA 独立调用同一个 `doubao-seed-2-0-lite-260428`：

- provider/base 与入库、QA 相同；
- `temperature=0`；
- `thinking={"type":"disabled"}`；
- `stream=false`；
- 输入为 `question`、用 ` | ` 拼接的全部 gold answers、generated answer；
- 要求只返回 `{"score": 0..4, "reasoning": "..."}`。

0–4 含义为：4 完全正确，3 基本正确但有轻微问题，2 部分正确，1 大部分错误，0 错误、
矛盾或核心幻觉。generated answer 只需匹配多个 gold answers 中任意一个；不矛盾的额外相关
信息不扣分。

### 9.2 Judge 异常处理

该实现与参考 baseline 对齐：

1. 首先把完整响应按 JSON 解析并读取 `score`；
2. 失败时，从原始文本匹配 `"score": [0-4]`；
3. 再失败时，匹配第一个独立的 0–4 整数；
4. 调用失败或仍不能解析时，score 记为 0；
5. 所有 score 最终 clamp 到 `[0,4]`。

如果 generated answer 和任一 gold answer 都被拒答启发式识别为不可回答，则覆盖为：

- F1 = 1；
- Accuracy score = 4；
- prompt type = `Heuristic_Refusal_Check`。

拒答短语包括 `not mentioned`、`no information`、`cannot be answered`、`none`、
`unknown`、`don't know`。

Judge 的 provider token 单独写入 `judge_telemetry.json`。它不计入“平均问答 token 成本”；
如果 Judge usage 缺失，当前实现把 `usage_complete` 记为 false，但仍保留该题 score。

## 10. 删除轮

一个共享 corpus 的所有实验变体完成 QA 与 Judge 后，只执行一次删除。计时从删除操作开始，
覆盖前端内存状态清理和文件系统清理。

删除内容包括：

- `wiki/` 生成页面；
- `media/`；
- `raw/parsed/`；
- 本 run 的 `raw/sources/.benchmark-<run_id>/` staged corpus；
- `.llm-wiki/` 下的 review、caption cache、ingest cache、向量/索引和其他运行状态。

系统在删除前验证项目 ID 没有变化、项目路径为绝对安全路径，并且只操作 run 注册的专用
项目。删除 `.llm-wiki/` 后恢复原项目的 `project.json`，以便下一 corpus 组复用同一项目。
`purpose.md` 和 `schema.md` 不删除。

删除整个 `wiki/` 也会删除初始 `index.md`、`overview.md` 和 `log.md`。为了保证第一个 corpus
与后续 corpus 初始状态一致，下一 corpus 创建 run 时必须在入库计时开始前恢复官方 General
脚手架；不能让第一次使用项目时存在占位文件、后续轮次却不存在。脚手架恢复属于实验环境
初始化，不计入 corpus 入库时间，也不能携带上一 corpus 的任何内容。

删除不调用 LLM 或 embedding，因此：

```text
deletion_input_tokens = 0
deletion_output_tokens = 0
deletion_embedding_tokens = 0
deletion_total_token_cost = 0
```

runner 和 bridge 都会校验删除 token 总数必须为 0。

## 11. 指标计算

### 11.1 Accuracy

对第 `i` 题，Judge 给出：

```text
s_i ∈ {0, 1, 2, 3, 4}
```

报告同时记录：

```text
Average Accuracy (Hit 0-4) = (Σ s_i) / N
Normalized Accuracy (0-1)  = (Σ s_i) / (4N)
```

`Average Accuracy (normalization)` 与 `Normalized Accuracy (0-1)` 当前为同一数值。

### 11.2 Token F1

每个 prediction 与每个 gold answer 分别计算 token F1，并取最大值：

```text
precision = overlap_tokens / prediction_tokens
recall    = overlap_tokens / gold_tokens
F1        = 2 × precision × recall / (precision + recall)
```

文本规范化包括：转小写、删除逗号和英文标点、删除英文冠词/连接词 `a/an/the/and`、压缩
空白。该 F1 是参考 baseline 的词级实现；对于没有空格分词的中文文本，它不等同于中文
分词 F1。

### 11.3 Recall

当前逐题结果初始化 `Recall=0.0`，没有实现基于 evidence/document ID 的 retrieval recall。
因此报告中的 `Average Recall` 目前恒为 0，不应解释为 Nashsu 的真实检索召回率。

### 11.4 入库效率

```text
Total Insertion Time
  = Σ 每批 active duration
  = 所有解析、caption、知识生成、写入、页面 embedding
    加最终唯一一次 review sweep 的时间

Total Insertion Token Cost
  = ingest_input_tokens + ingest_output_tokens + ingest_embedding_tokens
```

文件 staging/copy 和批间服务重启不计入主指标。报告同时记录实际 operational wall clock、
源文档数、批大小、批数、重启次数、`Ingest Concurrency=1`、
`Includes Review Sweep=true` 和 `Review Sweep Count=1`。以 PaperScope 93 篇为例，默认形成
25/25/25/18 四批，批间重启 3 次，最终只 sweep 1 次。

发生回滚时还记录 `Batch Retry Count`、`Discarded Retry Time`、失败 usage 可用性、
`Snapshot Creation/Restore/Cleanup Time` 和计划/重试两类重启次数。这些审计字段均不进入
主要入库指标。

### 11.5 QA 效率

令单题端到端时间为 `t_i`，单题三类 token 为 `I_i/O_i/E_i`：

```text
Average End-to-End QA Time
  = (Σ t_i) / N

Average QA Token Cost
  = Σ(I_i + O_i + E_i) / N

Total QA Token Cost
  = ΣI_i + ΣO_i + ΣE_i
```

当前 JSON 报告沿用参考 baseline 字段名 `Average Retrieval Time (s)` 和
`Average Retrieval Token Cost`，但它们覆盖完整 Agent QA，不是纯 retrieval 子阶段。

### 11.6 删除效率

```text
Total Deletion Time = 清除前端状态和全部本轮持久化数据的墙钟时间
Total Deletion Token Cost = 0
```

删除计时只覆盖活动 Nashsu 项目的正式清理。批次快照位于项目外，且通常在每批最终成功后
立即删除；异常退出遗留的快照会在下次运行开始前清理。所有快照清理都发生在删除指标计时
之外，其耗时只记为 `Snapshot Cleanup Time`，不进入 `Total Deletion Time`。

## 12. 输出文件

每个实验写入：

```text
<output_dir>/<experiment_id>/wiki/
├── generated_answers.json
├── qa_eval_detailed_results.json
├── judge_telemetry.json
└── benchmark_metrics_report.json
```

- `generated_answers.json`：逐题 answer、references、trace、QA latency 和 token；
- `qa_eval_detailed_results.json`：追加 Judge score、reasoning、F1、Accuracy；
- `judge_telemetry.json`：Judge input/output token 总计及完整性标志；
- `benchmark_metrics_report.json`：入库、QA、Accuracy/F1/Recall、删除的聚合指标。

每个共享 corpus 另写：

```text
<output_dir>/groups/<corpus_id>/run.json
```

该文件记录 fingerprint、共享实验列表、固定 config、run ID、入库指标、删除指标和状态。

所有 JSON 使用临时文件后原子替换，避免中途写入留下半截结果。

## 13. Fail-closed 条件

以下情况会直接使对应实验阶段失败，而不是继续记录估算值：

- prepared 数量、字段、路径、文件大小或 SHA-256 不符合契约；
- 项目路径与当前打开的专用项目不一致；
- 固定模型、模式、parser、caption 或 embedding manifest 不一致；
- `maxContextSize` 不是官方默认 204800；
- bridge 未报告 `top_k=5`；
- embedding 返回向量维度不是 1024；
- 入库或 QA 中任何已发出 provider 请求缺失真实 usage；
- QA 返回空 answer 或缺少 session ID；
- 删除产生任何非零模型/embedding token；
- bridge token 缺失或错误；
- headless bridge listener 或指定项目在 300 秒内未就绪；
- headless 项目目录非空但不是有效 LLM Wiki 项目。

## 14. 部署验证状态与结果解释限制

模式与输入原则已经确定：官方 General 模板不做人工编辑、`mode=standard`、
`retrievalMode=standard`，并且不增加 QA 动作硬限制。正式 PaperScope 实验启动前已完成：

1. **服务器完整编译验证**：四个补丁可顺序应用；TypeScript typecheck、相关 Vitest、Rust
   `cargo check`、Python 14 项单元测试和带 `tauri/custom-protocol` 的 Linux release 构建通过。
2. **重启续批 smoke test**：Xvfb/WebKit readiness、完整进程组重启、重启后
   `continuation=true` 的受保护脚手架校验均通过，未调用模型。
3. **Agent trace 审计**：不增加动作 allowlist，也不强制至少一次检索。正式 QA 的逐题输出
   会记录工具序列；直接 final 或理论上的非检索动作保留为 Nashsu 官方行为，通过 trace 审计。
4. **Recall 指标**：当前 Recall 为占位 0。若要比较检索质量，应定义基于 gold document IDs
   或 evidence 的 Recall@K，并确保 Nashsu trace 可以映射回标准文档。

General 脚手架恢复、`outputLanguage=auto`、官方默认 chunk 参数和
`persistExtractedMarkdown=false` 已由 bridge 固定并进入 manifest。Judge usage 缺失按已确认
策略只记录 `usage_complete=false`，不使实验失败；Judge token 不进入本实验要求的 Accuracy、
端到端 QA、入库或删除成本指标。

上述 Recall 限制不影响本实验要求的 Accuracy、端到端回答时间和 token/入库/删除指标。
