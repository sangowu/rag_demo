# Beta 架构设计 — 三层 + A-RAG

## 完整流程

```
【离线预处理】
原始文档
    ├─ LLM 抽取 doc_summary（文档级，每份文档一条）→ doc_store
    └─ LLM 抽取 ChunkMetadata（含 chunk summary）
           ↓
       规则后处理 → JSON 落盘
           ↓
       embed(chunk_summary + chunk_text) → chunk_store（ChromaDB）
       （Contextual Retrieval）

【在线查询】
Query 输入
    ↓
【第一层】doc_store 混合检索 → 候选文件列表
    ↓
【第二层】chunk_store 验证过滤 → 热文件 + top chunks
    ↓
【第三层】A-RAG 智能检索
         Generator 拿到：doc_summary（去重）+ chunk_summary + chunk_text（按文件对齐）
    ↓
最终答案 + 源信息引用
```

---

## 【离线预处理】Metadata 生成

### 流程

```
原始文档
    ↓
LLM 批量抽取 → ChunkMetadata (Pydantic)
    ↓
规则后处理（覆盖可验证字段）
    ↓
metadata.json 落盘缓存（每个 chunk 一条记录）
```

### Metadata 字段设计

```json
{
  "doc_id": "ADI_2009_page_49",
  "chunk_id": "ADI_2009_page_49_chunk_2",
  "company": "ADI",
  "fiscal_year": 2009,
  "filing_date": "2009-11-20",
  "text_type": "table",
  "has_table": true,
  "domain": "finance",
  "report_section": "income_statement",
  "summary": "ADI Q4 2009 revenue was $541M, full year $2.1B...",
  "keywords": ["revenue", "quarterly results", "fiscal 2009"],
  "financial_terms": ["EPS", "gross margin", "operating income"],
  "numerical_values": [541, 2100, 0.153]
}
```

### 字段说明

| 字段 | 类型 | 来源 | 说明 |
|------|------|------|------|
| `doc_id` | str | 规则解析 | 文档唯一标识 |
| `chunk_id` | str | 规则生成 | chunk 唯一标识，chunk_read 工具依赖 |
| `company` | str | 规则解析 doc_id | 从 doc_id 直接提取，不信任模型 |
| `fiscal_year` | int | 规则解析 doc_id | 从 doc_id 直接提取，支持范围过滤 |
| `filing_date` | str | 模型抽取 | 可选，原文中明确标注时才填 |
| `text_type` | Literal | 模型判断 | table/narrative/footnote/header |
| `has_table` | bool | 规则检测 | 正则检测 `\|` 符号，不信任模型 |
| `domain` | str | 硬编码/模型 | 顶层分类，多语料库扩展用 |
| `report_section` | Literal | 模型判断 | 金融文档内细分类型（枚举约束） |
| `summary` | str | 模型生成 | 第一层向量搜索的核心字段 |
| `keywords` | list[str] | 模型生成 | 增强 BM25 匹配 |
| `financial_terms` | list[str] | 模型生成 | 金融术语，可用词典后验 |
| `numerical_values` | list[float] | 模型生成 | 纯数值，支持范围查询 |

### report_section 枚举生成方式（两阶段）

**第一阶段** — 随机抽取 50-100 个 chunk，开放式让模型描述类型：
```
"quarterly revenue summary" / "income statement data" / "risk factors" ...
```

**第二阶段** — 将所有原始标签喂给模型，归并为 8-12 个互斥标准类型：
```python
ReportSection = Literal[
    "income_statement",
    "balance_sheet",
    "cash_flow_statement",
    "management_discussion",
    "segment_data",
    "earnings_summary",
    "footnotes",
    "risk_factors",
    "other",   # 兜底，永远保留
]
```

> 两阶段只跑一次（离线），最终枚举人工确认后固化进 schema。

### 字段可信度策略

| 可信度 | 字段 | 处理方式 |
|--------|------|----------|
| 规则覆盖 | company / fiscal_year / has_table | 用规则结果直接覆盖模型输出 |
| 半信任 | numerical_values | 用正则从原文提取数字交叉验证 |
| 信任模型 | summary / keywords / financial_terms / text_type / report_section | 接受模型输出 |
| 硬编码 | domain | 当前语料库固定为 `finance` |

### 结构化输出保障

使用 Pydantic + `with_structured_output` 强制格式：
```python
llm.with_structured_output(ChunkMetadata)
# 字段缺失或类型错误 → 触发重试或 fallback
```

---

## 【离线预处理】Contextual Retrieval

### 原理

传统 chunking 切分后，chunk 脱离原文上下文，语义变弱：
```
原文：
"Apple Inc. was founded in 1976.
 The company reported revenue of $394B in 2022."

Chunk 2: "The company reported revenue of $394B in 2022."
→ "The company" 失去指代，embedding 语义不完整
```

Contextual Retrieval 在 embed 时将 summary 作为 prefix 拼接到 chunk 原文前，
让每个 chunk 的向量携带文档级语境。

### 实现方式

summary 字段本身已包含位置语境 + 内容摘要，embed 时直接拼接：

```python
# embed 时使用拼接版本
embed_text = f"{chunk['summary']}\n\n{chunk['text']}"
vector = model.encode(embed_text)

# 存入 ChromaDB 时分开存储
{
    "id": chunk_id,
    "document": chunk["text"],           # 原文，返回给 LLM 使用
    "embedding": encode(embed_text),     # 拼接版本，检索匹配用
    "metadata": {...}
}
```

> 检索时匹配拼接后的语义，返回给 Generator 的是干净原文，避免 LLM 看到重复内容。

### 与 metadata 生成的关系

summary 字段在离线预处理阶段已经生成，Contextual Retrieval **零额外成本**：
- summary 前半句描述文档位置（"From ADI 2009 annual report income statement..."）
- summary 后半句描述内容（"Q4 revenue $541M, full year $2.1B..."）
- 两者合体 = Contextual prefix + 内容摘要，一个字段实现双重作用

### 成本对比

| | 本方案 | Anthropic 原版 |
|---|---|---|
| prefix 来源 | metadata 生成时顺带产出 | 每个 chunk 单独调用 LLM |
| 额外 LLM 调用 | 0 | N（chunk 数量） |
| embed 时间 | 略增（拼接后文本变长） | 略增 |

---

## 【第一层】Metadata 混合检索

**目标**：快速从全量文档中定位高相关候选

```
Query
  ├─ 结构化字段硬过滤（company / fiscal_year / report_section）
  │   → 从 2408 个文档直接缩减到个位数/十位数
  ↓
  ├─ BGE-M3 向量搜索 summary 字段（语义理解）
  ├─ BM25 搜索 keywords + financial_terms（关键词精确匹配）
  └─ RRF 融合两路结果

输出：20-50 个候选文件
```

**RRF 融合公式**：
```
RRF_score = Σ 1 / (k + rank_i)   k=60
```
比 alpha 加权更稳定，无需调参。

---

## 【第二层】Chunk 级验证过滤

**目标**：从候选文件中筛出真正包含高质量内容的热文件

```
20-50 个候选文件
  ↓
每个文件取 top-3 chunk
  ↓
BGE Reranker 对 (query, chunk) 打分
  ↓
按平均相关性分数排序
  ↓
输出：3-5 个热文件
```

---

## 【第三层】A-RAG 智能检索

**目标**：在热文件范围内，LLM 多轮自主搜索并生成答案

### 三个工具

```
Tool1: keyword_search_in_hotfiles(query)
        → 在 3-5 个热文件内做 BM25 搜索
        → 适合精确数字、术语查询

Tool2: semantic_search_in_hotfiles(query)
        → 在 3-5 个热文件内做向量搜索
        → 适合语义理解、概念查询

Tool3: chunk_read(chunk_id)
        → 读取并返回完整 chunk 内容
        → 适合需要完整表格或上下文时
```

### LLM 多轮迭代流程

```
思考：需要什么信息？
  ↓
决策：调用哪个工具？
  └─ 先 semantic_search 理解概览
     再 keyword_search 找具体数据
     需要完整内容时 chunk_read
  ↓
评估：信息足够吗？
  ├─ 足够 → 生成答案
  └─ 不足 → 继续调用工具（最多 N 轮）
  ↓
生成答案 + 源信息引用
```

---

## 关键特点

### 层级漏斗

```
全量文档 (2408)
    ↓ 第一层：硬过滤 + hybrid 检索
候选文件 (20-50)
    ↓ 第二层：chunk 验证 + rerank
热文件 (3-5)
    ↓ 第三层：A-RAG 多轮搜索
最终答案
```

### 成本分布

| 阶段 | 计算类型 | 成本 |
|------|----------|------|
| 离线预处理 | LLM 批量抽取 metadata + summary | 一次性，可复用 |
| 离线预处理 | embed(summary + chunk) 写入 ChromaDB | 一次性，Contextual Retrieval 零额外 LLM 成本 |
| 第一层 | 向量检索 + BM25 | 轻量，毫秒级 |
| 第二层 | Reranker 打分 | 中等，百毫秒级 |
| 第三层 | LLM 多轮 reasoning | 集中投入，但范围已缩小 |

### 与基线对比的预期提升

| 指标 | 预期变化 | 原因 |
|------|----------|------|
| context_precision | ↑ 明显 | 热文件过滤减少无关内容 |
| faithfulness | ↑ 小幅 | chunk_read 获取完整表格 |
| answer_relevancy | ↑ 小幅 | 多轮搜索覆盖更全面 |
| latency | ↑（变慢） | 多层检索 + LLM 多轮调用 |
