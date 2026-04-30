# Structured RAG — Project Context

## Goal

Build an **Agentic RAG** system for structured financial and legal documents, targeting AI Engineer interview readiness.

Primary objective: demonstrate the full AI Engineer stack — hybrid retrieval, agentic tool-use loop, grounding checks, observability, evaluation.

Evaluation datasets:
- **FinQA** (train + dev, 2409 docs / 7134 questions) — domain-specific financial Q&A
- **LegalBench-RAG-mini** (359 docs / 756 queries, CC-BY) — cross-domain generalization

Key metrics: Hit@1/3/5, MRR@5, ChunkPrec@5, avg_latency_ms

## Benchmark Results（2026-04-29，seed=42，n=200）

### FinQA（domain-specific，fixed-512 chunk — 最优）
| Config | Hit@1 | Hit@5 | ChunkPrec@5 | Latency |
|--------|-------|-------|-------------|---------|
| Dense only | 0.395 | 0.660 | 0.610 | 288ms |
| + BM25 hybrid | 0.285 | 0.560 | 0.445 | 252ms |
| **+ Reranker** | **0.475** | **0.740** | **0.685** | 2466ms |

### FinQA（fixed-1024 chunk — 对比）
| Config | Hit@1 | Hit@5 | ChunkPrec@5 | Latency |
|--------|-------|-------|-------------|---------|
| Dense only | 0.375 | 0.630 | — | 290ms |
| + BM25 hybrid | 0.280 | 0.550 | — | 433ms |
| + Reranker | 0.425 | 0.700 | — | 2612ms |

### LegalBench（cross-domain，fixed-1024 chunk）
| Config | Hit@1 | Hit@5 | ChunkPrec@5 | Latency |
|--------|-------|-------|-------------|---------|
| Dense only | 0.285 | 0.565 | 0.565 | 301ms |
| **+ BM25 hybrid** | **0.300** | **0.575** | **0.575** | 498ms |
| + Reranker | 0.255 | 0.535 | 0.535 | 2655ms |
| + Query Rewriter | 0.255 | 0.535 | 0.535 | 3482ms |
| + Header Inject | 0.265 | 0.560 | 0.560 | 3507ms |

### 关键发现
- **fixed-512 优于 fixed-1024（FinQA）**：Hit@5 +4%（0.700→0.740），Hit@1 +5%（0.425→0.475）
- **BM25 在 FinQA 拖累性能**（-10% Hit@5 vs Dense only）：金融术语稀疏，RRF 融合引入噪音
- **BM25 在 LegalBench 有效**（+1%）：法律合同词汇精确匹配有效
- **Reranker 在 LegalBench 有害**（-3%）：BGE-reranker 对法律域有偏差
- **Query Rewriter 无增益**：ticker 展开对法律文本无效，应改为 agent 自主决策
- **Chunk 策略实验结论**：FinQA 用 fixed-512 最优；LegalBench 用 fixed-1024 最优
- **Meta Filter 严重损害 FinQA**（-27% Hit@5）：年报 doc_id 年份与 query 年份不一致（2009 报告含 2008 数据）

### 评测函数修复（2026-04-29）
- **问题**：`_chunk_hits_gold` 对 `table_*` 类型 gold_inds 用 60 字符前缀匹配，但 gold 文本是 FinQA 数据集自带的线性化描述（`"the X of Y is VALUE ;"`），与我们存储的 Markdown 表格格式完全不同，导致大量正确检索被误判为 wrong_chunk
- **修复**：`table_*` 改为提取 `is VALUE` 单元格值，≥2 个值出现在 chunk 里即为命中；`text_*` 保持前缀匹配
- **效果**：CPrec@5 从 0.315 → **0.685**（FinQA + Reranker），反映真实检索能力

### 端到端 Agent 评测（n=50，seed=42，FinQA）
| 指标 | 数值 | 说明 |
|------|------|------|
| Judge Score | **4.91 / 5.0** | Gemini LLM-as-Judge 语义质量 |
| Judge Pass Rate (≥4) | **97.6%** | 已回答 query 中 |
| Refuse Rate | **16%** | grounding_score < 0.1 触发拒答 |
| Avg Latency | **2.5s** | 含检索 + 重排 + LLM 生成 |

注：FinQA gold answer 是程序执行结果（如 `0.5323`），LLM 答 `"53.23%"`，EM 恒为 0；LLM-as-Judge 是正确指标。

### Bad Case 分析（n=100，baseline + reranker）
| 数据集 | wrong_chunk | no_doc_retrieved |
|--------|-------------|-----------------|
| FinQA | 56% | 44% |
| LegalBench | 57% | 43% |

wrong_chunk 占多数原因（已修复）：FinQA table_* gold_inds 是线性化描述格式，与 Markdown 表格格式不同，60 字符前缀匹配永远失败；修复后改为单元格值匹配，CPrec@5 大幅提升。
LegalBench wrong_chunk：reranker 把同文档其他 chunk 排到 gold chunk 之前（真实检索失败，非评测 bug）。

## Tech Decisions

| Area | Choice |
|------|--------|
| Embedding | BAAI/bge-m3 |
| Vector store | pgvector (PostgreSQL), per-strategy tables |
| Sparse retrieval | rank-bm25 |
| Reranker | BAAI/bge-reranker-v2-m3 |
| Agent framework | LangGraph（升级中→动态 tool-use） |
| LLM | Gemini (gemini-3.1-flash-lite-preview) via Google API |
| Tracing | LangSmith |
| Evaluation | Hit@K / MRR / ChunkPrec + bad case analysis |
| API | FastAPI + SSE streaming |
| Frontend | Streamlit |
| GPU | EC2 Tesla T4 (AWS g4dn.xlarge, eu-west-1) — embedding server |
| Embedding Server | FastAPI on EC2:8000，systemd 开机自启，BGE-M3 + BGE-reranker |
| Local Dev GPU | RTX 4090D, 24 GB VRAM |
| Web Search | Tavily（框架就绪，待 API Key） |

## AWS Infrastructure

| Resource | Details |
|----------|---------|
| EC2 | `rag-embedding-server`，g4dn.xlarge，Tesla T4，eu-west-1 |
| RDS | `rag-database-1`，db.t3.micro，pgvector，endpoint: `rag-database-1.cn4eui80g8r4.eu-west-1.rds.amazonaws.com` |
| S3 | `rag-documents-sango`，BM25 索引存储 |
| ECS | `rag-demo-cluster` / `rag-demo-service`，Fargate，1vCPU/2GB |
| ECR | `rag-demo`，`569260897196.dkr.ecr.eu-west-1.amazonaws.com/rag-demo` |
| Secrets | `rag-demo/pg-dsn`，`rag-demo/google-api-key` |
| ALB | `rag-demo-alb-947745034.eu-west-1.elb.amazonaws.com` |

## Architecture

```
User Query
    ↓
[Planner Node]    — structured output: should_retrieve / should_rewrite
    ↓ (conditional edge)
    ├─ should_retrieve=False → [Generator] → [Reflector] → [Final]
    └─ should_retrieve=True  → [Tool Node]
                                    ├─ grounding ok (≥0.1) → [Generator] → [Reflector]
                                    │                                           ↓
                                    │                                  retry ≤2 or [Final]
                                    └─ grounding weak (<0.1)
                                            ├─ TAVILY_API_KEY set → [Web Search Node] → [Generator] → [Reflector] → [Final]
                                            └─ no Tavily          → [Refuse Node] → [Final]  (拒答)
```

- Reflector retry 仅在 `should_retrieve=True` 时触发（常识问题不走 tool 重试）
- `grounding_score` = top-1 reranker score（BGE cross-encoder，normalize=True）
- `grounding_threshold = 0.1`（settings.yaml 可配置，0.3 太严导致 30% refuse rate）
- 拒答消息含 score 和 threshold，便于调试
- 每个节点写入 `trace` list，`metrics.trace` 输出完整决策链

## Conventions

- Source documents stored as Markdown under `data/{dataset}/docs/`
- Each module in `src/` independently importable, no side effects at import
- All scripts runnable from project root: `python scripts/xxx.py`
- Temp/inspection scripts prefixed with `_`
- Chunking strategies use isolated resources: `chunks_{strategy}[_{suffix}]` table, `data/bm25_{strategy}[_{suffix}].pkl`, `data/registry_{strategy}[_{suffix}].json`
- LLM access centralised via `src/llm_factory.py` — auto-fallback to Secrets Manager if GOOGLE_API_KEY not set locally
- All retrieval tools return `list[RetrievalResult]` (see `src/schemas.py`); eval scripts call `.to_dict()` for backward compatibility

## Key Files

| File | Role |
|------|------|
| `src/schemas.py` | RetrievalResult 统一出口 schema |
| `src/retriever.py` | 混合检索核心，返回 list[RetrievalResult] |
| `src/vector_store.py` | pgvector，含 start_char/end_char 列 |
| `src/bm25_store.py` | BM25Okapi，S3 上传/下载 |
| `src/reranker.py` | BGE cross-encoder，懒加载 |
| `src/agent/tools.py` | search_local / rewrite_query / search_web |
| `src/llm_factory.py` | Gemini 工厂，Secrets Manager fallback |
| `scripts/build_index.py` | 建索引，支持 --chunk-size/--overlap/--chunk-strategy/--table-suffix |
| `scripts/ablation_study.py` | 消融实验，支持 --datasets/--table-suffix/--configs |
| `scripts/bad_case_analysis.py` | 失败案例分类与抽样展示 |
| `scripts/generate_ablation_chart.py` | 消融结果可视化 |

## Learning Progress

### Phase 1 — RAG Foundation
- [x] Task 1: `src/data_loader.py` — FinQA → 2408 docs + 7134 eval records
- [x] Task 2: `src/chunk_manager.py` — fixed/recursive/semantic 三策略，表格保护，LangChain splitters
- [x] Task 3: `src/vector_store.py` — pgvector + BGE-M3，per-strategy 独立表，embed_text/content 分离，start_char/end_char
- [x] Task 4: `src/bm25_store.py` — BM25Okapi 索引，pkl 持久化，S3 集成
- [x] Task 5: `src/reranker.py` — BGE cross-encoder，懒加载，normalize=True
- [x] Task 6: `src/retriever.py` — hybrid 检索，RRF 融合，返回 list[RetrievalResult]

### Phase 2 — Agentic Layer
- [x] Task 7: `src/agent/state.py` — AgentState TypedDict，operator.add reducer
- [x] Task 8: `src/agent/tools.py` — search_local / rewrite_query / search_web 三工具，QueryRewriter 从 Retriever 解耦
- [x] Task 9: `src/agent/nodes.py` — 5 节点，LLM 单例，generator/reflector prompt
- [x] Task 10: `src/agent/graph.py` — StateGraph 组装，条件边重试逻辑，compile
- [x] Task 10b: Agentic planner 升级 — 动态 tool 路由，grounding check，拒答机制

### Phase 3 — Production
- [x] Task 11: `src/api/main.py` — FastAPI SSE 流式查询，/ingest 文档摄入
- [x] Task 12: `src/ingestion_registry.py` — JSON 注册表
- [x] Task 13: Multi-turn conversation — messages 历史，history 注入
- [x] Task 14: Per-query metrics — latency_ms / prompt_tokens / completion_tokens / retry_count

### Phase 4 — Deployment
- [x] Task 15: `Dockerfile` + `docker-compose.yml`
- [x] Task 16: GitHub Actions eval CI — smoke test recall@5

### Phase 5 — Uplift
- [x] Task 18: `src/evaluator.py` (RAGAS batch)
- [x] Task 19: `src/llm_judge.py` (LLM-as-Judge)

### Phase 6 — Showcase
- [x] Task 20: Streamlit UI
- [x] Task 21: README + architecture diagram

### Phase 7 — Chunking Strategy Experiments
- [x] Task 22: `src/llm_factory.py` — 统一 Gemini 客户端工厂，Secrets Manager fallback
- [x] Task 23: `src/contextual_chunker.py` — Contextual Retrieval，Gemini 生成上下文前缀
- [x] Task 24: `scripts/build_index.py` — 4 策略 + --chunk-size/--overlap/--chunk-strategy/--table-suffix
- [x] Task 25: `scripts/eval_smoke.py` — recall@1/3/5 + MRR，char-level GT 支持
- [x] Task 27: 混合数据集支持 — FinQA + LegalBench 联合评测，每数据集独立采样

### Phase 8 — Ablation Study & Evaluation
- [x] Task 28: `scripts/ablation_study.py` — 5 配置，--datasets/--table-suffix/--configs，带时间戳日志
- [x] Task 29: `scripts/generate_ablation_chart.py` — 消融结果可视化
- [x] Task 30: LegalBench-RAG-mini 接入 — 359 docs / 756 queries，char-level 评测，Hit@5=57.5%
- [x] Task 31: `scripts/bad_case_analysis.py` — 失败案例三分类，per-dataset 统计，bad_cases.jsonl
- [x] Task 32: Chunk 策略实验 — fixed-1024/512 vs recursive-1024，结论：fixed-1024 最优
- [x] Task 33: `src/schemas.py` — RetrievalResult 统一 schema，from_local_dict / from_web_dict

### Phase 9 — Agentic RAG Upgrade（进行中）
- [x] Task 34: agent tools 拆分 — search_local / rewrite_query / search_web 独立 tool
- [x] Task 35: Planner 动态路由 — should_retrieve / should_rewrite 条件边，跳过不需要检索的 query
- [x] Task 36: Grounding check — grounding_score（max reranker score）< 0.3 时拒答，refuse 节点打标
- [x] Task 37: Web search 集成 — tavily-python 加入依赖，TAVILY_API_KEY 文档化，search_web tool 已就绪
- [x] Task 38: Agent trace 记录 — planner/tool/reflector/refuse 各节点写入 trace list，metrics 输出完整决策链
- [x] Task 38b: 端到端评测 — `scripts/eval_e2e.py`，Judge Score 4.91/5，Refuse Rate 16%，avg 2.5s
- [x] Task 39: Langfuse 接入 — src/tracing.py 单例，api/main.py + eval_e2e.py 注入 callback，未配置时静默跳过

## Ablation Study 配置说明

| Config | 组件 | 关键参数 |
|--------|------|---------|
| 1 Dense only | BGE-M3 dense | NullBM25 + PassThroughReranker |
| 2 + BM25 hybrid | Dense + BM25 RRF | PassThroughReranker |
| 3 + Reranker | Dense + BM25 + BGE reranker | 完整 pipeline |
| 4 + Query Rewriter | Config 3 + LLM ticker 展开（已废弃，改为 agent tool） | rewrite=True |
| 5 + Header Inject | Config 4 + contextual 索引 | chunks_contextual 表 |

最优配置：FinQA → Config 3（Reranker），LegalBench → Config 2（BM25 hybrid）
