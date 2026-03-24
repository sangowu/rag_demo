# Structured RAG — Project Context

## Goal

Build an **Agentic RAG** system for structured financial documents (FinQA dataset), targeting B2B interview readiness.

Primary objective: demonstrate the full AI Engineer stack — hybrid retrieval, LangGraph agentic loop, reflection, observability, evaluation.

Evaluation dataset: **FinQA** (train + dev, ~7134 questions).
Key metrics: **RAGAS** (faithfulness, context_precision, answer_relevancy) + **LLM-as-Judge**.

## Tech Decisions

| Area | Choice |
|------|--------|
| Embedding | BAAI/bge-m3 |
| Vector store | ChromaDB (cosine, persist to `data/chroma/`) |
| Sparse retrieval | rank-bm25 |
| Reranker | BAAI/bge-reranker-v2-m3 |
| Agent framework | LangGraph |
| LLM | Qwen3-8B via ModelScope API |
| Tracing | LangSmith |
| Evaluation | RAGAS + LLM-as-Judge |
| API | FastAPI + SSE streaming |
| Frontend | Streamlit |
| GPU | RTX 4090D, 24 GB VRAM |

## Architecture

```
User Query
    ↓
[Planner Node]    — decide whether to retrieve
    ↓
[Tool Node]       — search_internal (hybrid: vector + BM25 + reranker)
    ↓
[Generator Node]  — produce answer with inline citations
    ↓
[Reflector Node]  — self-evaluate; retry ≤2 times if quality low
    ↓
[Final Node]      — format output + source attribution
```

## Conventions

- Source documents stored as Markdown under `data/finqa/docs/`
- Each module in `src/` independently importable, no side effects at import
- All scripts runnable from project root: `python src/xxx.py` or `python scripts/xxx.py`
- Temp/inspection scripts prefixed with `_`

## Learning Progress

### Phase 1 — RAG Foundation
- [x] Task 1: `src/data_loader.py` — FinQA → 2408 docs + 7134 eval records
- [x] Task 2: `src/chunk_manager.py` — fixed/recursive/semantic 三策略，表格保护，LangChain splitters
- [x] Task 3: `src/vector_store.py` — ChromaDB + BGE-M3，dense/sparse 双模式，CRUD
- [x] Task 4: `src/bm25_store.py` — BM25Okapi 索引，pkl 持久化，归一化分数，delete 重建
- [x] Task 5: `src/reranker.py` — BGE cross-encoder，懒加载，normalize=True，top-k 重排
- [x] Task 6: `src/retriever.py` — custom/m3_hybrid 双模式，alpha 融合，依赖注入

### Phase 2 — Agentic Layer
- [x] Task 7: `src/agent/state.py` — AgentState TypedDict，operator.add reducer
- [x] Task 8: `src/agent/tools.py` — @tool 装饰器，Retriever 单例，tools 列表导出
- [x] Task 9: `src/agent/nodes.py` — 5 节点，LLM 单例，generator/reflector prompt
- [x] Task 10: `src/agent/graph.py` — StateGraph 组装，条件边重试逻辑，compile

### Phase 3 — Production
- [x] Task 11: `src/api/main.py` — FastAPI SSE 流式查询，/ingest 文档摄入
- [x] Task 12: `src/ingestion_registry.py` — JSON 注册表，is_registered/register/list_all
- [x] Task 13: Multi-turn conversation — messages 历史，add_messages reducer，history 注入
- [x] Task 14: Per-query metrics — latency_ms / prompt_tokens / completion_tokens / retry_count

### Phase 4 — Deployment
- [x] Task 15: `Dockerfile` + `docker-compose.yml` — slim 镜像，data/models volume 挂载
- [x] Task 16: GitHub Actions eval CI — smoke test recall@5，threshold 0.6，push/PR 触发

### Phase 5 — Uplift
- [ ] Task 17: `src/metadata_extractor.py` — 跳过（后期优化）
- [x] Task 18: `src/evaluator.py` (RAGAS batch)
- [x] Task 19: `src/llm_judge.py` (LLM-as-Judge)

### Phase 6 — Showcase
- [x] Task 20: Streamlit UI
- [x] Task 21: README + architecture diagram
