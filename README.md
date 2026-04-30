# Structured RAG — FinQA Agentic RAG System

An end-to-end **Agentic RAG** system for structured financial documents, built to demonstrate the full AI Engineer stack: hybrid retrieval, LangGraph agentic loop, reflection, observability, and cross-dataset evaluation.

## Evaluation Results

### Retrieval Ablation — FinQA (n=200, seed=42, fixed-512 chunks)

Best configuration: `Dense (BGE-M3) + BM25 RRF + BGE-reranker-v2-m3, candidate_k=50`

| Config | Hit@1 | Hit@3 | Hit@5 | MRR@5 | ChunkPrec@5 | Latency |
|--------|-------|-------|-------|-------|-------------|---------|
| Dense only | 0.395 | 0.540 | 0.660 | 0.483 | 0.610 | 288ms |
| + BM25 hybrid | 0.285 | 0.470 | 0.560 | 0.386 | 0.445 | 252ms |
| **+ Reranker** ✓ | **0.475** | **0.680** | **0.740** | **0.574** | **0.685** | 2466ms |
| Dense + Reranker (no BM25) | 0.455 | 0.665 | 0.720 | 0.554 | 0.680 | 1399ms |

Key findings:
- BM25 alone hurts FinQA (financial terms are sparse; RRF introduces noise)
- Reranker recovers and improves by leveraging BM25 candidates as extra pool
- Increasing candidate_k 50→100 gives +1% Hit@5 at 2× latency cost — not worth it

### Retrieval Ablation — LegalBench-RAG-mini (n=200, seed=42, fixed-1024 chunks)

| Config | Hit@1 | Hit@5 | ChunkPrec@5 | Latency |
|--------|-------|-------|-------------|---------|
| Dense only | 0.285 | 0.565 | 0.565 | 301ms |
| **+ BM25 hybrid** ✓ | **0.300** | **0.575** | **0.575** | 498ms |
| + Reranker | 0.255 | 0.535 | 0.535 | 2655ms |

BM25 helps on legal text (exact keyword matching); Reranker hurts (BGE has financial bias).

### End-to-End Agent Evaluation — FinQA (n=50, seed=42)

Full agent pipeline: Planner → Tool → Generator → Reflector → Final

| Metric | Value | Notes |
|--------|-------|-------|
| Judge Score | **4.91 / 5.0** | Gemini LLM-as-Judge, semantic quality |
| Judge Pass Rate (≥4) | **97.6%** | Over answered queries |
| Refuse Rate | **16%** | grounding_score < 0.1 → structured refusal |
| Avg Latency | **2.5s / query** | Includes retrieval + reranking + LLM generation |

FinQA requires multi-step numerical reasoning; Exact Match is not a meaningful metric (gold answers are program-computed values like `0.5323` vs LLM's `"53.23%"`). LLM-as-Judge is the primary quality signal.

**Grounding check**: queries with max reranker score < 0.1 receive a structured refusal with score and threshold, rather than a hallucinated answer. Verified on out-of-domain queries (e.g. real-time stock prices → refused; financial document questions → answered).

## Architecture

![System Architecture](docs/architecture.png)

```
User Query
    ↓
[FastAPI /query]  ── SSE streaming ──→  [HTML/JS Demo UI]
    ↓
[Semantic Cache]  — return cached result if cosine similarity > 0.92
    ↓
[LangGraph Agent]
    ├─ Planner Node      — structured output: should_retrieve / should_rewrite
    │       ↓ (conditional routing)
    ├─ Tool Node         — hybrid retrieval; computes grounding_score (max reranker score)
    │       ├─ grounding ok (≥ 0.1)  → Generator Node
    │       └─ grounding weak (< 0.1)
    │               ├─ TAVILY_API_KEY set → Web Search Node → Generator Node
    │               └─ no Tavily key    → Refuse Node → Final Node
    ├─ Generator Node    — produce answer with inline citations
    ├─ Reflector Node    — self-evaluate quality; retry ≤ 2 times
    │                      (skipped after web search or knowledge-only answers)
    └─ Final Node        — format output; deduplicate sources by retrieval type
         ↓
[Hybrid Retriever]
    ├─ VectorStore     — pgvector + BGE-M3 dense vectors (fixed-512 chunks)
    ├─ BM25Store       — rank-bm25 sparse index, RRF fusion (candidate_k=50)
    └─ Reranker        — BGE-reranker-v2-m3 cross-encoder
```

**Routing logic:**
- `should_retrieve=False` → skip Tool Node, generate from LLM knowledge directly
- `grounding_score < 0.1` + Tavily available → web search fallback, then generate
- `grounding_score < 0.1` + no Tavily → structured refusal with score and threshold
- Reflector retry only fires when `should_retrieve=True` and no web search was used

## Tech Stack

| Area | Choice |
|------|--------|
| Embedding | BAAI/bge-m3 |
| Vector store | pgvector (PostgreSQL), per-strategy tables |
| Sparse retrieval | rank-bm25, RRF fusion |
| Reranker | BAAI/bge-reranker-v2-m3 |
| Agent framework | LangGraph |
| LLM | Gemini (gemini-3.1-flash-lite-preview) via Google API |
| Web Search | Tavily (fallback when local grounding < 0.1) |
| Tracing | Langfuse + LangSmith |
| Evaluation | Hit@K / MRR / ChunkPrec + LLM-as-Judge |
| API | FastAPI + SSE streaming |
| Frontend | Gradio |

## Project Structure

```
rag_demo/
├── config/
│   └── settings.yaml              # Central configuration
├── data/
│   ├── finqa/docs/                # FinQA source documents (.md)
│   ├── chroma/                    # ChromaDB persistent storage
│   ├── bm25_index.pkl             # BM25 index
│   ├── semantic_cache.pkl         # Semantic cache (persisted)
│   ├── doc_summaries.jsonl        # LLM-generated doc summaries
│   ├── eval_results.json          # Latest evaluation results
│   └── eval_log.jsonl             # Experiment comparison log
├── src/
│   ├── config.py                  # Config loader
│   ├── data_loader.py             # FinQA dataset downloader
│   ├── chunk_manager.py           # Text chunking (fixed/recursive/semantic)
│   ├── vector_store.py            # ChromaDB + BGE-M3 (ticker/year metadata)
│   ├── bm25_store.py              # BM25 sparse index
│   ├── reranker.py                # BGE cross-encoder reranker
│   ├── retriever.py               # Hybrid retrieval + semantic cache + query rewrite
│   ├── query_rewriter.py          # LLM-based ticker expansion (ADI → Analog Devices)
│   ├── semantic_cache.py          # Cosine similarity cache, persisted to pkl
│   ├── guardrails.py              # Input relevance check + output grounding check
│   ├── metadata_extractor.py      # ticker/year extraction for pre-filtering
│   ├── doc_metadata_extractor.py  # LLM-based metadata extraction for multi-source docs
│   ├── document_loader.py         # URL / PDF / Markdown loader
│   ├── doc_summarizer.py          # LLM batch doc summarization
│   ├── summary_store.py           # ChromaDB summary collection (2-stage retrieval)
│   ├── ingestion_registry.py      # Document ingestion tracker
│   ├── evaluator.py               # RAGAS + Hit@K/MRR batch evaluation
│   ├── llm_judge.py               # LLM-as-Judge per-query scoring
│   ├── tracing.py                 # Langfuse callback singleton (atexit flush)
│   ├── agent/
│   │   ├── state.py               # LangGraph AgentState
│   │   ├── tools.py               # search_local / rewrite_query / search_web
│   │   ├── nodes.py               # 6 agent nodes + PlannerDecision structured output
│   │   └── graph.py               # LangGraph StateGraph with web search routing
│   └── api/
│       ├── main.py                # FastAPI SSE endpoints + async ingestion
│       └── static/
│           └── index.html         # HTML/JS demo UI
└── scripts/
    ├── ingest_finqa.py            # Batch FinQA document ingestion (--force to re-ingest)
    ├── ingest_financebench.py     # FinanceBench ingestion (evidence_text_full_page)
    ├── merge_eval.py              # Merge FinQA + FinanceBench into mixed eval set
    ├── generate_financebench_qa.py # Convert FinanceBench dataset to eval format
    ├── generate_qa.py             # LLM-based QA pair generation
    ├── eval_retrieval.py          # Standalone retrieval evaluation
    ├── compare_evals.py           # Multi-run experiment comparison table
    └── eval_smoke.py              # CI smoke test (recall@5 ≥ 0.6)
```

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set environment variables

```bash
# Required
export GOOGLE_API_KEY=your_gemini_key          # Gemini LLM
export PG_DSN=postgresql://user:pass@host/db   # pgvector database

# Optional — web search fallback (free tier: https://app.tavily.com)
export TAVILY_API_KEY=tvly-xxxx

# Optional — observability (free tier: https://cloud.langfuse.com)
export LANGFUSE_PUBLIC_KEY=pk-lf-xxxx
export LANGFUSE_SECRET_KEY=sk-lf-xxxx

# Optional — LangSmith tracing
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_API_KEY=your_langsmith_key
export LANGCHAIN_PROJECT=rag-demo
```

All keys can alternatively be placed in a `.env` file at the project root.

### 3. Ingest documents

```bash
# FinQA
python src/data_loader.py
python scripts/ingest_finqa.py

# FinanceBench (no PDF download required)
python scripts/generate_financebench_qa.py
python scripts/ingest_financebench.py
```

### 5. Start the API + Demo UI

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8080
# Open http://localhost:8080
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | HTML/JS demo UI |
| `GET` | `/health` | Liveness probe — returns ChromaDB chunk count + BM25 status |
| `POST` | `/query` | SSE streaming query (LangGraph agent) · rate limit 30/min |
| `POST` | `/ingest` | Async document ingestion → returns `job_id` · rate limit 10/min |
| `GET` | `/ingest/{job_id}` | Poll ingestion status and progress |

**Authentication**: set `RAG_API_KEY` env var to enable Bearer token auth. Unset = open (local dev).

```bash
export RAG_API_KEY=your-secret-key
curl -H "Authorization: Bearer your-secret-key" http://localhost:8080/query ...
```

**Logging**: structured JSON logs by default. Set `LOG_FORMAT=text` for human-readable output during development.

### Async Ingestion

```bash
# Submit ingestion job
curl -X POST http://localhost:8080/ingest \
  -H "Content-Type: application/json" \
  -d '{"doc_id": "AAPL_2023_page_1.pdf", "text": "..."}'
# → {"job_id": "a1b2c3d4", "status": "queued"}

# Poll status
curl http://localhost:8080/ingest/a1b2c3d4
# → {"status": "done", "progress": 12, "total": 12}
```

## Docker

```bash
cp .env.example .env   # fill in GOOGLE_API_KEY, PG_DSN
docker compose up --build
```

## Evaluation

### RAGAS + Hit@K/MRR batch evaluation

```bash
# FinQA — 50 RAGAS samples + 200 retrieval samples
python src/evaluator.py --n 50 --n-retrieval 200 --tag baseline

# FinanceBench — 20 RAGAS samples + 150 retrieval samples
python src/evaluator.py --tag financebench_evidence \
  --eval-path data/results/financebench_qa.jsonl \
  --n 20 --n-retrieval 150

# Mixed (FinQA + FinanceBench) — 50 RAGAS samples + 300 retrieval samples
python scripts/merge_eval.py   # generates data/eval_mixed.jsonl
python src/evaluator.py --tag mixed_eval \
  --eval-path data/eval_mixed.jsonl \
  --n 50 --n-retrieval 300

# With query rewriting enabled
python src/evaluator.py --n 10 --n-retrieval 200 --tag rewrite --query-rewrite

# Compare multiple experiment runs
python scripts/compare_evals.py
```

Metrics tracked per run: `faithfulness`, `context_precision`, `answer_relevancy`, `hit@1/3/5`, `mrr@1/3/5`, per-phase latency, token counts.

### LLM-as-Judge

```bash
python src/llm_judge.py
```

### CI smoke test

Runs automatically on every push/PR via GitHub Actions (recall@5 threshold: 0.6).

```bash
python scripts/eval_smoke.py --n 10 --top-k 5 --threshold 0.6
```

## Configuration

Key settings in `config/settings.yaml`:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `chunking.strategy` | `fixed` | `fixed` / `recursive` / `semantic` |
| `chunking.chunk_size` | `512` | Tokens per chunk |
| `retriever.mode` | `custom` | `custom` (BM25+dense) or `m3_hybrid` |
| `retriever.top_k` | `5` | Final top-k results |
| `retriever.custom.alpha` | `0.7` | Dense weight (1.0=pure dense, 0.0=pure BM25) |
| `retriever.custom.candidate_k` | `40` | Candidates before reranking |
| `semantic_cache.threshold` | `0.92` | Cosine similarity threshold for cache hit |
| `agent.max_retries` | `2` | Max reflection retries |
| `llm.base_url` | `http://localhost:8000/v1` | LLM endpoint (local or API) |

## Experiment Findings

| Experiment | Dataset | Hit@5 | Faithfulness | Notes |
|-----------|---------|-------|--------------|-------|
| baseline (alpha=0.4, ck=20) | FinQA | 0.500 | 0.750 | Starting point |
| alpha=0.7, candidate_k=40 | FinQA | 0.896 | 0.925 | +39.6% Hit@5 |
| recursive chunking (char-based) | FinQA | 0.710 | 0.763 | Bug: chunk_size was in chars not tokens → 5× smaller chunks |
| semantic chunking (unconstrained) | FinQA | 0.715 | 0.749 | Bug: min_chunk_size not set → over-splits short financial sentences |
| metadata pre-filter (ticker/year) | FinQA | 0.760 | 0.675 | Low ticker recall from query |
| summary-based 2-stage filter | FinQA | 0.754 | 0.775 | −32% retrieval time, lower Hit@5 |
| **ticker-year header injection** | **FinQA** | **0.967** | **1.000** | Largest single improvement |
| query rewriting (full) | FinQA | 0.967 | 1.000 | +8.7% Answer Relevancy, +68% retrieval latency |
| chunk_size=1024 | FinQA | 0.645 | 0.804 | Larger chunks hurt: context too coarse for reranker |
| **FinanceBench cross-domain** | FinanceBench | **0.907** | 0.733 | Zero-shot, −6% vs FinQA best |
| Mixed eval (fixed/1024) | Mixed | **0.813** | 0.852 | Both corpora in retrieval pool as hard negatives |

**Key insight — Header injection**: FinQA documents never mention ticker symbols in body text. Prepending `[TICKER | YEAR]` to every chunk at index time bridges the vocabulary gap for both BM25 and dense retrieval, yielding the largest single improvement (+8% Hit@5, +17.5% Faithfulness). Applied to both FinQA and FinanceBench at ingest time.

**Chunking unit bug**: `RecursiveCharacterTextSplitter` measures `chunk_size` in characters, while `TokenTextSplitter` (fixed strategy) uses tokens. Setting `chunk_size=512` produced 5× smaller chunks for recursive (avg 72 words vs 386 words). Fixed by switching to `from_tiktoken_encoder`. `SemanticChunker` does not accept `chunk_size` at all — fixed by setting `min_chunk_size=500` and `breakpoint_threshold_amount=95` to prevent over-splitting.

**Agentic query rewriting**: The Planner node uses structured output (`PlannerDecision`) to decide whether a query contains ticker symbols needing expansion. This avoids latency cost for queries that already use full company names, while gaining Answer Relevancy improvements for ticker-heavy queries.

**Cross-domain generalization**: Zero-shot evaluation on FinanceBench (2022–2023 10-K/10-Q) shows Hit@5 degrading only 6% from FinQA best (0.967 → 0.907), validating that the hybrid retrieval design generalizes across financial document sources without retraining.

**Mixed evaluation**: Stratified sampling (150 FinQA + 150 FinanceBench, seed=42) with the full combined retrieval pool. Hit@5 0.813 reflects a harder setting than single-dataset runs — all documents from both corpora act as hard negatives.
