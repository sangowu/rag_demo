# Structured RAG — FinQA Agentic RAG System

An end-to-end **Agentic RAG** system for structured financial documents (FinQA dataset), built to demonstrate the full AI Engineer stack: hybrid retrieval, LangGraph agentic loop, reflection, observability, and evaluation.

## Evaluation Results

Best configuration: `fixed chunking / chunk_size=512 / alpha=0.7 / candidate_k=40 / ticker-year header injection`

| Metric | Score |
|--------|-------|
| Hit@1 | 0.814 |
| Hit@3 | 0.940 |
| Hit@5 | **0.967** |
| MRR@5 | 0.878 |
| Faithfulness | **1.000** |
| Context Precision | 0.833 |
| Answer Relevancy | 0.890 |

Evaluated on 200 QA pairs (retrieval) + 10 samples (RAGAS), using locally generated QA pairs from FinQA documents.

## Architecture

```
User Query
    ↓
[FastAPI /query]  ── SSE streaming ──→  [Streamlit UI]
    ↓
[LangGraph Agent]
    ├─ Planner Node    — decide whether to retrieve
    ├─ Tool Node       — search_internal (hybrid retrieval)
    ├─ Generator Node  — produce answer with inline citations
    ├─ Reflector Node  — self-evaluate quality; retry ≤ 2 times
    └─ Final Node      — format output + source attribution
         ↓
[Hybrid Retriever]
    ├─ VectorStore     — ChromaDB + BGE-M3 dense vectors
    ├─ BM25Store       — rank-bm25 sparse index
    └─ Reranker        — BGE-reranker-v2-m3 cross-encoder
```

## Tech Stack

| Area | Choice |
|------|--------|
| Embedding | BAAI/bge-m3 |
| Vector store | ChromaDB (cosine, persistent) |
| Sparse retrieval | rank-bm25 |
| Reranker | BAAI/bge-reranker-v2-m3 |
| Agent framework | LangGraph |
| LLM | Qwen3-8B (local via llama.cpp or ModelScope API) |
| Tracing | LangSmith |
| Evaluation | RAGAS + LLM-as-Judge + Hit@K/MRR |
| API | FastAPI + SSE streaming |
| Frontend | Streamlit |

## Project Structure

```
rag_demo/
├── config/
│   └── settings.yaml              # Central configuration
├── data/
│   ├── finqa/docs/                # FinQA source documents (.md)
│   ├── chroma/                    # ChromaDB persistent storage
│   ├── bm25_index.pkl             # BM25 index
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
│   ├── retriever.py               # Hybrid retrieval + meta/summary filter
│   ├── metadata_extractor.py      # ticker/year extraction for pre-filtering
│   ├── doc_summarizer.py          # LLM batch doc summarization
│   ├── summary_store.py           # ChromaDB summary collection (2-stage retrieval)
│   ├── ingestion_registry.py      # Document ingestion tracker
│   ├── evaluator.py               # RAGAS + Hit@K/MRR batch evaluation
│   ├── llm_judge.py               # LLM-as-Judge per-query scoring
│   ├── app.py                     # Streamlit frontend
│   ├── agent/
│   │   ├── state.py               # LangGraph AgentState
│   │   ├── tools.py               # search_internal tool
│   │   ├── nodes.py               # 5 agent nodes
│   │   └── graph.py               # LangGraph StateGraph
│   └── api/
│       └── main.py                # FastAPI SSE endpoints
└── scripts/
    ├── ingest_finqa.py            # Batch document ingestion
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
# ModelScope API (for cloud LLM)
export MODELSCOPE_API_KEY=your_api_key_here

# LangSmith tracing (configured — view traces at smith.langchain.com)
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_API_KEY=your_langsmith_key
export LANGCHAIN_PROJECT=rag-demo
```

### 3. Start local LLM (optional — or use ModelScope API)

```bash
# Build llama.cpp
git clone https://github.com/ggerganov/llama.cpp && cd llama.cpp
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
cmake --build build --config Release -j$(nproc)

# Start server (Qwen3-8B GGUF Q4_K_M)
./build/bin/llama-server \
    -m /path/to/Qwen3-8B-Q4_K_M.gguf \
    --n-gpu-layers -1 --ctx-size 2048 \
    --port 8000 --api-key local

# Update config/settings.yaml: base_url: "http://localhost:8000/v1"
```

### 4. Ingest documents

```bash
python src/data_loader.py
python scripts/ingest_finqa.py
```

### 5. Start the API

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8080
```

### 6. Start the UI

```bash
streamlit run src/app.py
# Open http://localhost:8501
```

## Docker

```bash
echo "MODELSCOPE_API_KEY=your_key" > .env
docker compose up --build
```

## Evaluation

### RAGAS + Hit@K/MRR batch evaluation

```bash
# Using generated QA pairs, 10 RAGAS samples + 200 retrieval samples
python src/evaluator.py --n 10 --n-retrieval 200 --tag baseline --qa-pairs

# Compare multiple experiment runs
python scripts/compare_evals.py
```

Metrics tracked per run: `faithfulness`, `context_precision`, `answer_relevancy`, `hit@1/3/5`, `mrr@1/3/5`, per-phase latency, token counts.

### Generate QA pairs

```bash
python scripts/generate_qa.py --n 200 --sample-mode stratified
```

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
| `agent.max_retries` | `2` | Max reflection retries |
| `llm.base_url` | `http://localhost:8000/v1` | LLM endpoint (local or API) |

## Experiment Findings

| Experiment | Hit@5 | Faithfulness | Notes |
|-----------|-------|--------------|-------|
| baseline (alpha=0.4, ck=20) | 0.500 | 0.750 | Starting point |
| alpha=0.7, candidate_k=40 | 0.896 | 0.925 | +39.6% Hit@5 |
| recursive chunking | 0.710 | 0.763 | Worse than fixed for FinQA |
| semantic chunking | 0.715 | 0.749 | Marginal over fixed |
| metadata pre-filter (ticker/year) | 0.760 | 0.675 | Low ticker recall from query |
| summary-based 2-stage filter | 0.754 | 0.775 | -32% retrieval time, lower Hit@5 |
| **ticker-year header injection** | **0.967** | **1.000** | +8% Hit@5, +12% Faithfulness over prev best |

**Key insight**: FinQA documents never mention company ticker symbols in body text — only "the Company" or full names. Prepending a `[TICKER | YEAR]` header to every chunk at index time bridges this gap for both BM25 and dense retrieval, yielding the largest single improvement across all experiments.
