# Structured RAG — FinQA Agentic RAG System

An end-to-end **Agentic RAG** system for structured financial documents (FinQA dataset), built to demonstrate the full AI Engineer stack.

## Architecture

```
User Query
    ↓
[FastAPI /query]  ── SSE streaming ──→  [Gradio UI]
    ↓
[LangGraph Agent]
    ├─ Planner Node    — decide whether to retrieve
    ├─ Tool Node       — search_internal (hybrid retrieval)
    ├─ Generator Node  — produce answer with context
    ├─ Reflector Node  — self-evaluate; retry ≤ 2 times
    └─ Final Node      — format output + source attribution
         ↓
[Hybrid Retriever]
    ├─ VectorStore     — ChromaDB + BGE-M3 dense vectors
    ├─ BM25Store       — rank-bm25 sparse index
    └─ Reranker        — BGE cross-encoder reranker
```

## Tech Stack

| Area | Choice |
|------|--------|
| Embedding | BAAI/bge-m3 |
| Vector store | ChromaDB |
| Sparse retrieval | rank-bm25 |
| Reranker | BAAI/bge-reranker-v2-m3 |
| Agent framework | LangGraph |
| LLM | Qwen3-8B via ModelScope API |
| Tracing | LangSmith |
| Evaluation | RAGAS + LLM-as-Judge |
| API | FastAPI + SSE streaming |
| Frontend | Gradio |

## Project Structure

```
rag_demo/
├── config/
│   └── settings.yaml          # Central configuration
├── data/
│   ├── finqa/docs/            # FinQA source documents (.md)
│   ├── chroma/                # ChromaDB persistent storage
│   └── bm25_index.pkl         # BM25 index
├── src/
│   ├── config.py              # Config loader
│   ├── data_loader.py         # FinQA dataset downloader
│   ├── chunk_manager.py       # Text chunking (fixed/recursive/semantic)
│   ├── vector_store.py        # ChromaDB + BGE-M3
│   ├── bm25_store.py          # BM25 sparse index
│   ├── reranker.py            # BGE cross-encoder reranker
│   ├── retriever.py           # Hybrid retrieval (custom / m3_hybrid)
│   ├── ingestion_registry.py  # Document ingestion tracker
│   ├── evaluator.py           # RAGAS batch evaluation
│   ├── llm_judge.py           # LLM-as-Judge per-query scoring
│   ├── app.py                 # Gradio frontend
│   ├── agent/
│   │   ├── state.py           # LangGraph AgentState
│   │   ├── tools.py           # search_internal tool
│   │   ├── nodes.py           # 5 agent nodes
│   │   └── graph.py           # LangGraph StateGraph
│   └── api/
│       └── main.py            # FastAPI endpoints
└── scripts/
    ├── ingest_finqa.py        # Batch document ingestion
    └── eval_smoke.py          # CI smoke test
```

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set environment variables

```bash
export MODELSCOPE_API_KEY=your_api_key_here
```

### 3. Download FinQA data

```bash
python src/data_loader.py
```

### 4. Ingest documents

```bash
python scripts/ingest_finqa.py
```

### 5. Start the API

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

### 6. Start the UI

```bash
python src/app.py
# Open http://localhost:7860
```

## Docker

```bash
echo "MODELSCOPE_API_KEY=your_key" > .env
docker compose up --build
```

## Evaluation

### RAGAS batch evaluation

```bash
python src/evaluator.py --n 50 --output data/eval_results.json
```

Metrics: `faithfulness`, `context_precision`, `answer_relevancy`

### CI smoke test

Runs automatically on every push/PR via GitHub Actions.

```bash
python scripts/eval_smoke.py --n 10 --top-k 5 --threshold 0.6
```

## Configuration

Key settings in `config/settings.yaml`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `retriever.mode` | `custom` | `custom` (BM25+dense) or `m3_hybrid` |
| `retriever.top_k` | `5` | Final top-k results |
| `retriever.custom.alpha` | `0.5` | Dense/sparse fusion weight (1.0=pure dense) |
| `retriever.custom.candidate_k` | `20` | Candidates before reranking |
| `agent.max_retries` | `2` | Max reflection retries |
| `llm.model` | `Qwen/Qwen3-8B` | LLM model name |
