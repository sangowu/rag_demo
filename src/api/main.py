"""
main.py
=======
FastAPI 服务：提供 Agent 查询（SSE 流式）和文档摄入两个端点。

端点：
  POST /query          — 调用 LangGraph Agent，SSE 流式返回进度 + 答案
  POST /ingest         — 异步摄入文档，立即返回 job_id
  GET  /ingest/{job_id} — 查询摄入任务状态和进度

SSE 消息格式：
  {"event": "retrieved", "count": 5}
  {"event": "token",     "text": "ADI revenue..."}
  {"event": "reflection","text": "good: ..."}
  {"event": "done",      "answer": "...", "sources": [...]}
  {"event": "error",     "message": "..."}

Usage:
    uvicorn src.api.main:app --reload --port 8000
"""

import json
import os
import time
import uuid
from pathlib import Path
from typing import AsyncGenerator

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Request, Security
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from src.agent.graph import graph
from src.bm25_store import BM25Store
from src.chunk_manager import ChunkManager
from src.doc_metadata_extractor import build_chunk_header, extract_doc_metadata_llm
from src.document_loader import DocumentLoader
from src.guardrails import Guardrails
from src.logging_config import get_logger, setup_logging
from src.vector_store import VectorStore

setup_logging()
log = get_logger(__name__)

# ── 限流：每 IP 每分钟最多 30 次查询，每分钟最多 10 次摄入 ──
limiter = Limiter(key_func=get_remote_address)

app = FastAPI(title="Structured RAG API", version="0.1.0")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ── Bearer Token 认证 ──────────────────────────────────────
_bearer = HTTPBearer(auto_error=False)
_API_KEY = os.getenv("RAG_API_KEY", "")   # 未设置时跳过认证（开发模式）

def _require_auth(
    credentials: HTTPAuthorizationCredentials | None = Security(_bearer),
) -> None:
    """验证 Bearer token。RAG_API_KEY 未设置时跳过（本地开发）。"""
    if not _API_KEY:
        return
    if credentials is None or credentials.credentials != _API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")

_STATIC_DIR = Path(__file__).parent / "static"
_STATIC_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


@app.get("/")
async def index():
    return FileResponse(str(_STATIC_DIR / "index.html"))


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

@app.get("/health", tags=["ops"])
async def health():
    """
    健康检查端点，供 Docker / K8s liveness probe 使用。

    返回：
      status       : "ok"
      vector_store : ChromaDB 中的 chunk 总数
      bm25_ready   : BM25 索引是否已加载
    """
    try:
        chunk_count = _vs._collection.count()
        bm25_ready  = _bm25._index is not None
    except Exception as e:
        log.error("health check failed", error=str(e))
        return JSONResponse(status_code=503, content={"status": "degraded", "error": str(e)})

    return {
        "status":       "ok",
        "vector_store": chunk_count,
        "bm25_ready":   bm25_ready,
    }

# 摄入用单例（query 路径不需要直接访问这两个）
_vs = VectorStore()
_bm25 = BM25Store()
_guardrails = Guardrails()

# 异步任务状态表：job_id → {status, progress, total, error}
# 生产环境可替换为 Redis
_jobs: dict[str, dict] = {}


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    query: str
    messages: list[dict] | None = None  # [{"role":"user","content":"..."}]


class IngestRequest(BaseModel):
    doc_id: str | None = None   # 不填时自动生成
    text: str | None = None     # 直接提供文本
    url:  str | None = None     # 或提供 URL，服务端自动抓取


# ---------------------------------------------------------------------------
# SSE helper
# ---------------------------------------------------------------------------

def _sse(event: str, **kwargs) -> str:
    """将事件序列化为 SSE 格式的单条消息。"""
    return f"data: {json.dumps({'event': event, **kwargs})}\n\n"


# ---------------------------------------------------------------------------
# POST /query
# ---------------------------------------------------------------------------

@app.post("/query", dependencies=[Depends(_require_auth)])
@limiter.limit("30/minute")
async def query_endpoint(request: Request, req: QueryRequest):
    """
    调用 LangGraph Agent，以 SSE 流式推送执行进度和最终答案。
    """
    t0 = time.time()
    log.info("query received", query=req.query[:80])

    # 输入 guardrail：非金融问题直接拒绝，不进入 agent
    ok, reason = _guardrails.check_input(req.query)
    if not ok:
        log.warning("query blocked by guardrails", reason=reason)
        async def _blocked():
            yield _sse("error", message=reason)
        return StreamingResponse(_blocked(), media_type="text/event-stream")

    async def event_stream() -> AsyncGenerator[str, None]:
        try:
            # 实时 token 计数（tiktoken 估算，无需等待 response 结束）
            try:
                import tiktoken
                _enc = tiktoken.get_encoding("cl100k_base")
                def _count(text: str) -> int:
                    return len(_enc.encode(text))
            except Exception:
                def _count(text: str) -> int:  # fallback: 字符数 / 4
                    return max(1, len(text) // 4)

            _stream_tokens = 0

            history = []
            if req.messages:
                for m in req.messages:
                    if m["role"] == "user":
                        history.append(HumanMessage(content=m["content"]))
                    elif m["role"] == "assistant":
                        history.append(AIMessage(content=m["content"]))
            initial_state = {
                "query": req.query,
                "retry_count": 0,
                "retrieved_chunks": [],
                "sources": [],
                "messages": history,
                "prompt_tokens": 0,
                "completion_tokens": 0,
            }
            for chunk in graph.stream(
                initial_state,
                stream_mode=["updates", "messages"],
            ):
                mode, data = chunk
                if mode == "updates":
                    node = list(data.keys())[0]
                    if node == "tool":
                        chunks = data[node].get("retrieved_chunks", [])
                        rewritten = data[node].get("rewritten_query", "")
                        yield _sse("retrieved", count=len(chunks), rewritten_query=rewritten)
                    elif node == "reflector":
                        reflection = data[node].get("reflection", "")
                        yield _sse("reflection", text=reflection)
                    elif node == "final":
                        node_data = data[node]
                        answer = node_data.get("final_answer", "")
                        chunks = node_data.get("retrieved_chunks", [])
                        grounded, grounding_warning = _guardrails.check_output(answer, chunks)
                        yield _sse(
                            "done",
                            answer=answer,
                            sources=node_data.get("sources", []),
                            metrics=node_data.get("metrics", {}),
                            grounding_warning=None if grounded else grounding_warning,
                        )
                elif mode == "messages":
                    msg_chunk, metadata = data
                    if metadata.get("langgraph_node") == "generator" and msg_chunk.content:
                        _stream_tokens += _count(msg_chunk.content)
                        yield _sse("token", text=msg_chunk.content,
                                   tokens_so_far=_stream_tokens)
        except Exception as e:
            log.error("agent error", error=str(e), query=req.query[:80])
            yield _sse("error", message=str(e))

    log.info("query completed", latency_ms=round((time.time() - t0) * 1000))
    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# Background task
# ---------------------------------------------------------------------------

def _run_ingest(job_id: str, doc_id: str, text: str, source_type: str = "text") -> None:
    """
    后台执行文档摄入：
      1. LLM 提取文档级元信息（company_name / ticker / year）
      2. 构造 chunk header（通用，不依赖文件名）
      3. chunk → ChromaDB → BM25

    通过 _jobs[job_id] 向外暴露进度，客户端轮询 GET /ingest/{job_id} 获取。
    """
    try:
        # Step 1: LLM 提取元信息（只读前 2000 字，成本低）
        meta   = extract_doc_metadata_llm(text)
        header = build_chunk_header(meta)
        _jobs[job_id]["meta"] = meta

        # Step 2: chunk（ChunkManager 负责切分，header 注入在 VectorStore/BM25 层）
        chunks = ChunkManager(
            semantic_embeddings=_vs.langchain_dense_embeddings(),
        ).split(text, doc_id=doc_id)
        total  = len(chunks)
        _jobs[job_id]["total"] = total

        # Step 3: 将 header 注入每个 chunk 的 text（供 embed 和 BM25 使用）
        if header:
            for c in chunks:
                c["_header"] = header   # 标记，vector_store 检测并使用

        # 写入 ChromaDB
        _vs.add_documents(chunks, header_override=header if header else None)
        _jobs[job_id]["progress"] = total

        # 重建 BM25
        existing = _bm25._chunks or []
        _bm25.build(existing + chunks, header_override={doc_id: header} if header else None)

        _jobs[job_id]["status"] = "done"
        log.info("ingest job done", job_id=job_id, doc_id=doc_id, chunks=total)

    except Exception as e:
        _jobs[job_id]["status"] = "failed"
        _jobs[job_id]["error"] = str(e)
        log.error("ingest job failed", job_id=job_id, doc_id=doc_id, error=str(e))


# ---------------------------------------------------------------------------
# POST /ingest
# ---------------------------------------------------------------------------

@app.post("/ingest", status_code=202, dependencies=[Depends(_require_auth)])
@limiter.limit("10/minute")
async def ingest_endpoint(request: Request, req: IngestRequest, background_tasks: BackgroundTasks):
    """
    异步摄入文档，支持多来源：
      - 直接提供 text
      - 提供 url，服务端自动抓取正文

    LLM 自动提取 company_name / year，注入 chunk header。
    用 GET /ingest/{job_id} 轮询进度。
    """
    loader = DocumentLoader()

    if req.url:
        # URL 来源：抓取后异步处理
        try:
            doc = loader.load_url(req.url, doc_id=req.doc_id)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to fetch URL: {e}")
        text        = doc["text"]
        doc_id      = doc["doc_id"]
        source_type = "url"
    elif req.text:
        text        = req.text
        doc_id      = req.doc_id or uuid.uuid4().hex[:8]
        source_type = "text"
    else:
        raise HTTPException(status_code=422, detail="Provide either 'text' or 'url'.")

    job_id = uuid.uuid4().hex[:8]
    _jobs[job_id] = {"status": "running", "progress": 0, "total": 0,
                     "doc_id": doc_id, "source_type": source_type}
    background_tasks.add_task(_run_ingest, job_id, doc_id, text, source_type)
    log.info("ingest job queued", job_id=job_id, doc_id=doc_id, source=source_type)
    return {"job_id": job_id, "status": "queued", "doc_id": doc_id}


# ---------------------------------------------------------------------------
# GET /ingest/{job_id}
# ---------------------------------------------------------------------------

@app.get("/ingest/{job_id}")
async def ingest_status(job_id: str):
    """
    查询摄入任务状态。

    Returns:
        status   : "running" | "done" | "failed"
        progress : 已处理 chunk 数
        total    : 总 chunk 数
        error    : 失败原因（仅 failed 时有）
    """
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found")
    return job
