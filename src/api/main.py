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
import uuid
from typing import AsyncGenerator

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage

from src.agent.graph import graph
from src.bm25_store import BM25Store
from src.chunk_manager import ChunkManager
from src.vector_store import VectorStore

app = FastAPI(title="Structured RAG API", version="0.1.0")

# 摄入用单例（query 路径不需要直接访问这两个）
_vs = VectorStore()
_bm25 = BM25Store()

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
    doc_id: str
    text: str          # 原始文档文本，API 内部负责 chunk


# ---------------------------------------------------------------------------
# SSE helper
# ---------------------------------------------------------------------------

def _sse(event: str, **kwargs) -> str:
    """将事件序列化为 SSE 格式的单条消息。"""
    return f"data: {json.dumps({'event': event, **kwargs})}\n\n"


# ---------------------------------------------------------------------------
# POST /query
# ---------------------------------------------------------------------------

@app.post("/query")
async def query_endpoint(req: QueryRequest):
    """
    调用 LangGraph Agent，以 SSE 流式推送执行进度和最终答案。
    """
    async def event_stream() -> AsyncGenerator[str, None]:
        try:
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
                        yield _sse("retrieved", count=len(chunks))
                    elif node == "reflector":
                        reflection = data[node].get("reflection", "")
                        yield _sse("reflection", text=reflection)
                    elif node == "final":
                        node_data = data[node]
                        yield _sse(
                            "done",
                            answer=node_data.get("final_answer", ""),
                            sources=node_data.get("sources", []),
                            metrics=node_data.get("metrics", {}),
                        )
                elif mode == "messages":
                    msg_chunk, metadata = data
                    if metadata.get("langgraph_node") == "generator" and msg_chunk.content:
                        yield _sse("token", text=msg_chunk.content)
        except Exception as e:
            yield _sse("error", message=str(e))

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# Background task
# ---------------------------------------------------------------------------

def _run_ingest(job_id: str, doc_id: str, text: str) -> None:
    """
    后台执行文档摄入：chunk → ChromaDB → BM25。
    通过 _jobs[job_id] 向外暴露进度，客户端轮询 GET /ingest/{job_id} 获取。
    """
    try:
        chunks = ChunkManager().split(text, doc_id=doc_id)
        total = len(chunks)
        _jobs[job_id]["total"] = total

        # 写入 ChromaDB（内部已按 embed_batch_size 分批）
        _vs.add_documents(chunks)
        _jobs[job_id]["progress"] = total

        # 重建 BM25
        existing = _bm25._chunks or []
        _bm25.build(existing + chunks)

        _jobs[job_id]["status"] = "done"

    except Exception as e:
        _jobs[job_id]["status"] = "failed"
        _jobs[job_id]["error"] = str(e)


# ---------------------------------------------------------------------------
# POST /ingest
# ---------------------------------------------------------------------------

@app.post("/ingest", status_code=202)
async def ingest_endpoint(req: IngestRequest, background_tasks: BackgroundTasks):
    """
    异步摄入文档。立即返回 job_id，实际处理在后台进行。
    用 GET /ingest/{job_id} 轮询进度。
    """
    job_id = uuid.uuid4().hex[:8]
    _jobs[job_id] = {"status": "running", "progress": 0, "total": 0}
    background_tasks.add_task(_run_ingest, job_id, req.doc_id, req.text)
    return {"job_id": job_id, "status": "queued"}


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
