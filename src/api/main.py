"""
main.py
=======
FastAPI 服务：提供 Agent 查询（SSE 流式）和文档摄入两个端点。

端点：
  POST /query   — 调用 LangGraph Agent，SSE 流式返回进度 + 答案
  POST /ingest  — 接收文档文本，写入 ChromaDB + BM25

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
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException
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
# POST /ingest
# ---------------------------------------------------------------------------

@app.post("/ingest")
async def ingest_endpoint(req: IngestRequest):
    """
    接收文档文本，chunk 后写入 ChromaDB 和 BM25 索引。
    """
    try:
        chunks = ChunkManager().split(req.text, doc_id=req.doc_id)
        _vs.add_documents(chunks)
        existing = _bm25._chunks or []
        _bm25.build(existing + chunks)
        return {"doc_id": req.doc_id, "chunks": len(chunks)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
