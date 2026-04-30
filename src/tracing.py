"""
tracing.py
==========
Langfuse 可观测性集成（直接 REST API，不依赖 SDK 队列机制）。

当 LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY 存在时自动启用，否则静默跳过。

Usage:
    from src.tracing import record_agent_trace
    result = graph.invoke(state)
    record_agent_trace(query, result)
"""

import base64
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env", override=False)
except ImportError:
    pass


def _get_creds() -> tuple[str, str] | None:
    pub = os.getenv("LANGFUSE_PUBLIC_KEY", "")
    sec = os.getenv("LANGFUSE_SECRET_KEY", "")
    if pub and sec:
        return pub, sec
    return None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def record_agent_trace(query: str, state: dict, run_name: str = "rag_query") -> None:
    """将 AgentState 结果上报为一条 Langfuse trace，含每个节点的 span。"""
    creds = _get_creds()
    if creds is None:
        return

    import requests

    pub, sec = creds
    host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
    auth = base64.b64encode(f"{pub}:{sec}".encode()).decode()
    headers = {"Authorization": f"Basic {auth}", "Content-Type": "application/json"}

    metrics   = state.get("metrics", {})
    node_trace = metrics.get("trace", state.get("trace", []))
    trace_id  = str(uuid.uuid4())
    ts        = _now()

    batch = [
        {
            "id":        str(uuid.uuid4()),
            "type":      "trace-create",
            "timestamp": ts,
            "body": {
                "id":       trace_id,
                "name":     run_name,
                "input":    {"query": query},
                "output":   {"answer": state.get("final_answer", "")},
                "metadata": {
                    "grounding_score": state.get("grounding_score"),
                    "refused":         state.get("refuse", False),
                    "web_searched":    state.get("web_searched", False),
                    "latency_ms":      metrics.get("latency_ms"),
                    "retry_count":     metrics.get("retry_count", 0),
                    "sources":         state.get("sources", []),
                },
            },
        }
    ]

    for step in node_trace:
        step = dict(step)
        name = step.pop("step", "unknown")
        batch.append({
            "id":        str(uuid.uuid4()),
            "type":      "span-create",
            "timestamp": ts,
            "body": {
                "id":       str(uuid.uuid4()),
                "traceId":  trace_id,
                "name":     name,
                "input":    step,
                "output":   step,
                "startTime": ts,
                "endTime":   ts,
            },
        })

    try:
        requests.post(
            f"{host}/api/public/ingestion",
            headers=headers,
            json={"batch": batch},
            timeout=10,
        )
    except Exception:
        pass


# 兼容旧调用
def get_callbacks() -> list:
    return []
