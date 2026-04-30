"""
nodes.py
========
LangGraph Agent 的 5 个节点函数。

节点签名统一为：fn(state: AgentState) -> dict
返回值是要更新的字段（增量），LangGraph 自动 merge 到 State。

节点列表：
  - planner_node   : 决策是否需要检索
  - tool_node      : 调用 search_internal 执行检索
  - generator_node : 用 LLM + chunks 生成答案
  - reflector_node : 用 LLM 评估答案质量，决定是否重试
  - final_node     : 格式化最终输出，附加引用来源
"""

import time
import warnings

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

warnings.filterwarnings("ignore", message="Pydantic serializer warnings", category=UserWarning)

from src.agent.state import AgentState
from src.agent.tools import _retriever, offload_retrieval_models
from src.config import config
from src.llm_factory import get_llm

_agent_cfg = config.get("agent", {})

# 模块级 LLM 单例
_llm = get_llm()

_MAX_RETRIES = _agent_cfg.get("max_retries", 2)
_GROUNDING_THRESHOLD = _agent_cfg.get("grounding_threshold", 0.3)


def _to_str(content) -> str:
    """Gemini 响应的 content 有时是 list[part]，part 可能是 dict{'type','text'} 或字符串。"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, dict):
                parts.append(p.get("text", str(p)))
            else:
                parts.append(str(p))
        return " ".join(parts)
    if isinstance(content, dict):
        return content.get("text", str(content))
    return str(content)

# ---------------------------------------------------------------------------
# Planner Node
# ---------------------------------------------------------------------------

class PlannerDecision(BaseModel):
    should_retrieve: bool = Field(description="Whether retrieval from documents is needed to answer the query")
    should_rewrite: bool = Field(description="Whether the query contains ticker symbols or ambiguous company names that need expansion (e.g. ADI, JPM, C)")
    reasoning: str = Field(description="One sentence explaining the decisions")


_planner_llm = _llm.with_structured_output(PlannerDecision)

_PLANNER_PROMPT = """\
You are a planning agent for a financial document QA system.
Analyze the user query and make two decisions:

1. should_retrieve: Does this query require looking up information from financial documents?
   - True for most financial questions (revenue, earnings, ratios, etc.)
   - False only if the query is a greeting, meta-question, or answerable from common knowledge

2. should_rewrite: Does the query contain stock ticker symbols or ambiguous short company names that should be expanded?
   - True if you see tickers like ADI, JPM, C, GS, MS, BAC, WFC, AAPL, etc.
   - True if the company reference is ambiguous or abbreviated
   - False if the full company name is already used

Query: {query}"""


def planner_node(state: AgentState) -> dict:
    """
    决策是否需要检索，以及是否需要 query rewriting。
    用 structured output 输出 JSON，同时记录 reasoning 供 LangSmith 可观测。
    """
    t0 = time.time()
    try:
        decision = _planner_llm.invoke(_PLANNER_PROMPT.format(query=state["query"]))
        entry = {
            "step": "planner",
            "should_retrieve": decision.should_retrieve,
            "should_rewrite": decision.should_rewrite,
            "reasoning": decision.reasoning,
        }
        return {
            "should_retrieve": decision.should_retrieve,
            "should_rewrite": decision.should_rewrite,
            "start_time": t0,
            "trace": [entry],
        }
    except Exception as e:
        entry = {
            "step": "planner",
            "should_retrieve": True,
            "should_rewrite": False,
            "reasoning": f"structured output failed ({e}), defaulting to retrieve=True",
        }
        return {"should_retrieve": True, "should_rewrite": False, "start_time": t0, "trace": [entry]}

# ---------------------------------------------------------------------------
# Tool Node
# ---------------------------------------------------------------------------

def tool_node(state: AgentState) -> dict:
    """
    执行混合检索，将结果写入 state。
    由 planner 决定是否启用 query rewriting（should_rewrite）。
    检索完成后计算 grounding_score（top chunk 的 reranker score）。
    """
    query = state["query"]
    should_rewrite = state.get("should_rewrite", False)

    raw = _retriever.search(query, rewrite=should_rewrite)
    chunks = [r.to_dict() for r in raw]
    offload_retrieval_models()
    sources = list({c["doc_id"] for c in chunks if c.get("doc_id")})

    rewritten = (
        _retriever._rewriter._cache.get(query, query)
        if should_rewrite and _retriever._rewriter is not None
        else query
    )

    grounding_score = max((c.get("score", 0.0) for c in chunks), default=0.0)
    grounding_ok = grounding_score >= _GROUNDING_THRESHOLD

    entry = {
        "step": "tool",
        "rewritten_query": rewritten if rewritten != state["query"] else None,
        "chunks_retrieved": len(chunks),
        "grounding_score": round(grounding_score, 4),
        "grounding_ok": grounding_ok,
        "decision": "proceed" if grounding_ok else f"refuse (score {grounding_score:.3f} < threshold {_GROUNDING_THRESHOLD})",
    }

    return {
        "retrieved_chunks": chunks,
        "sources": sources,
        "rewritten_query": rewritten,
        "grounding_score": grounding_score,
        "trace": [entry],
    }

# ---------------------------------------------------------------------------
# Generator Node
# ---------------------------------------------------------------------------

def generator_node(state: AgentState) -> dict:
    """
    用 LLM 结合检索到的 chunks 生成答案。
    """
    chunks = state["retrieved_chunks"]
    context = "\n\n".join(f"[{i+1}] {c['text']}" for i, c in enumerate(chunks))

    system_prompt = (
        "You are a financial analyst assistant. "
        "Answer the question using ONLY the provided context. "
        "Be concise and precise. Lead with the direct answer, then provide supporting evidence. "
        "If the context does not contain enough information, say so."
    )

    human_prompt = f"Context:\n{context}\n\nQuestion: {state['query']}"

    history = state.get("messages", [])
    response = _llm.invoke([
        SystemMessage(content=system_prompt),
        *history,
        HumanMessage(content=human_prompt),
    ])

    usage = response.response_metadata.get("usage_metadata", {})
    return {
        "answer": _to_str(response.content),
        "messages": [HumanMessage(state["query"]), response],
        "prompt_tokens": usage.get("prompt_token_count", 0),
        "completion_tokens": usage.get("candidates_token_count", 0),
    }


# ---------------------------------------------------------------------------
# Reflector Node
# ---------------------------------------------------------------------------

def reflector_node(state: AgentState) -> dict:
    """
    用 LLM 评估答案质量。若质量低且未超过重试上限，触发重试。
    若已标记 refuse（grounding 不足），跳过评估直接通过。
    """
    if state.get("refuse"):
        return {
            "reflection": "refused: insufficient evidence",
            "trace": [{"step": "reflector", "decision": "skipped (refused)"}],
        }

    system_prompt = (
        "You are a strict answer quality evaluator. "
        "Given a question and an answer, reply with ONLY 'good' or 'bad' followed by a brief reason. "
        "'good' means the answer is accurate, grounded, and directly addresses the question. "
        "'bad' means the answer is vague, incorrect, or does not address the question."
    )
    eval_prompt = (
        f"Question: {state['query']}\n\n"
        f"Answer: {state['answer']}\n\n"
        "Evaluate the answer quality. Reply with 'good' or 'bad' and a brief reason."
    )
    reflection = _to_str(_llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=eval_prompt)
    ]).content)

    will_retry = (
        state.get("should_retrieve", True)
        and not state.get("web_searched", False)
        and "bad" in reflection.lower()
        and state["retry_count"] < _MAX_RETRIES
    )
    entry = {
        "step": "reflector",
        "verdict": "bad" if "bad" in reflection.lower() else "good",
        "reflection": reflection[:120],
        "decision": f"retry (attempt {state['retry_count'] + 1})" if will_retry else "accept",
    }
    if will_retry:
        return {"reflection": reflection, "retry_count": state["retry_count"] + 1, "trace": [entry]}
    return {"reflection": reflection, "trace": [entry]}

# ---------------------------------------------------------------------------
# Web Search Node
# ---------------------------------------------------------------------------

def web_search_node(state: AgentState) -> dict:
    """本地检索 grounding 不足时，调用 Tavily 联网检索作为 fallback。"""
    from src.agent.tools import search_web

    query = state.get("rewritten_query") or state["query"]
    results: list[dict] = search_web.invoke({"query": query, "top_k": 3})

    valid = [r for r in results
             if not r["text"].startswith("Web search unavailable")
             and not r["text"].startswith("Web search error:")]
    sources = [r["source"] for r in valid if r.get("source")]

    entry = {
        "step": "web_search",
        "query": query,
        "results": len(valid),
        "ok": bool(valid),
    }

    if valid:
        return {
            "retrieved_chunks": valid,
            "sources": sources,
            "web_searched": True,
            "trace": [entry],
        }
    return {"web_searched": False, "trace": [entry]}


# ---------------------------------------------------------------------------
# Final Node
# ---------------------------------------------------------------------------

def final_node(state: AgentState) -> dict:
    """
    格式化最终输出，附加引用来源，汇总 metrics。
    若 refuse=True，输出拒答消息而非生成答案。
    """
    latency_ms = round((time.time() - state["start_time"]) * 1000)
    metrics = {
        "latency_ms": latency_ms,
        "prompt_tokens": state.get("prompt_tokens", 0),
        "completion_tokens": state.get("completion_tokens", 0),
        "retry_count": state["retry_count"],
        "grounding_score": state.get("grounding_score", None),
        "refused": state.get("refuse", False),
        "trace": state.get("trace", []),
    }

    if state.get("refuse"):
        final_answer = (
            "I was unable to find sufficient evidence in the document library to answer this question. "
            f"(grounding score: {state.get('grounding_score', 0.0):.3f} < threshold {_GROUNDING_THRESHOLD:.3f})\n\n"
            "Please try rephrasing your question or check if the relevant documents have been indexed."
        )
        return {"final_answer": final_answer, "sources": [], "metrics": metrics}

    all_sources = list(dict.fromkeys(state["sources"]))
    if state.get("web_searched"):
        sources = [s for s in all_sources if s.startswith("http")]
    else:
        sources = all_sources
    final_answer = state["answer"] + "\n\nSources: " + ", ".join(sources)
    return {"final_answer": final_answer, "sources": sources, "metrics": metrics}
