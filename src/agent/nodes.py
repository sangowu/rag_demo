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

import os
import time

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from src.agent.state import AgentState
from src.agent.tools import _retriever, offload_retrieval_models
from src.config import config

_llm_cfg = config.get("llm", {})
_agent_cfg = config.get("agent", {})

# 模块级 LLM 单例
_llm = ChatOpenAI(
    model=_llm_cfg.get("model", "Qwen/Qwen3-8B"),
    base_url=_llm_cfg.get("base_url", "http://localhost:8001/v1"),
    api_key=os.environ.get(_llm_cfg.get("api_key_env", "MODELSCOPE_API_KEY"), "local"),
    temperature=_llm_cfg.get("temperature", 0.1),
    max_tokens=_llm_cfg.get("max_tokens", 1024),
    extra_body={"enable_thinking": False},
)

_MAX_RETRIES = _agent_cfg.get("max_retries", 2)

# ---------------------------------------------------------------------------
# Planner Node
# ---------------------------------------------------------------------------

class _PlannerDecision(BaseModel):
    should_retrieve: bool = Field(description="Whether retrieval from documents is needed to answer the query")
    should_rewrite: bool = Field(description="Whether the query contains ticker symbols or ambiguous company names that need expansion (e.g. ADI, JPM, C)")
    reasoning: str = Field(description="One sentence explaining the decisions")


_planner_llm = _llm.with_structured_output(_PlannerDecision)

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
    try:
        decision = _planner_llm.invoke(_PLANNER_PROMPT.format(query=state["query"]))
        return {
            "should_retrieve": decision.should_retrieve,
            "should_rewrite": decision.should_rewrite,
            "start_time": time.time(),
        }
    except Exception:
        # 结构化输出失败时降级为始终检索、不改写
        return {"should_retrieve": True, "should_rewrite": False, "start_time": time.time()}

# ---------------------------------------------------------------------------
# Tool Node
# ---------------------------------------------------------------------------

def tool_node(state: AgentState) -> dict:
    """
    执行混合检索，将结果写入 state。
    由 planner 决定是否启用 query rewriting（should_rewrite）。
    检索完成后释放 BGE-M3 + Reranker 显存，为 LLM 推理腾出空间。
    """
    query = state["query"]
    should_rewrite = state.get("should_rewrite", False)

    chunks = _retriever.search(query, rewrite=should_rewrite)
    offload_retrieval_models()
    sources = list({chunk["doc_id"] for chunk in chunks if "doc_id" in chunk})

    # 记录改写后的 query 供 LangSmith 可观测（未改写时与原始相同）
    rewritten = _retriever._rewriter._cache.get(query, query) if should_rewrite else query

    return {"retrieved_chunks": chunks, "sources": sources, "rewritten_query": rewritten}

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
        "Be concise and precise. If the context does not contain enough information, say so."
    )

    human_prompt = f"Context:\n{context}\n\nQuestion: {state['query']}"

    history = state.get("messages", [])
    response = _llm.invoke([
        SystemMessage(content=system_prompt),
        *history,
        HumanMessage(content=human_prompt),
    ])

    usage = response.response_metadata.get("token_usage", {})

    return {
        "answer": response.content,
        "messages": [HumanMessage(state["query"]), response],
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0)
    }


# ---------------------------------------------------------------------------
# Reflector Node
# ---------------------------------------------------------------------------

def reflector_node(state: AgentState) -> dict:
    """
    用 LLM 评估答案质量。若质量低且未超过重试上限，触发重试。
    """
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
    reflection = _llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=eval_prompt)
    ]).content

    if "bad" in reflection.lower() and state["retry_count"] < _MAX_RETRIES:
        return {"reflection": reflection, "retry_count": state["retry_count"] + 1}
    else:
        return {"reflection": reflection}

# ---------------------------------------------------------------------------
# Final Node
# ---------------------------------------------------------------------------

def final_node(state: AgentState) -> dict:
    """
    格式化最终输出，附加引用来源，汇总 metrics。
    """
    answer, sources = state["answer"], state["sources"]
    final_answer = answer + "\n\nSources: " + ", ".join(sources)

    latency_ms = round((time.time() - state["start_time"]) * 1000)
    metrics = {
        "latency_ms": latency_ms,
        "prompt_tokens": state.get("prompt_tokens", 0),
        "completion_tokens": state.get("completion_tokens", 0),
        "retry_count": state["retry_count"],
    }
    
    return {"final_answer": final_answer,
            "metrics": metrics}
