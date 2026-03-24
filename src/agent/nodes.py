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

from src.agent.state import AgentState
from src.agent.tools import search_internal
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
    model_kwargs={"extra_body": {"top_p": 0.8, "top_k": 20, "min_p": 0.0}},
)

_MAX_RETRIES = _agent_cfg.get("max_retries", 2)

# ---------------------------------------------------------------------------
# Planner Node
# ---------------------------------------------------------------------------

def planner_node(state: AgentState) -> dict:
    """
    决策是否需要检索。当前策略：始终检索。
    后续可扩展为：若 query 可直接回答则跳过检索。
    """
    # TODO: 同时记录开始时间，返回 start_time=time.time()
    return {"should_retrieve": True,
            "start_time": time.time()}

# ---------------------------------------------------------------------------
# Tool Node
# ---------------------------------------------------------------------------

def tool_node(state: AgentState) -> dict:
    """
    调用 search_internal 工具，将检索结果写入 state。
    """
    chunks = search_internal.invoke({"query": state["query"]})
    sources = list({chunk["doc_id"] for chunk in chunks if "doc_id" in chunk})

    return {"retrieved_chunks": chunks, "sources": sources}

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
