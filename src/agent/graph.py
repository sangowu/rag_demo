"""
graph.py
========
LangGraph Agent 图组装：将节点连接为可运行的状态机。

图结构：
  START → planner
    ├─ should_retrieve=False → generator → reflector → final → END
    └─ should_retrieve=True  → tool
                                  ├─ grounding ok               → generator → reflector
                                  │                                               ├─ retry → tool
                                  │                                               └─ final → END
                                  └─ grounding weak
                                        ├─ TAVILY_API_KEY 已设置 → web_search
                                        │                               ├─ 有结果 → generator
                                        │                               └─ 无结果 → refuse → final → END
                                        └─ 无 TAVILY              → refuse → final → END  (拒答)

Usage:
    from src.agent.graph import graph
    result = graph.invoke({"query": "What was ADI revenue in 2009?", "retry_count": 0})
    print(result["final_answer"])
"""

import os

from langgraph.graph import END, START, StateGraph

from src.agent.nodes import (
    _GROUNDING_THRESHOLD,
    final_node,
    generator_node,
    planner_node,
    reflector_node,
    tool_node,
    web_search_node,
)
from src.agent.state import AgentState
from src.config import config

_MAX_RETRIES = config.get("agent", {}).get("max_retries", 2)


def _refuse_node(state: AgentState) -> dict:
    """grounding 不足时打标，交由 final_node 输出拒答消息。"""
    return {
        "refuse": True,
        "trace": [{
            "step": "refuse",
            "reason": f"grounding_score {state.get('grounding_score', 0):.3f} < threshold {_GROUNDING_THRESHOLD}",
        }],
    }


def _route_planner(state: AgentState) -> str:
    """planner 后路由：是否需要检索。"""
    if state.get("should_retrieve", True):
        return "tool"
    return "generator"


def _route_grounding(state: AgentState) -> str:
    """tool 后路由：证据充分 → generator；不足 → web_search（有 key）或 refuse。"""
    score = state.get("grounding_score", 0.0)
    if score >= _GROUNDING_THRESHOLD:
        return "generator"
    if os.environ.get("TAVILY_API_KEY"):
        return "web_search"
    return "refuse"


def _route_web_search(state: AgentState) -> str:
    """web_search 后路由：有结果 → generator；无结果 → refuse。"""
    if state.get("web_searched"):
        return "generator"
    return "refuse"


def _route_reflector(state: AgentState) -> str:
    """reflector 后路由：重试或结束。
    should_retrieve=False 或已走过 web_search 时不重试（重试无增益）。
    """
    if (
        state.get("should_retrieve", True)
        and not state.get("web_searched", False)
        and "bad" in state["reflection"].lower()
        and state["retry_count"] < _MAX_RETRIES
    ):
        return "retry"
    return "final"


# ---------------------------------------------------------------------------
# 构建图
# ---------------------------------------------------------------------------

builder = StateGraph(AgentState)

builder.add_node("planner", planner_node)
builder.add_node("tool", tool_node)
builder.add_node("web_search", web_search_node)
builder.add_node("refuse", _refuse_node)
builder.add_node("generator", generator_node)
builder.add_node("reflector", reflector_node)
builder.add_node("final", final_node)

builder.add_edge(START, "planner")

builder.add_conditional_edges(
    "planner",
    _route_planner,
    {"tool": "tool", "generator": "generator"},
)

builder.add_conditional_edges(
    "tool",
    _route_grounding,
    {
        "generator": "generator",
        "web_search": "web_search",  # grounding 不足 + TAVILY 可用 → 联网
        "refuse": "refuse",          # grounding 不足 + 无 TAVILY → 拒答
    },
)

builder.add_conditional_edges(
    "web_search",
    _route_web_search,
    {"generator": "generator", "refuse": "refuse"},
)

builder.add_edge("refuse", "final")

builder.add_edge("generator", "reflector")

builder.add_conditional_edges(
    "reflector",
    _route_reflector,
    {"retry": "tool", "final": "final"},
)

builder.add_edge("final", END)

# 编译图
graph = builder.compile()
