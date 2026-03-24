"""
graph.py
========
LangGraph Agent 图组装：将 5 个节点连接为可运行的状态机。

图结构：
  START → planner → tool → generator → reflector
                              ↑              ↓ (条件边)
                              └─── retry ────┤
                                             └─── final → END

Usage:
    from src.agent.graph import graph
    result = graph.invoke({"query": "What was ADI revenue in 2009?", "retry_count": 0})
    print(result["final_answer"])
"""

from langgraph.graph import END, START, StateGraph

from src.agent.nodes import (
    final_node,
    generator_node,
    planner_node,
    reflector_node,
    tool_node,
)
from src.agent.state import AgentState
from src.config import config

_MAX_RETRIES = config.get("agent", {}).get("max_retries", 2)


def _should_retry(state: AgentState) -> str:
    """
    条件路由函数：决定 reflector 之后走重试还是结束。

    Returns:
        "retry" → 回到 tool_node 重新检索
        "final" → 进入 final_node 输出结果
    """
    if "bad" in state["reflection"].lower() and state["retry_count"] < _MAX_RETRIES:
        return "retry"
    else:
        return "final"


# ---------------------------------------------------------------------------
# 构建图
# ---------------------------------------------------------------------------

builder = StateGraph(AgentState)

builder.add_node("planner", planner_node)
builder.add_node("tool", tool_node)                                                                                  
builder.add_node("generator", generator_node)                                                                           
builder.add_node("reflector", reflector_node)
builder.add_node("final", final_node)

builder.add_edge(START, "planner")
builder.add_edge("planner", "tool")                                                                                            
builder.add_edge("tool", "generator")                                                                                          
builder.add_edge("generator", "reflector")

builder.add_conditional_edges(
    "reflector",
    _should_retry,
    {"retry": "tool", "final": "final"},
)

builder.add_edge("final", END)

# 编译图
graph = builder.compile()
