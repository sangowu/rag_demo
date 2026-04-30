"""
state.py
========
LangGraph Agent 共享状态定义。

所有节点（Planner、Tool、Generator、Reflector、Final）共享同一个 AgentState。
每个节点读取 State、修改部分字段、返回增量更新字典。

Usage:
    from src.agent.state import AgentState
"""

import operator
from typing import Annotated, TypedDict
from typing_extensions import NotRequired
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    # 用户原始问题（初始化必填）
    query: str

    # 已重试次数（初始化必填，默认 0）
    retry_count: int

    # 检索到的 chunks，operator.add 启用列表追加语义（初始化必填，默认 []）
    retrieved_chunks: Annotated[list[dict], operator.add]

    # 引用来源，operator.add 启用列表追加语义（初始化必填，默认 []）
    sources: Annotated[list[str], operator.add]

    messages: Annotated[list[BaseMessage], add_messages]

    # token 用量，operator.add 跨重试累加（初始化必填，默认 0）
    prompt_tokens: Annotated[int, operator.add]
    completion_tokens: Annotated[int, operator.add]

    # 以下字段由各节点在运行时填充，初始化时不需要提供
    should_retrieve: NotRequired[bool]
    should_rewrite: NotRequired[bool]  # planner 决策：是否需要 query rewriting
    rewritten_query: NotRequired[str]  # query rewriter 改写后的 query，用于 LangSmith 可观测
    grounding_score: NotRequired[float]  # 检索结果最高 reranker score，用于 grounding check
    refuse: NotRequired[bool]            # True → 证据不足，输出拒答
    web_searched: NotRequired[bool]      # True → 已执行联网检索
    trace: Annotated[list[dict], operator.add]  # 各节点决策日志，追加语义
    answer: NotRequired[str]
    reflection: NotRequired[str]
    final_answer: NotRequired[str]
    start_time: NotRequired[float]   # planner 记录，final 计算延迟
    metrics: NotRequired[dict]       # final 汇总输出
