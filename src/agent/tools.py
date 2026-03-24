"""
tools.py
========
LangGraph Tool 定义：将 Retriever 包装为可供 Tool Node 调用的工具。

Usage:
    from src.agent.tools import tools
"""

from langchain_core.tools import tool

from src.bm25_store import BM25Store
from src.reranker import Reranker
from src.retriever import Retriever
from src.vector_store import VectorStore

# 模块级单例：只初始化一次
_retriever = Retriever(
    vector_store=VectorStore(),
    bm25_store=BM25Store(),
    reranker=Reranker(),
)


@tool
def search_internal(query: str) -> list[dict]:
    """Search internal FinQA documents and return relevant chunks.

    Args:
        query: The user's question to search for.

    Returns:
        List of relevant document chunks with text and metadata.
    """
    return _retriever.search(query)


# 导出工具列表供 graph 使用
tools = [search_internal]
