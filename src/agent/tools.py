"""
tools.py
========
LangGraph Tool 定义：将 Retriever 包装为可供 Tool Node 调用的工具。

Usage:
    from src.agent.tools import tools
"""

from langchain_core.tools import tool

from src.bm25_store import BM25Store
from src.query_rewriter import QueryRewriter
from src.reranker import Reranker
from src.retriever import Retriever
from src.semantic_cache import SemanticCache
from src.vector_store import VectorStore

# 模块级单例：只初始化一次
_retriever = Retriever(
    vector_store=VectorStore(),
    bm25_store=BM25Store(),
    reranker=Reranker(),
    query_rewriter=QueryRewriter(),
    semantic_cache=SemanticCache(),
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


def offload_retrieval_models() -> None:
    """检索完成后释放 BGE-M3 + Reranker 的 GPU 显存，为 LLM 推理腾出空间。"""
    _retriever._vs.offload()
    _retriever._reranker.offload()


# 导出工具列表供 graph 使用
tools = [search_internal]
