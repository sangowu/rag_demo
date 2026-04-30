"""
tools.py
========
Agentic RAG 的工具定义。Agent 自主决定调用哪些工具、调用顺序。

工具列表：
  search_local   — 本地文档库混合检索（dense + BM25 + reranker）
  rewrite_query  — LLM 改写 query（含 ticker 展开、歧义消解）
  search_web     — 联网检索（Tavily），处理时效性问题

设计原则：
  - 每个工具职责单一，agent 自主组合
  - 全部返回统一的 list[dict]（由 RetrievalResult.to_dict() 产出）
  - rewrite_query 附带 reason 字段，决策过程可追溯
"""

from langchain_core.tools import tool

from src.bm25_store import BM25Store
from src.reranker import Reranker
from src.retriever import Retriever
from src.schemas import RetrievalResult
from src.semantic_cache import SemanticCache
from src.vector_store import VectorStore

# ── 模块级单例：只初始化一次，不再内置 QueryRewriter（已拆为独立 tool）──────
_retriever = Retriever(
    vector_store=VectorStore(),
    bm25_store=BM25Store(),
    reranker=Reranker(),
    semantic_cache=SemanticCache(),
)


# ─────────────────────────────────────────────────────────────────────────────
# Tool 1: 本地文档检索
# ─────────────────────────────────────────────────────────────────────────────

@tool
def search_local(query: str, top_k: int = 5) -> list[dict]:
    """Search the internal financial and legal document library.

    Use this when the question is about company financials, SEC filings,
    earnings reports, or legal contracts that are in the local knowledge base.

    Args:
        query: The search query (use the rewritten query if rewrite_query was called).
        top_k: Number of results to return (default 5).

    Returns:
        List of relevant chunks with text, source, score, and metadata.
    """
    results: list[RetrievalResult] = _retriever.search(query, top_k=top_k)
    return [r.to_dict() for r in results]


# ─────────────────────────────────────────────────────────────────────────────
# Tool 2: Query 改写
# ─────────────────────────────────────────────────────────────────────────────

@tool
def rewrite_query(query: str, reason: str) -> str:
    """Rewrite an ambiguous or ticker-heavy query into a clearer retrieval query.

    Call this BEFORE search_local when:
    - The query contains ticker symbols (e.g. "ADI", "MSFT") without company names
    - The query is vague or uses pronouns ("it", "they", "the company")
    - The initial search returned poor results and a rephrased query may help

    Args:
        query:  The original user query to rewrite.
        reason: Why this query needs rewriting (for traceability).

    Returns:
        A clearer, expanded query suitable for retrieval.
    """
    from src.query_rewriter import QueryRewriter
    rewritten = QueryRewriter().rewrite(query)
    return rewritten


# ─────────────────────────────────────────────────────────────────────────────
# Tool 3: 联网检索
# ─────────────────────────────────────────────────────────────────────────────

@tool
def search_web(query: str, top_k: int = 3) -> list[dict]:
    """Search the web for recent or real-time information.

    Use this when:
    - The question asks about recent events not covered by the local knowledge base
    - The user asks about current stock prices, recent news, or latest filings
    - Local search returned no relevant results

    Args:
        query: The search query.
        top_k: Number of web results to return (default 3).

    Returns:
        List of web results with text, source URL, and metadata.
    """
    import os
    api_key = os.environ.get("TAVILY_API_KEY", "")
    if not api_key:
        return [{"text": "Web search unavailable: TAVILY_API_KEY not set.",
                 "source": "", "source_type": "web", "score": 0.0, "metadata": {}}]
    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=api_key)
        response = client.search(query, max_results=top_k)
        results = response.get("results", [])
        return [RetrievalResult.from_web_dict(r).to_dict() for r in results]
    except ImportError:
        return [{"text": "Web search unavailable: install tavily-python.",
                 "source": "", "source_type": "web", "score": 0.0, "metadata": {}}]
    except Exception as e:
        return [{"text": f"Web search error: {e}",
                 "source": "", "source_type": "web", "score": 0.0, "metadata": {}}]


# ─────────────────────────────────────────────────────────────────────────────

def offload_retrieval_models() -> None:
    """释放 BGE-M3 + Reranker 的 GPU 显存，为 LLM 推理腾出空间。"""
    _retriever._vs.offload()
    _retriever._reranker.offload()


# 导出工具列表供 LangGraph ToolNode 使用
tools = [search_local, rewrite_query, search_web]
