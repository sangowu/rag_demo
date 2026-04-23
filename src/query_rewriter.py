"""
query_rewriter.py
=================
LLM-based query rewriting for improved retrieval.

FinQA documents do not mention company ticker symbols in body text.
This module expands tickers and clarifies financial terms before retrieval.

Usage:
    from src.query_rewriter import QueryRewriter
    rewriter = QueryRewriter()
    rewritten = rewriter.rewrite("What was ADI revenue in 2009?")
    # → "What was Analog Devices total revenue in fiscal year 2009?"
"""

from src.llm_factory import get_llm

_rw_cfg = {}

_PROMPT_TEMPLATE = """\
You are a financial document retrieval assistant.
Rewrite the query to improve retrieval from annual report documents:
1. Expand company ticker symbols to full company names (e.g. ADI → Analog Devices, JPM → JPMorgan Chase & Co, C → Citigroup)
2. Keep financial terms precise (revenue, net income, operating expenses, EPS)
3. Output ONE concise sentence — no explanation, no punctuation changes

Original query: {query}
Rewritten query:"""


class QueryRewriter:
    def __init__(self):
        self._llm = get_llm(temperature=0.0, max_output_tokens=64)
        self._cache: dict[str, str] = {}

    def rewrite(self, query: str) -> str:
        """
        改写 query，提升检索命中率。
        相同 query 使用缓存，避免重复 LLM 调用。

        Args:
            query: 用户原始问题

        Returns:
            改写后的 query（失败时返回原始 query）
        """
        if query in self._cache:
            return self._cache[query]

        try:
            result = self._llm.invoke(_PROMPT_TEMPLATE.format(query=query))
            rewritten = result.content.strip()
            # 防止 LLM 输出多行或带前缀
            rewritten = rewritten.splitlines()[0].strip()
            if not rewritten:
                rewritten = query
        except Exception:
            rewritten = query

        self._cache[query] = rewritten
        return rewritten
