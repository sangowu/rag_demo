"""
retriever.py
============
Hybrid retriever: combines VectorStore, BM25Store, and Reranker into a unified search interface.

Supports two modes (config.retriever.mode):
  - custom    : dense (VectorStore) + sparse (BM25) alpha-weighted fusion → rerank
  - m3_hybrid : BGE-M3 dense+sparse fusion (VectorStore.search_with_sparse) → rerank

Usage:
    from src.vector_store import VectorStore
    from src.bm25_store import BM25Store
    from src.reranker import Reranker
    from src.retriever import Retriever

    retriever = Retriever(VectorStore(), BM25Store(), Reranker())
    results = retriever.search("What was ADI revenue in 2009?")
"""

from collections import defaultdict

from src.config import config
from src.metadata_extractor import build_chroma_filter, extract_query_metadata

_cfg = config.get("retriever", {})
_MODE = _cfg.get("mode", "custom")
_TOP_K = _cfg.get("top_k", 5)
_CUSTOM_CFG = _cfg.get("custom", {})
_M3_CFG = _cfg.get("m3_hybrid", {})


class Retriever:
    def __init__(self, vector_store, bm25_store, reranker, summary_store=None, query_rewriter=None, semantic_cache=None):
        """
        依赖注入：外部传入已初始化的组件。

        Args:
            vector_store   : VectorStore 实例
            bm25_store     : BM25Store 实例
            reranker       : Reranker 实例
            summary_store  : SummaryStore 实例（可选，summary_filter 模式需要）
            query_rewriter : QueryRewriter 实例（可选，rewrite=True 时需要）
            semantic_cache : SemanticCache 实例（可选，命中时跳过检索）
        """
        self._vs       = vector_store
        self._bm25     = bm25_store
        self._reranker = reranker
        self._summary  = summary_store
        self._rewriter = query_rewriter
        self._cache    = semantic_cache

    def search(
        self,
        query: str,
        top_k: int = None,
        rewrite: bool = False,
        meta_filter: bool = False,
        summary_filter: bool = False,
        summary_top_n: int = 30,
    ) -> list[dict]:
        """
        统一检索入口，根据 config.retriever.mode 分发。

        Args:
            query          : 用户查询文本
            top_k          : 覆盖 config 默认值（可选）
            rewrite        : 是否先用 LLM 改写 query（需注入 query_rewriter）
            meta_filter    : 从 query 提取 ticker/year 做 ChromaDB 预过滤
            summary_filter : 先用 summary collection 找相关文档，再在其 chunk 里搜索
            summary_top_n  : summary 阶段返回的候选文档数（默认 30）

        Returns:
            list of dicts: chunk 字段 + score，按 score 降序，长度 ≤ top_k
        """
        k = top_k if top_k is not None else _TOP_K

        # semantic cache：先 embed query，命中则跳过后续所有检索步骤
        if self._cache is not None:
            query_vec = self._vs._embed_query(query)["dense_vec"]
            cached = self._cache.get(query_vec)
            if cached is not None:
                return cached[:k]

        # query rewriting：改写后用于 dense + BM25，原始 query 保留给 reranker
        retrieval_query = query
        if rewrite and self._rewriter is not None:
            retrieval_query = self._rewriter.rewrite(query)
        # embed 已在 cache 检查时完成，避免重复；无 cache 时在 VectorStore.search 内部完成

        where = None
        if summary_filter and self._summary is not None:
            top_doc_ids = self._summary.search(query, top_n=summary_top_n)
            if top_doc_ids:
                where = {"doc_id": {"$in": top_doc_ids}}
        elif meta_filter:
            known_tickers = self._vs.get_known_tickers()
            query_meta = extract_query_metadata(query, known_tickers)
            where = build_chroma_filter(query_meta)

        if _MODE == "m3_hybrid":
            return self._search_m3_hybrid(retrieval_query, k, original_query=query)
        return self._search_custom(retrieval_query, k, where=where, original_query=query)

    def _rrf_merge(self, result_lists: list[list[dict]], k: int = 60) -> list[dict]:
        """
        Reciprocal Rank Fusion：只用排名位置，不依赖原始分数量纲。

        公式：score(doc) = Σ 1 / (k + rank + 1)，rank 从 0 开始
        k=60 是学术标准默认值，无需调参。
        """
        scores: dict[tuple, float] = defaultdict(float)
        doc_map: dict[tuple, dict] = {}

        for results in result_lists:
            for rank, r in enumerate(results):
                key = (r["doc_id"], r["chunk_index"])
                scores[key] += 1.0 / (k + rank + 1)
                if key not in doc_map:
                    doc_map[key] = r

        sorted_keys = sorted(scores, key=scores.__getitem__, reverse=True)
        return [{**doc_map[key], "score": scores[key]} for key in sorted_keys]

    def _search_custom(self, query: str, top_k: int, where: dict = None, original_query: str = None) -> list[dict]:
        """
        custom 模式：dense + BM25 双路召回 → RRF 融合 → rerank。

        去重 key：doc_id + chunk_index
        rerank 使用 original_query（未改写的原始问题），保证语义对齐。
        """
        candidate_k = _CUSTOM_CFG.get("candidate_k", 20)
        rrf_k = _CUSTOM_CFG.get("rrf_k", 60)

        dense_results = self._vs.search(query, top_k=candidate_k, where=where)
        bm25_results = self._bm25.search(query, top_k=candidate_k)

        candidates = self._rrf_merge([dense_results, bm25_results], k=rrf_k)

        rerank_query = original_query if original_query is not None else query
        results = self._reranker.rerank(rerank_query, candidates, top_k=top_k)

        # 写入 semantic cache
        if self._cache is not None:
            cache_key = original_query if original_query is not None else query
            cache_vec = self._vs._embed_query(cache_key)["dense_vec"]
            self._cache.set(cache_vec, cache_key, results)

        return results

    def _search_m3_hybrid(self, query: str, top_k: int, original_query: str = None) -> list[dict]:
        """
        m3_hybrid 模式：BGE-M3 dense+sparse 融合召回 → rerank。
        """
        dense_weight = _M3_CFG.get("dense_weight", 0.5)
        sparse_weight = _M3_CFG.get("sparse_weight", 0.5)
        candidate_k = _cfg.get("top_k", 20)
        candidates = self._vs.search_with_sparse(
            query,
            top_k=candidate_k,
            dense_weight=dense_weight,
            sparse_weight=sparse_weight,
        )
        rerank_query = original_query if original_query is not None else query
        return self._reranker.rerank(rerank_query, candidates, top_k=top_k)
        
