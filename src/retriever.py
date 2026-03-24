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

from src.config import config

_cfg = config.get("retriever", {})
_MODE = _cfg.get("mode", "custom")
_TOP_K = _cfg.get("top_k", 5)
_CUSTOM_CFG = _cfg.get("custom", {})
_M3_CFG = _cfg.get("m3_hybrid", {})


class Retriever:
    def __init__(self, vector_store, bm25_store, reranker):
        """
        依赖注入：外部传入已初始化的三个组件。

        Args:
            vector_store : VectorStore 实例
            bm25_store   : BM25Store 实例
            reranker     : Reranker 实例
        """
        self._vs = vector_store
        self._bm25 = bm25_store
        self._reranker = reranker

    def search(self, query: str, top_k: int = None) -> list[dict]:
        """
        统一检索入口，根据 config.retriever.mode 分发。

        Args:
            query : 用户查询文本
            top_k : 覆盖 config 默认值（可选）

        Returns:
            list of dicts: chunk 字段 + score，按 score 降序，长度 ≤ top_k
        """
        k = top_k if top_k is not None else _TOP_K
        if _MODE == "m3_hybrid":
            return self._search_m3_hybrid(query, k)
        return self._search_custom(query, k)

    def _search_custom(self, query: str, top_k: int) -> list[dict]:
        """
        custom 模式：dense + BM25 双路召回 → alpha 加权融合 → rerank。

        融合公式：final_score = alpha * dense_score + (1 - alpha) * bm25_score
        去重 key：doc_id + chunk_index
        """
        alpha = _CUSTOM_CFG.get("alpha", 0.5)
        candidate_k = _CUSTOM_CFG.get("candidate_k", 20)

        dense_results = self._vs.search(query, top_k=candidate_k)
        bm25_results = self._bm25.search(query, top_k=candidate_k)
        merged = {}

        for r in dense_results:
            key = (r["doc_id"], r["chunk_index"])
            merged[key] = {**r, "dense_score": r["score"], "bm25_score": 0.0}
        for r in bm25_results:
            key = (r["doc_id"], r["chunk_index"])
            if key in merged:
                merged[key]["bm25_score"] = r["score"]
            else:
                merged[key] = {**r, "dense_score": 0.0, "bm25_score": r["score"]}

        diff = 1.0 - alpha
        candidates = []

        for chunk in merged.values():
            chunk["score"] = (alpha * chunk["dense_score"]) + (diff * chunk["bm25_score"])
            candidates.append(chunk)

        return self._reranker.rerank(query, candidates, top_k=top_k)

    def _search_m3_hybrid(self, query: str, top_k: int) -> list[dict]:
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
        result = self._reranker.rerank(query, candidates, top_k=top_k)
        return result
        
