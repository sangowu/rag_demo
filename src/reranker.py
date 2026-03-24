"""
reranker.py
===========
BGE cross-encoder reranker for re-scoring retrieval candidates.

Provides:
  - rerank : 对候选 chunk 列表重新打分，返回归一化分数的 top-k 结果

Usage:
    from src.reranker import Reranker
    reranker = Reranker()
    results = reranker.rerank("What was ADI revenue in 2009?", chunks, top_k=5)
"""

from src.config import config

_cfg = config.get("reranker", {})


class Reranker:
    def __init__(self):
        # BGE Reranker 模型懒加载，首次 rerank 时才初始化
        self._model = None

    def _load_model(self):
        """懒加载 FlagReranker，只初始化一次。"""
        if self._model is None:
            from FlagEmbedding import FlagReranker
            self._model = FlagReranker(
                'BAAI/bge-reranker-v2-m3',
                use_fp16=True,
                devices=["cuda:0"], 
            )

    def rerank(self, query: str, chunks: list[dict], top_k: int = 5) -> list[dict]:
        """
        对候选 chunk 列表用 cross-encoder 重新打分。

        Args:
            query  : 用户查询文本
            chunks : 候选 chunk 列表，每个 dict 含 text, doc_id, chunk_index 等字段
            top_k  : 返回得分最高的 top_k 个结果

        Returns:
            list of dicts: 原始字段 + score（归一化 0-1），按 score 降序
        """
        self._load_model()
        pairs = [(query, chunk["text"]) for chunk in chunks]
        scores = self._model.compute_score(pairs, normalize=True)
        for chunk, score in zip(chunks, scores):
            chunk["score"] = float(score)
        chunks.sort(key=lambda x: x["score"], reverse=True)

        return chunks[:top_k]
        

