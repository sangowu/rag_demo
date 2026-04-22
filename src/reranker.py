"""
reranker.py
===========
BGE cross-encoder reranker for re-scoring retrieval candidates.

Provides:
  - rerank : 对候选 chunk 列表重新打分，返回归一化分数的 top-k 结果

LRU 缓存策略：
  对 (query, doc_id) 对的 score 做 LRU 缓存（默认容量 1024），
  避免相同文档在相似查询下重复跑 cross-encoder 推理。
  offload() 时同步清空缓存。

Usage:
    from src.reranker import Reranker
    reranker = Reranker()
    results = reranker.rerank("What was ADI revenue in 2009?", chunks, top_k=5)
"""

from collections import OrderedDict

from src.config import config

_cfg = config.get("reranker", {})
_CACHE_SIZE = _cfg.get("cache_size", 1024)   # LRU 最大条目数


class _LRUCache:
    """简单的线程不安全 LRU 缓存，key=(query, doc_id)，value=score。"""

    def __init__(self, maxsize: int = 1024):
        self._cache: OrderedDict[tuple, float] = OrderedDict()
        self._maxsize = maxsize

    def get(self, key: tuple) -> float | None:
        if key not in self._cache:
            return None
        self._cache.move_to_end(key)   # 最近使用移到末尾
        return self._cache[key]

    def put(self, key: tuple, value: float) -> None:
        if key in self._cache:
            self._cache.move_to_end(key)
        else:
            if len(self._cache) >= self._maxsize:
                self._cache.popitem(last=False)   # 淘汰最久未用
        self._cache[key] = value

    def clear(self) -> None:
        self._cache.clear()

    def __len__(self) -> int:
        return len(self._cache)


_vs_cfg = config.get("vector_store", {})


class Reranker:
    def __init__(self):
        self._model = None
        self._cache = _LRUCache(maxsize=_CACHE_SIZE)

    def _use_remote(self) -> bool:
        return bool(_vs_cfg.get("embedding_server_url", "").strip())

    def _remote_rerank(self, query: str, texts: list[str]) -> list[float]:
        import requests
        url = _vs_cfg["embedding_server_url"].rstrip("/") + "/rerank"
        resp = requests.post(url, json={"query": query, "texts": texts}, timeout=120)
        resp.raise_for_status()
        return resp.json()["scores"]

    def _load_model(self):
        if self._model is None:
            from FlagEmbedding import FlagReranker
            import torch
            model_name = _cfg.get("model_path") or "BAAI/bge-reranker-v2-m3"
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            self._model = FlagReranker(
                model_name,
                use_fp16=torch.cuda.is_available(),
                devices=[device],
            )

    def rerank(self, query: str, chunks: list[dict], top_k: int = 5) -> list[dict]:
        """
        对候选 chunk 列表用 cross-encoder 重新打分。
        命中缓存的 (query, doc_id) 对直接复用分数，跳过推理。

        Args:
            query  : 用户查询文本
            chunks : 候选 chunk 列表，每个 dict 含 text, doc_id, chunk_index 等字段
            top_k  : 返回得分最高的 top_k 个结果

        Returns:
            list of dicts: 原始字段 + score（归一化 0-1），按 score 降序
        """
        # 分离缓存命中 / 未命中
        miss_indices = []
        for i, chunk in enumerate(chunks):
            cached = self._cache.get((query, chunk["doc_id"]))
            if cached is not None:
                chunk["score"] = cached
            else:
                miss_indices.append(i)

        if miss_indices:
            texts = [chunks[i]["text"] for i in miss_indices]
            if self._use_remote():
                scores = self._remote_rerank(query, texts)
            else:
                self._load_model()
                pairs = [(query, t) for t in texts]
                scores = self._model.compute_score(pairs, normalize=True)

            for i, score in zip(miss_indices, scores):
                s = float(score)
                chunks[i]["score"] = s
                self._cache.put((query, chunks[i]["doc_id"]), s)

        chunks.sort(key=lambda x: x["score"], reverse=True)
        return chunks[:top_k]

    def cache_info(self) -> dict:
        """返回缓存统计信息。"""
        return {"size": len(self._cache), "maxsize": _CACHE_SIZE}

    def offload(self) -> None:
        """释放 BGE-reranker 占用的 GPU 显存，同步清空 score 缓存。"""
        if self._model is not None:
            import torch
            del self._model
            self._model = None
            torch.cuda.empty_cache()
        self._cache.clear()
        

