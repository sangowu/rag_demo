"""
semantic_cache.py
=================
语义缓存：对语义相似的 query 复用已有检索结果，减少重复推理开销。

原理：
  - 每次检索后缓存 (query_embedding, results)
  - 新 query 先 embed，与缓存做余弦相似度对比
  - 相似度 > threshold 时直接返回缓存，跳过检索 + rerank

持久化：
  - 缓存保存到 data/semantic_cache.pkl，服务重启后保留

Usage:
    from src.semantic_cache import SemanticCache
    cache = SemanticCache()
    result = cache.get(query_vec)       # None 表示未命中
    cache.set(query_vec, query, result) # 写入缓存
"""

import pickle
from pathlib import Path

import numpy as np

from src.config import config

_ROOT = Path(__file__).parent.parent
_CACHE_PATH = _ROOT / "data" / "semantic_cache.pkl"
_cfg = config.get("semantic_cache", {})
_THRESHOLD = _cfg.get("threshold", 0.92)
_MAX_SIZE = _cfg.get("max_size", 1000)


class SemanticCache:
    def __init__(self):
        # 每条记录：{"vec": np.ndarray, "query": str, "results": list[dict]}
        self._entries: list[dict] = []
        self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, query_vec: list[float]) -> list[dict] | None:
        """
        查找语义相似的缓存条目。

        Args:
            query_vec: 当前 query 的 dense embedding

        Returns:
            命中时返回缓存的检索结果；未命中返回 None
        """
        if not self._entries:
            return None

        q = np.array(query_vec, dtype=np.float32)
        q = q / (np.linalg.norm(q) + 1e-9)

        vecs = np.stack([e["vec"] for e in self._entries])  # (N, D)
        sims = vecs @ q                                      # cosine similarity

        best_idx = int(np.argmax(sims))
        if sims[best_idx] >= _THRESHOLD:
            return self._entries[best_idx]["results"]
        return None

    def set(self, query_vec: list[float], query: str, results: list[dict]) -> None:
        """
        写入缓存并持久化。超过 max_size 时淘汰最早的条目（FIFO）。

        Args:
            query_vec : query 的 dense embedding
            query     : 原始 query 文本（供调试查看）
            results   : 检索结果
        """
        vec = np.array(query_vec, dtype=np.float32)
        vec = vec / (np.linalg.norm(vec) + 1e-9)

        self._entries.append({"vec": vec, "query": query, "results": results})

        if len(self._entries) > _MAX_SIZE:
            self._entries = self._entries[-_MAX_SIZE:]

        self._save()

    def size(self) -> int:
        return len(self._entries)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if _CACHE_PATH.exists():
            try:
                with open(_CACHE_PATH, "rb") as f:
                    self._entries = pickle.load(f)
            except Exception:
                self._entries = []

    def _save(self) -> None:
        _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(_CACHE_PATH, "wb") as f:
            pickle.dump(self._entries, f)
