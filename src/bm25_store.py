"""
bm25_store.py
=============
BM25 sparse retrieval over the chunk corpus using rank-bm25.

Provides:
  - build            : 分词、建索引、持久化到 pkl
  - load             : 加载已持久化的索引
  - search           : BM25 检索，返回归一化分数的 top-k 结果
  - delete_by_doc_id : 删除某文档的 chunk 并重建索引

Usage:
    from src.bm25_store import BM25Store
    store = BM25Store()
    store.build(chunks)
    results = store.search("What was ADI revenue in 2009?", top_k=5)
"""

import pickle
import numpy as np
from rank_bm25 import BM25Okapi
from pathlib import Path
from src.config import config

_ROOT = Path(__file__).parent.parent
_cfg = config["bm25"]
_INDEX_PATH = _ROOT / _cfg["index_path"]


def _tokenize(text: str) -> list[str]:
    """
    分词函数，build 和 search 必须使用同一个函数保证一致性。
    # 转小写后按空格切分
    # 例：_tokenize("ADI Revenue 2009") → ["adi", "revenue", "2009"]
    """
    return text.lower().split()


class BM25Store:
    def __init__(self):
        # BM25Okapi 索引对象，build 或 load 后赋值
        self._bm25 = None
        # 与索引对齐的原始 chunk 列表，用于还原检索结果的元数据
        self._chunks = None

    def build(self, chunks: list[dict]) -> None:
        """
        分词并建立 BM25 索引，持久化到 _INDEX_PATH。

        Args:
            chunks: list of dicts with keys text, doc_id, chunk_index
        """
        tokenized_corpus = [_tokenize(chunk["text"]) for chunk in chunks]
        self._bm25 = BM25Okapi(tokenized_corpus)
        self._chunks = chunks
        _INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(_INDEX_PATH, "wb") as f:
            pickle.dump({
                "bm25": self._bm25,
                "chunks": self._chunks
            }, f)

    def load(self) -> None:
        """
        加载已持久化的 BM25 索引。
        未建立索引时抛出明确错误。
        """
        if not _INDEX_PATH.exists():
            raise FileNotFoundError("BM25 Path Doesn't exist!")
        
        with open(_INDEX_PATH, "rb") as f:
            db = pickle.load(f)
        self._bm25 = db.get("bm25")
        self._chunks = db.get("chunks")


    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """
        BM25 检索，返回 top-k 结果。

        Returns:
            list of dicts: text, doc_id, chunk_index, score（归一化 0-1）
        """
        if self._bm25 is None: 
            self.load()

        query_tokens = _tokenize(query)
        scores = self._bm25.get_scores(query_tokens)
        max_score = scores.max()
        if max_score > 0:
            scores = scores / (max_score + 1e-9)
        top_k_idx = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_k_idx:
            chunk = self._chunks[idx]
            results.append({
                "text": chunk["text"],
                "doc_id": chunk["doc_id"],
                "chunk_index": chunk["chunk_index"],
                "score": float(scores[idx])
            })

        return results

    def delete_by_doc_id(self, doc_id: str) -> None:
        """
        删除某文档的所有 chunk 并重建索引。

        Args:
            doc_id: 要删除的文档 ID
        """
        if self._bm25 is None: 
            self.load()

        remaining = [c for c in self._chunks if c["doc_id"] != doc_id]

        if not remaining:
            _INDEX_PATH.unlink(missing_ok=True)
            self._bm25 = None
            self._chunks = None
        else:
            self.build(remaining)
