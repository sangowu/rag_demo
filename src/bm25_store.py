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

import os
import pickle
import re
import numpy as np
from rank_bm25 import BM25Okapi
from pathlib import Path
from src.config import config
from src.metadata_extractor import extract_doc_metadata

_ROOT = Path(__file__).parent.parent
_cfg = config["bm25"]
_INDEX_PATH = _ROOT / _cfg["index_path"]


def _tokenize(text: str) -> list[str]:
    """
    分词函数，build 和 search 必须使用同一个函数保证一致性。
    # 转小写后按空格切分
    # 例：_tokenize("ADI Revenue 2009") → ["adi", "revenue", "2009"]
    """
    return re.findall(r'\b\w+\b', text.lower())


class BM25Store:
    def __init__(self, index_path: "str | Path | None" = None):
        self._index_path = Path(index_path) if index_path else _INDEX_PATH
        self._bm25 = None
        self._chunks = None

    def build(self, chunks: list[dict], header_override: dict = None) -> None:
        """
        分词并建立 BM25 索引，持久化到 _INDEX_PATH。

        Args:
            chunks          : list of dicts with keys text, doc_id, chunk_index
            header_override : {doc_id: header_str}，用于通用来源文档覆盖文件名解析
                              若为 None，从 doc_id 文件名解析 ticker/year（FinQA 兼容）
        """
        def _chunk_text(chunk: dict) -> str:
            doc_id = chunk["doc_id"]
            if header_override and doc_id in header_override:
                header = header_override[doc_id]
            else:
                dm = extract_doc_metadata(doc_id)
                header = f"[{dm['ticker']} | {dm['year']}]\n" if dm["ticker"] else ""
            return header + chunk["text"]

        tokenized_corpus = [_tokenize(_chunk_text(chunk)) for chunk in chunks]
        self._bm25 = BM25Okapi(tokenized_corpus)
        self._chunks = chunks
        self._index_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._index_path, "wb") as f:
            pickle.dump({
                "bm25": self._bm25,
                "chunks": self._chunks
            }, f)
        s3_bucket = os.environ.get("S3_BUCKET")
        if s3_bucket:
            self._upload_to_s3(s3_bucket)

    def _upload_to_s3(self, bucket: str) -> None:
        import boto3
        s3 = boto3.client("s3")
        s3_key = f"bm25/{self._index_path.name}"
        s3.upload_file(str(self._index_path), bucket, s3_key)

    def load(self) -> None:
        """
        加载 BM25 索引：优先从 S3 下载，本地作为缓存。
        S3_BUCKET 环境变量未设置时退回本地文件。
        """
        if not self._index_path.exists():
            s3_bucket = os.environ.get("S3_BUCKET")
            if s3_bucket:
                self._download_from_s3(s3_bucket)
            else:
                raise FileNotFoundError("BM25 index not found locally and S3_BUCKET not set")

        with open(self._index_path, "rb") as f:
            db = pickle.load(f)
        self._bm25 = db.get("bm25")
        self._chunks = db.get("chunks")

    def _download_from_s3(self, bucket: str) -> None:
        import boto3
        s3 = boto3.client("s3")
        s3_key = f"bm25/{self._index_path.name}"
        self._index_path.parent.mkdir(parents=True, exist_ok=True)
        s3.download_file(bucket, s3_key, str(self._index_path))


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
            self._index_path.unlink(missing_ok=True)
            self._bm25 = None
            self._chunks = None
        else:
            self.build(remaining)
