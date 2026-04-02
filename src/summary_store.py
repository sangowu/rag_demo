"""
summary_store.py
================
独立的 ChromaDB collection，存储每个文档的 summary 向量（一条 doc = 一条向量）。
用于两阶段检索的 Stage 1：语义匹配找到相关 doc_id，再交给 VectorStore 做 chunk 检索。

Provides:
  - build  : 从 doc_summaries.jsonl 批量 embed 并写入 ChromaDB
  - search : 返回语义最相关的 top-n doc_id 列表

Usage:
    from src.summary_store import SummaryStore
    ss = SummaryStore()
    ss.build()
    doc_ids = ss.search("ADI revenue 2009", top_n=20)
"""

import json
from pathlib import Path

from src.config import config

_ROOT       = _ROOT = Path(__file__).parent.parent
_vs_cfg     = config["vector_store"]
_SUMMARY_PATH = _ROOT / "data/doc_summaries.jsonl"
_COLLECTION   = "finqa_summaries"


class SummaryStore:
    def __init__(self):
        import chromadb
        chroma_path = str(_ROOT / _vs_cfg["chroma_path"])
        self._client     = chromadb.PersistentClient(path=chroma_path)
        self._collection = self._client.get_or_create_collection(
            name=_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
        self._model = None

    def _load_model(self):
        if self._model is None:
            from FlagEmbedding import BGEM3FlagModel
            model_name = _vs_cfg.get("embedding_model_path") or _vs_cfg["embedding_model"]
            self._model = BGEM3FlagModel(model_name, use_fp16=True)

    def offload(self) -> None:
        """释放 GPU 显存。"""
        if self._model is not None:
            import torch
            del self._model
            self._model = None
            torch.cuda.empty_cache()

    def build(self, summary_path: Path = _SUMMARY_PATH, batch_size: int = 32) -> None:
        """
        从 doc_summaries.jsonl 批量 embed summary，upsert 到 ChromaDB。
        已存在的 doc_id 自动跳过。
        """
        if not summary_path.exists():
            raise FileNotFoundError(
                f"{summary_path} not found. Run: python src/doc_summarizer.py"
            )

        self._load_model()

        with open(summary_path, encoding="utf-8") as f:
            records = [json.loads(line) for line in f if line.strip()]

        existing = set(self._collection.get(include=[])["ids"])
        pending  = [r for r in records if r["doc_id"] not in existing]
        print(f"SummaryStore: {len(existing)} existing, {len(pending)} to add")

        for i in range(0, len(pending), batch_size):
            batch    = pending[i : i + batch_size]
            texts    = [r["summary"] for r in batch]
            doc_ids  = [r["doc_id"]  for r in batch]

            result = self._model.encode_corpus(
                texts,
                batch_size=batch_size,
                return_dense=True,
                return_sparse=False,
                return_colbert_vecs=False,
            )
            embeddings = result["dense_vecs"].tolist()

            self._collection.upsert(
                ids=doc_ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=[{"doc_id": d} for d in doc_ids],
            )

        print(f"SummaryStore build done. Total: {self._collection.count()} docs")

    def search(self, query: str, top_n: int = 30) -> list[str]:
        """
        语义搜索 summary collection，返回最相关的 top_n 个 doc_id。

        Args:
            query  : 用户查询文本
            top_n  : 返回的文档数量（作为 Stage 2 的过滤范围）

        Returns:
            list of doc_id strings，按相关度降序
        """
        self._load_model()

        result = self._model.encode_queries(
            [query],
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        query_vec = result["dense_vecs"][0].tolist()

        results = self._collection.query(
            query_embeddings=[query_vec],
            n_results=min(top_n, self._collection.count()),
            include=["metadatas"],
        )
        return [m["doc_id"] for m in results["metadatas"][0]]
