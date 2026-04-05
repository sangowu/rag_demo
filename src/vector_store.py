"""
vector_store.py
===============
ChromaDB + BGE-M3 vector store wrapper.

支持两种检索模式（由 config retriever.mode 控制）：
  - custom    : 只存储 dense 向量，配合 BM25Store + Reranker 使用
  - m3_hybrid : 同时存储 dense + sparse 向量，用 BGE-M3 原生混合检索

Provides:
  - add_documents      : embed chunks，upsert 到 ChromaDB（跳过已存在的）
  - search             : dense 向量检索，返回 top-k 结果
  - search_with_sparse : dense + sparse 混合检索（m3_hybrid 模式用）
  - delete_by_doc_id   : 删除某文档的所有 chunk
  - get_indexed_ids    : 返回已索引的所有 chunk ID

Usage:
    from src.vector_store import VectorStore
    vs = VectorStore()
    vs.add_documents(chunks)
    results = vs.search("What was ADI revenue in 2009?", top_k=5)
"""
from pathlib import Path
import json
import logging
from langchain_core.embeddings import Embeddings

from src.config import config
from src.metadata_extractor import extract_doc_metadata

_ROOT = Path(__file__).parent.parent
_vs_cfg = config["vector_store"]
_r_cfg = config["retriever"]
_chunk_cfg = config.get("chunking") or {}
_logger = logging.getLogger(__name__)

_store_sparse = _vs_cfg.get("store_sparse", False)
_d_w = _r_cfg.get("m3_hybrid").get("dense_weight")
_s_w = _r_cfg.get("m3_hybrid").get("sparse_weight")


class VectorStoreDenseLangchainEmbeddings(Embeddings):
    """
    LangChain Embeddings adapter backed by the same BGEM3FlagModel as VectorStore.

    SemanticChunker calls embed_documents with many sentences at once; this class
    re-batches internally so VRAM stays bounded.

    Args:
        vector_store: VectorStore instance that owns the shared model.
        batch_size:   Max texts per encode_corpus call; defaults from config
                      chunking.semantic_embed_batch_size or vector_store.embed_batch_size.
    """

    def __init__(self, vector_store: "VectorStore", batch_size: int | None = None):
        self._vs = vector_store
        default_bs = _chunk_cfg.get("semantic_embed_batch_size")
        if default_bs is None:
            default_bs = _vs_cfg.get("embed_batch_size", 32)
        self._batch_size = int(batch_size) if batch_size is not None else int(default_bs)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Encode document texts to dense vectors using the shared BGE-M3 model.

        Args:
            texts: Non-empty list of strings to embed.

        Returns:
            One dense vector per input string, same order.

        Raises:
            Exception: Propagated from FlagEmbedding / CUDA if encoding fails.
        """
        if not texts:
            return []
        self._vs._load_model()
        out: list[list[float]] = []
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            try:
                embed_res = self._vs._embed_documents(batch)
            except Exception as exc:
                _logger.error(
                    "embed_documents batch failed (start=%s, len=%s): %s",
                    i,
                    len(batch),
                    exc,
                )
                raise
            out.extend(embed_res["dense_vecs"])
        return out

    def embed_query(self, text: str) -> list[float]:
        """
        Encode a single query string to a dense vector.

        Args:
            text: Query text.

        Returns:
            Dense embedding vector.

        Raises:
            Exception: Propagated from FlagEmbedding / CUDA if encoding fails.
        """
        try:
            return self._vs._embed_query(text)["dense_vec"]
        except Exception as exc:
            _logger.error("embed_query failed: %s", exc)
            raise


class VectorStore:
    def __init__(self):
        # 初始化 ChromaDB PersistentClient，持久化到 _cfg["chroma_path"]
        # 获取或创建 collection，使用 cosine 距离（metadata={"hnsw:space": "cosine"}）
        # BGE-M3 模型懒加载，首次 add/search 时才初始化
        import chromadb
        chroma_path = str(_ROOT / _vs_cfg["chroma_path"])
        self._client = chromadb.PersistentClient(path=chroma_path)
        self._collection = self._client.get_or_create_collection(
            name=_vs_cfg["collection_name"],
            metadata={"hnsw:space": "cosine"}
        )
        self._model = None
        self._langchain_dense_embeddings: VectorStoreDenseLangchainEmbeddings | None = None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_model(self):
        """懒加载 BGE-M3 模型，只初始化一次。"""
        # 用 BGEM3FlagModel 加载模型，use_fp16=True 节省显存
        if self._model is None:
            from FlagEmbedding import BGEM3FlagModel
            model_name = _vs_cfg.get("embedding_model_path") or _vs_cfg["embedding_model"]
            self._model = BGEM3FlagModel(
                model_name,
                use_fp16=True
            )

    def _embed_documents(self, texts: list[str]) -> dict:
        """
        批量 embed 文档文本。

        Returns:
            dict with keys:
              "dense_vecs"      : list[list[float]]
              "sparse_weights"  : list[dict]  （仅 store_sparse=True 时有效）
        """
        if self._model is None:
            self._load_model()
        results = self._model.encode_corpus(
            texts,
            batch_size=_vs_cfg["embed_batch_size"],
            max_length=_vs_cfg["embed_max_token"],
            return_dense=True,
            return_sparse=_store_sparse,
            return_colbert_vecs=False,
        )

        return {
            "dense_vecs": results["dense_vecs"].tolist(),
            "sparse_weights": results.get("lexical_weights")
        }

    def _embed_query(self, query: str) -> dict:
        """
        Embed 单条查询文本。

        Returns:
            dict with keys:
              "dense_vec"      : list[float]
              "sparse_weights" : dict          （仅 store_sparse=True 时有效）
        """
        if self._model is None:
            self._load_model()

        results = self._model.encode_queries(
            [query],
            return_dense=True,
            return_sparse=_store_sparse,
            return_colbert_vecs=False,
        )
        sparse_data = results.get("lexical_weights")
        return {
            "dense_vec": results["dense_vecs"][0].tolist(),
            "sparse_weights": sparse_data[0] if sparse_data is not None else {}
        }

    @staticmethod
    def _sparse_score(query_weights: dict, doc_weights: dict) -> float:
        """计算两个 sparse weight dict 的点积得分。"""
        score = sum(
            q_w * doc_weights.get(str(tid), 0.0)
            for tid, q_w in query_weights.items()
        )
        return score

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def langchain_dense_embeddings(
        self,
        batch_size: int | None = None,
    ) -> VectorStoreDenseLangchainEmbeddings:
        """
        Return a LangChain Embeddings object that shares this store's BGE-M3 model.

        Use with ChunkManager(semantic_embeddings=...) when strategy is semantic
        to avoid loading a second SentenceTransformer copy.

        Args:
            batch_size: Override semantic embedding batch size; None uses config.

        Returns:
            Cached VectorStoreDenseLangchainEmbeddings (one per VectorStore).
        """
        if self._langchain_dense_embeddings is None:
            self._langchain_dense_embeddings = VectorStoreDenseLangchainEmbeddings(
                self,
                batch_size=batch_size,
            )
        return self._langchain_dense_embeddings

    def get_known_tickers(self) -> set[str]:
        """返回 ChromaDB 中所有已索引的 ticker 集合，用于 query 元信息匹配。"""
        result = self._collection.get(include=["metadatas"])
        return {m["ticker"] for m in result["metadatas"] if m.get("ticker")}

    def get_indexed_ids(self) -> set[str]:
        """返回 collection 中已有的所有 chunk ID。"""
        result = self._collection.get(include=[])
        return set(result["ids"])

    def add_documents(self, chunks: list[dict], header_override: str = None) -> None:
        """
        Embed chunks and upsert into ChromaDB.
        Skips chunks whose ID is already in the collection.

        Args:
            chunks          : list of dicts with keys text, doc_id, chunk_index
            header_override : 若提供，所有 chunk 使用此 header（用于通用来源文档）
                              若为 None，从 doc_id 文件名解析 ticker/year（FinQA 兼容）
        """
        if self._model is None:
            self._load_model()

        existing = self.get_indexed_ids()
        new_chunks = [c for c in chunks if f"{c['doc_id']}_{c['chunk_index']}" not in existing]

        if not new_chunks:
            return

        batch_size = _vs_cfg.get("embed_batch_size", 32)
        for i in range(0, len(new_chunks), batch_size):
            batch = new_chunks[i : i + batch_size]

            batch_texts = []
            for c in batch:
                if header_override is not None:
                    header = header_override
                else:
                    # FinQA 兼容：从文件名解析 ticker/year
                    dm = extract_doc_metadata(c["doc_id"])
                    header = f"[{dm['ticker']} | {dm['year']}]\n" if dm["ticker"] else ""
                batch_texts.append(header + c["text"])
            embed_res = self._embed_documents(batch_texts)
            
            ids = [f"{c['doc_id']}_{c['chunk_index']}" for c in batch]
            metadatas = []
            for idx, c in enumerate(batch):
                doc_meta = extract_doc_metadata(c["doc_id"])
                meta = {
                    "doc_id":      c["doc_id"],
                    "chunk_index": c["chunk_index"],
                    "ticker":      doc_meta["ticker"],
                    "year":        doc_meta["year"],
                }
                if _store_sparse:
                    sparse = {k: float(v) for k, v in embed_res["sparse_weights"][idx].items()}
                    meta["sparse_weights"] = json.dumps(sparse)
                metadatas.append(meta)
                
            self._collection.upsert(
                ids=ids,
                embeddings=embed_res["dense_vecs"],
                documents=batch_texts,
                metadatas=metadatas
            )


    def search(self, query: str, top_k: int = 5, where: dict = None) -> list[dict]:
        """
        Dense 向量检索，返回 top-k 结果。
        custom 模式使用此方法。

        Args:
            where: ChromaDB 元数据过滤条件，如 {"ticker": {"$eq": "ADI"}}
                   传入可缩小搜索范围，None 表示全量搜索。

        Returns:
            list of dicts: text, doc_id, chunk_index, score
        """
        if self._model is None:
            self._load_model()

        query_vec = self._embed_query(query)["dense_vec"]
        query_kwargs = dict(
            query_embeddings=[query_vec],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        if where:
            query_kwargs["where"] = where
        results = self._collection.query(**query_kwargs)
        
        ids = results['ids'][0]
        documents = results['documents'][0]
        metadatas = results['metadatas'][0]
        distances = results['distances'][0]

        final_results = []
        for i in range(len(ids)):
            score = 1.0 - distances[i]
            meta = metadatas[i]
            final_results.append({
                "text": documents[i],
                "doc_id": meta.get("doc_id"),
                "chunk_index": meta.get("chunk_index"),
                "score": score,
            })
        
        return final_results

    def search_with_sparse(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Dense + sparse 混合检索，m3_hybrid 模式使用此方法。
        先用 dense 召回 top_k * 4 候选，再用 sparse 重新打分融合。

        Returns:
            list of dicts: text, doc_id, chunk_index, dense_score, sparse_score, score
        """
        if not _store_sparse:
            raise RuntimeError("Sparse storage is disabled in config.")
        if self._model is None:
            self._load_model()

        embed_result = self._embed_query(query)
        query_dense = embed_result["dense_vec"]
        query_sparse = embed_result["sparse_weights"]

        results = self._collection.query(
            query_embeddings=[query_dense],
            n_results=top_k * 4, 
            include=["documents", "metadatas", "distances"]
        )

        ids = results['ids'][0]
        documents = results['documents'][0]
        metadatas = results['metadatas'][0]
        distances = results['distances'][0]

        scored_candidates = []
        for i in range(len(ids)):
            meta = metadatas[i]
            dense_score = 1.0 - distances[i]
            doc_sparse = json.loads(meta.get("sparse_weights", "{}"))
            sparse_score = self._sparse_score(query_sparse, doc_sparse)
            final_score = (_d_w * dense_score) + (_s_w * sparse_score)

            scored_candidates.append({
                "text": documents[i],
                "doc_id": meta.get("doc_id"),
                "chunk_index": meta.get("chunk_index"),
                "dense_score": dense_score,
                "sparse_score": sparse_score,
                "score": final_score
            })

        scored_candidates.sort(key=lambda x: x["score"], reverse=True)
        return scored_candidates[:top_k]

    def offload(self) -> None:
        """释放 BGE-M3 占用的 GPU 显存。检索完成后调用，为 LLM 推理腾出空间。"""
        if self._model is not None:
            import torch
            del self._model
            self._model = None
            torch.cuda.empty_cache()
        # Next langchain_dense_embeddings() still returns the same adapter; it will
        # reload the model on first encode.

    def delete_by_doc_id(self, doc_id: str) -> None:
        """删除某文档的所有 chunk。"""
        self._collection.delete(where={"doc_id": doc_id})
