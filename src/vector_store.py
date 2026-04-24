"""
vector_store.py
===============
pgvector + BGE-M3 vector store wrapper.

Provides:
  - add_documents   : embed chunks，upsert 到 PostgreSQL（跳过已存在的）
  - search          : dense 向量检索，返回 top-k 结果
  - delete_by_doc_id: 删除某文档的所有 chunk
  - get_indexed_ids : 返回已索引的所有 chunk ID

Usage:
    from src.vector_store import VectorStore
    vs = VectorStore()
    vs.add_documents(chunks)
    results = vs.search("What was ADI revenue in 2009?", top_k=5)
"""
import json
import logging
import os
from pathlib import Path

import psycopg2
import psycopg2.extras
from langchain_core.embeddings import Embeddings

from src.config import config
from src.metadata_extractor import extract_doc_metadata

_ROOT = Path(__file__).parent.parent
_vs_cfg = config["vector_store"]
_r_cfg = config["retriever"]
_chunk_cfg = config.get("chunking") or {}
_logger = logging.getLogger(__name__)

_store_sparse = _vs_cfg.get("store_sparse", False)

_HNSW_M = _vs_cfg.get("hnsw_m", 16)
_HNSW_EF_CONSTRUCTION = _vs_cfg.get("hnsw_ef_construction", 64)
_HNSW_EF_SEARCH = _vs_cfg.get("hnsw_ef_search", 100)

# {{table}} becomes {table} after the first .format(), then filled in _init_schema()
_DDL_TEMPLATE = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS {{table}} (
    id             TEXT PRIMARY KEY,
    doc_id         TEXT NOT NULL,
    chunk_index    INT  NOT NULL,
    ticker         TEXT,
    year           TEXT,
    content        TEXT NOT NULL,
    embedding      vector(1024),
    sparse_weights JSONB
);

CREATE INDEX IF NOT EXISTS {{table}}_embedding_hnsw_idx
    ON {{table}} USING hnsw (embedding vector_cosine_ops)
    WITH (m = {m}, ef_construction = {ef_construction});

CREATE INDEX IF NOT EXISTS {{table}}_sparse_gin_idx
    ON {{table}} USING gin (sparse_weights);

CREATE INDEX IF NOT EXISTS {{table}}_doc_id_idx
    ON {{table}} (doc_id);
""".format(m=_HNSW_M, ef_construction=_HNSW_EF_CONSTRUCTION)


def _get_dsn() -> str:
    """优先从环境变量读取 DSN，fallback 到 config。"""
    return os.environ.get("PG_DSN") or _vs_cfg["pg_dsn"]


class VectorStoreDenseLangchainEmbeddings(Embeddings):
    """
    LangChain Embeddings adapter backed by the same BGEM3FlagModel as VectorStore.

    SemanticChunker calls embed_documents with many sentences at once; this class
    re-batches internally so VRAM stays bounded.
    """

    def __init__(self, vector_store: "VectorStore", batch_size: int | None = None):
        self._vs = vector_store
        default_bs = _chunk_cfg.get("semantic_embed_batch_size")
        if default_bs is None:
            default_bs = _vs_cfg.get("embed_batch_size", 32)
        self._batch_size = int(batch_size) if batch_size is not None else int(default_bs)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        self._vs._load_model()
        out: list[list[float]] = []
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            try:
                embed_res = self._vs._embed_documents(batch)
            except Exception as exc:
                _logger.error("embed_documents batch failed (start=%s, len=%s): %s", i, len(batch), exc)
                raise
            out.extend(embed_res["dense_vecs"])
        return out

    def embed_query(self, text: str) -> list[float]:
        try:
            return self._vs._embed_query(text)["dense_vec"]
        except Exception as exc:
            _logger.error("embed_query failed: %s", exc)
            raise


class VectorStore:
    def __init__(self, table_name: str = "chunks"):
        self._table = table_name
        self._conn = psycopg2.connect(_get_dsn())
        self._conn.autocommit = False
        self._init_schema()
        self._model = None
        self._langchain_dense_embeddings: VectorStoreDenseLangchainEmbeddings | None = None

    # ------------------------------------------------------------------
    # Schema init
    # ------------------------------------------------------------------

    def _init_schema(self) -> None:
        """建表、建索引（幂等）。"""
        with self._conn.cursor() as cur:
            cur.execute(_DDL_TEMPLATE.format(table=self._table))
        self._conn.commit()

    def _ensure_conn(self) -> None:
        """连接断开后自动重连。"""
        if self._conn.closed:
            self._conn = psycopg2.connect(_get_dsn())
            self._conn.autocommit = False

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _embedding_server_url(self) -> str:
        return os.environ.get("EMBEDDING_SERVER_URL", "") or _vs_cfg.get("embedding_server_url", "")

    def _use_remote(self) -> bool:
        return bool(self._embedding_server_url().strip())

    def _remote_embed(self, texts: list[str], is_query: bool = False) -> list[list[float]]:
        import requests
        url = self._embedding_server_url().rstrip("/") + "/embed"
        resp = requests.post(url, json={"inputs": texts, "is_query": is_query}, timeout=120)
        resp.raise_for_status()
        return resp.json()["embeddings"]

    def _load_model(self) -> None:
        if self._model is None:
            from FlagEmbedding import BGEM3FlagModel
            model_name = _vs_cfg.get("embedding_model_path") or _vs_cfg["embedding_model"]
            self._model = BGEM3FlagModel(model_name, use_fp16=True)

    def _embed_documents(self, texts: list[str]) -> dict:
        if self._use_remote():
            return {"dense_vecs": self._remote_embed(texts, is_query=False), "sparse_weights": None}
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
            "sparse_weights": results.get("lexical_weights"),
        }

    def _embed_query(self, query: str) -> dict:
        if self._use_remote():
            vec = self._remote_embed([query], is_query=True)[0]
            return {"dense_vec": vec, "sparse_weights": {}}
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
            "sparse_weights": sparse_data[0] if sparse_data is not None else {},
        }



    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def langchain_dense_embeddings(self, batch_size: int | None = None) -> VectorStoreDenseLangchainEmbeddings:
        if self._langchain_dense_embeddings is None:
            self._langchain_dense_embeddings = VectorStoreDenseLangchainEmbeddings(self, batch_size=batch_size)
        return self._langchain_dense_embeddings

    def get_known_tickers(self) -> set[str]:
        self._ensure_conn()
        with self._conn.cursor() as cur:
            cur.execute(f"SELECT DISTINCT ticker FROM {self._table} WHERE ticker IS NOT NULL")
            return {row[0] for row in cur.fetchall()}

    def get_indexed_ids(self) -> set[str]:
        self._ensure_conn()
        with self._conn.cursor() as cur:
            cur.execute(f"SELECT id FROM {self._table}")
            return {row[0] for row in cur.fetchall()}

    def add_documents(self, chunks: list[dict], header_override: str = None) -> None:
        """
        Embed chunks and upsert into PostgreSQL.
        Skips chunks whose ID is already in the table.

        Args:
            chunks          : list of dicts with keys text, doc_id, chunk_index
            header_override : 若提供，所有 chunk 使用此 header（用于通用来源文档）
        """
        if self._model is None:
            self._load_model()

        self._ensure_conn()
        existing = self.get_indexed_ids()
        new_chunks = [c for c in chunks if f"{c['doc_id']}_{c['chunk_index']}" not in existing]

        if not new_chunks:
            return

        batch_size = _vs_cfg.get("embed_batch_size", 32)
        for i in range(0, len(new_chunks), batch_size):
            batch = new_chunks[i : i + batch_size]

            # embed_text: what to encode into a vector (may differ from stored content)
            # content: what to return in search results (stored in DB)
            batch_embed_texts = []
            batch_content_texts = []
            for c in batch:
                if c.get("embed_text") is not None:
                    embed_text = c["embed_text"]
                elif header_override is not None:
                    dm = extract_doc_metadata(c["doc_id"])
                    embed_text = header_override + c["text"]
                else:
                    dm = extract_doc_metadata(c["doc_id"])
                    header = f"[{dm['ticker']} | {dm['year']}]\n" if dm["ticker"] else ""
                    embed_text = header + c["text"]
                batch_embed_texts.append(embed_text)

                content = c.get("content") if c.get("content") is not None else embed_text
                batch_content_texts.append(content)

            embed_res = self._embed_documents(batch_embed_texts)

            rows = []
            for idx, c in enumerate(batch):
                doc_meta = extract_doc_metadata(c["doc_id"])
                chunk_id = f"{c['doc_id']}_{c['chunk_index']}"
                sparse_json = None
                if _store_sparse and embed_res["sparse_weights"] is not None:
                    sparse_json = json.dumps({k: float(v) for k, v in embed_res["sparse_weights"][idx].items()})
                rows.append((
                    chunk_id,
                    c["doc_id"],
                    c["chunk_index"],
                    doc_meta["ticker"],
                    doc_meta["year"],
                    batch_content_texts[idx],
                    embed_res["dense_vecs"][idx],
                    sparse_json,
                ))

            with self._conn.cursor() as cur:
                psycopg2.extras.execute_values(
                    cur,
                    f"""
                    INSERT INTO {self._table} (id, doc_id, chunk_index, ticker, year, content, embedding, sparse_weights)
                    VALUES %s
                    ON CONFLICT (id) DO UPDATE SET
                        content        = EXCLUDED.content,
                        embedding      = EXCLUDED.embedding,
                        sparse_weights = EXCLUDED.sparse_weights
                    """,
                    rows,
                    template="(%s, %s, %s, %s, %s, %s, %s::vector, %s::jsonb)",
                )
            self._conn.commit()

        self._conn.autocommit = True
        with self._conn.cursor() as cur:
            cur.execute(f"VACUUM ANALYZE {self._table}")
        self._conn.autocommit = False

    def add_documents_with_embeddings(
        self,
        chunks: list[dict],
        embeddings: list[list[float]],
        header_override: str = None,
    ) -> None:
        """
        将预计算好的 embeddings 连同 chunks 写入 PostgreSQL（Late Chunking 专用）。
        跳过已存在的 chunk ID，其余逻辑与 add_documents 相同。

        Args:
            chunks     : list of dicts with keys text, doc_id, chunk_index
            embeddings : 与 chunks 等长的向量列表（1024 维）
            header_override : 写入 content 列时前置的 header 文本
        """
        assert len(chunks) == len(embeddings), "chunks 和 embeddings 长度必须一致"
        self._ensure_conn()

        existing = self.get_indexed_ids()
        rows = []
        for chunk, emb in zip(chunks, embeddings):
            chunk_id = f"{chunk['doc_id']}_{chunk['chunk_index']}"
            if chunk_id in existing:
                continue
            doc_meta = extract_doc_metadata(chunk["doc_id"])
            if header_override is not None:
                content = header_override + chunk["text"]
            else:
                dm = extract_doc_metadata(chunk["doc_id"])
                header = f"[{dm['ticker']} | {dm['year']}]\n" if dm["ticker"] else ""
                content = header + chunk["text"]
            rows.append((
                chunk_id,
                chunk["doc_id"],
                chunk["chunk_index"],
                doc_meta["ticker"],
                doc_meta["year"],
                content,
                emb,
                None,  # sparse_weights 暂不支持 late chunking 路径
            ))

        if not rows:
            return

        with self._conn.cursor() as cur:
            psycopg2.extras.execute_values(
                cur,
                f"""
                INSERT INTO {self._table} (id, doc_id, chunk_index, ticker, year, content, embedding, sparse_weights)
                VALUES %s
                ON CONFLICT (id) DO UPDATE SET
                    content   = EXCLUDED.content,
                    embedding = EXCLUDED.embedding
                """,
                rows,
                template="(%s, %s, %s, %s, %s, %s, %s::vector, %s::jsonb)",
            )
        self._conn.commit()

        self._conn.autocommit = True
        with self._conn.cursor() as cur:
            cur.execute(f"VACUUM ANALYZE {self._table}")
        self._conn.autocommit = False

    def search(self, query: str, top_k: int = 5, where: dict = None) -> list[dict]:
        """
        Dense 向量检索，返回 top-k 结果。
        custom 模式使用此方法。

        Args:
            where: 元数据过滤条件，支持 {"ticker": "ADI"} 或 {"ticker": {"$eq": "ADI"}} 两种格式。
                   当前支持字段：ticker, year, doc_id。

        Returns:
            list of dicts: text, doc_id, chunk_index, score
        """
        if self._model is None:
            self._load_model()
        self._ensure_conn()

        query_vec = self._embed_query(query)["dense_vec"]

        sql_where, where_params = self._build_where(where)
        vec_param = json.dumps(query_vec)

        sql = f"""
            SELECT id, doc_id, chunk_index, content,
                   1 - (embedding <=> %s::vector) AS score
            FROM {self._table}
            {sql_where}
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """
        all_params = [vec_param] + where_params + [vec_param, top_k]

        with self._conn.cursor() as cur:
            cur.execute(f"SET LOCAL hnsw.ef_search = {_HNSW_EF_SEARCH}")
            cur.execute(sql, all_params)
            rows = cur.fetchall()

        return [
            {"text": row[3], "doc_id": row[1], "chunk_index": row[2], "score": float(row[4])}
            for row in rows
        ]

    def delete_by_doc_id(self, doc_id: str) -> None:
        self._ensure_conn()
        with self._conn.cursor() as cur:
            cur.execute(f"DELETE FROM {self._table} WHERE doc_id = %s", (doc_id,))
        self._conn.commit()

    def clear_all(self) -> None:
        """清空当前策略表的全部数据（保留表结构）。"""
        self._ensure_conn()
        with self._conn.cursor() as cur:
            cur.execute(f"TRUNCATE TABLE {self._table}")
        self._conn.commit()

    def offload(self) -> None:
        """释放 BGE-M3 占用的 GPU 显存。检索完成后调用，为 LLM 推理腾出空间。"""
        if self._model is not None:
            import torch
            del self._model
            self._model = None
            torch.cuda.empty_cache()

    @staticmethod
    def _build_where(where: dict | None) -> tuple[str, list]:
        """
        将过滤 dict 转成 SQL WHERE 子句和参数列表。

        支持两种格式：
          {"ticker": "ADI"}                  → WHERE ticker = %s
          {"ticker": {"$eq": "ADI"}}         → WHERE ticker = %s
        支持字段：ticker, year, doc_id
        """
        _ALLOWED = {"ticker", "year", "doc_id"}
        if not where:
            return "", []

        clauses, params = [], []
        for field, val in where.items():
            if field not in _ALLOWED:
                continue
            if isinstance(val, dict):
                op_map = {"$eq": "=", "$ne": "!=", "$gt": ">", "$gte": ">=", "$lt": "<", "$lte": "<="}
                for op_key, op_val in val.items():
                    sql_op = op_map.get(op_key, "=")
                    clauses.append(f"{field} {sql_op} %s")
                    params.append(op_val)
            else:
                clauses.append(f"{field} = %s")
                params.append(val)

        if not clauses:
            return "", []
        return "WHERE " + " AND ".join(clauses), params
