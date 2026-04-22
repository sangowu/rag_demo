"""
embedding_server.py
===================
轻量 Embedding + Reranker Server，基于 FlagEmbedding + FastAPI。
运行在 AutoDL 6006 端口（公网或 SSH 隧道可访问）。

Usage:
    python scripts/embedding_server.py

Endpoints:
    POST /embed   : 批量 embed 文本，返回 dense 向量
    POST /rerank  : 对 query-chunk pairs 打分，返回分数列表
    GET  /health  : 健康检查
"""

import logging
import os
import time

import uvicorn
from fastapi import FastAPI, Request
from pydantic import BaseModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("embedding_server.log", encoding="utf-8"),
    ],
)
_logger = logging.getLogger(__name__)

EMBEDDING_MODEL_PATH = os.environ.get(
    "EMBEDDING_MODEL_PATH",
    "/root/autodl-tmp/models/BAAI/bge-m3",
)
RERANKER_MODEL_PATH = os.environ.get(
    "RERANKER_MODEL_PATH",
    "/root/autodl-tmp/models/BAAI/bge-reranker-v2-m3",
)

app = FastAPI(title="BGE Embedding & Reranker Server")

_embed_model = None
_rerank_model = None


def _get_embed_model():
    global _embed_model
    if _embed_model is None:
        from FlagEmbedding import BGEM3FlagModel
        _embed_model = BGEM3FlagModel(EMBEDDING_MODEL_PATH, use_fp16=True)
    return _embed_model


def _get_rerank_model():
    global _rerank_model
    if _rerank_model is None:
        from FlagEmbedding import FlagReranker
        _rerank_model = FlagReranker(
            RERANKER_MODEL_PATH,
            use_fp16=True,
            devices=["cuda:0"],
        )
    return _rerank_model


class EmbedRequest(BaseModel):
    inputs: list[str] | str
    is_query: bool = False


class RerankRequest(BaseModel):
    query: str
    texts: list[str]  # 待排序的文本列表


@app.post("/embed")
def embed(req: EmbedRequest):
    texts = req.inputs if isinstance(req.inputs, list) else [req.inputs]
    mode = "query" if req.is_query else "corpus"
    t0 = time.perf_counter()
    model = _get_embed_model()
    if req.is_query:
        result = model.encode_queries(texts, return_dense=True, return_sparse=False, return_colbert_vecs=False)
    else:
        result = model.encode_corpus(texts, return_dense=True, return_sparse=False, return_colbert_vecs=False)
    latency_ms = (time.perf_counter() - t0) * 1000
    dim = len(result["dense_vecs"][0]) if len(result["dense_vecs"]) > 0 else 0
    _logger.info("embed | mode=%s num_texts=%d dim=%d latency=%.0fms", mode, len(texts), dim, latency_ms)
    return {"embeddings": result["dense_vecs"].tolist()}


@app.post("/rerank")
def rerank(req: RerankRequest):
    t0 = time.perf_counter()
    model = _get_rerank_model()
    pairs = [[req.query, text] for text in req.texts]
    scores = model.compute_score(pairs, normalize=True)
    latency_ms = (time.perf_counter() - t0) * 1000
    if not isinstance(scores, list):
        scores = scores.tolist()
    top_score = max(scores) if scores else 0.0
    _logger.info("rerank | num_pairs=%d top_score=%.4f latency=%.0fms", len(pairs), top_score, latency_ms)
    return {"scores": scores}


@app.get("/health")
def health():
    return {"status": "ok", "embedding_model": EMBEDDING_MODEL_PATH, "reranker_model": RERANKER_MODEL_PATH}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=6006)
