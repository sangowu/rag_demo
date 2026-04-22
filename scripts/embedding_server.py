"""
embedding_server.py
===================
轻量 Embedding Server，基于 FlagEmbedding + FastAPI。
接口兼容 TEI 格式，运行在 AutoDL 6006 端口（公网可访问）。

Usage:
    python scripts/embedding_server.py

Endpoints:
    POST /embed   : 批量 embed 文本，返回 dense 向量
    GET  /health  : 健康检查
"""

import os

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

MODEL_PATH = os.environ.get(
    "EMBEDDING_MODEL_PATH",
    "/root/autodl-tmp/models/BAAI/bge-m3",
)

app = FastAPI(title="BGE-M3 Embedding Server")

_model = None


def _get_model():
    global _model
    if _model is None:
        from FlagEmbedding import BGEM3FlagModel
        _model = BGEM3FlagModel(MODEL_PATH, use_fp16=True)
    return _model


class EmbedRequest(BaseModel):
    inputs: list[str] | str
    is_query: bool = False  # True: encode_queries（检索时用），False: encode_corpus（摄入时用）


@app.post("/embed")
def embed(req: EmbedRequest):
    texts = req.inputs if isinstance(req.inputs, list) else [req.inputs]
    model = _get_model()
    if req.is_query:
        result = model.encode_queries(
            texts,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
    else:
        result = model.encode_corpus(
            texts,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
    return {"embeddings": result["dense_vecs"].tolist()}


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_PATH}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=6006)
