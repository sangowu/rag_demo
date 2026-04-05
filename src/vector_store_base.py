"""
vector_store_base.py
====================
向量库抽象接口（Protocol）。

工业实践中 RAG 系统通常需要支持多种向量库后端（ChromaDB / Pinecone / Weaviate /
Qdrant），通过抽象接口隔离业务代码与具体实现，切换后端无需修改上层逻辑。

当前实现：VectorStore（ChromaDB）
可扩展至：PineconeVectorStore / QdrantVectorStore 等，实现相同接口即可。

Usage:
    from src.vector_store_base import VectorStoreBase
    def build_retriever(vs: VectorStoreBase): ...   # 依赖接口，不依赖实现
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class VectorStoreBase(Protocol):
    """
    向量库后端的最小接口契约。
    任何实现了以下方法的类均自动满足此 Protocol，无需显式继承。
    """

    def add_documents(
        self,
        chunks: list[dict],
        header_override: str | dict | None = None,
    ) -> None:
        """
        将文档 chunks 写入向量库。

        Args:
            chunks          : ChunkManager.split() 返回的 chunk 列表，
                              每项含 text / doc_id / chunk_index 字段。
            header_override : 注入每个 chunk 文本前的 header 字符串，
                              或 {doc_id: header} 映射（不同文档不同 header）。
        """
        ...

    def search(
        self,
        query: str,
        top_k: int = 10,
        where: dict | None = None,
    ) -> list[dict]:
        """
        语义搜索，返回最相似的 top_k 个 chunk。

        Args:
            query : 查询文本
            top_k : 返回数量
            where : 元数据过滤条件（ChromaDB where 语法）

        Returns:
            list of dicts，每项含 text / doc_id / score 等字段。
        """
        ...

    def delete_by_doc_id(self, doc_id: str) -> None:
        """删除指定 doc_id 的所有 chunks。"""
        ...

    def offload(self) -> None:
        """释放 GPU 显存（模型卸载）。推理结束后调用，与 BM25 / Reranker 错峰使用。"""
        ...
