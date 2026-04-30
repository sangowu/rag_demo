"""
schemas.py
==========
统一的数据结构定义，供 agent tools 和所有检索来源共用。

RetrievalResult 是所有检索 tool 的统一出口格式，无论来源是：
  - 本地 RAG（pgvector + BM25）
  - Web 搜索（Tavily / Serper）
  - 未来其他来源（数据库、API）

agent 下游只处理 list[RetrievalResult]，不关心来源。
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RetrievalResult:
    text: str                        # 证据文本（chunk 原文或 web snippet）
    source: str                      # 来源标识：doc_id 或 URL
    source_type: str                 # "local" | "web"
    score: float                     # 相关性分数（归一化到 0-1）
    metadata: dict[str, Any] = field(default_factory=dict)
    # metadata 常用 key：
    #   local: chunk_index, start_char, end_char, ticker, year
    #   web:   title, published_date, domain

    def to_dict(self) -> dict:
        return {
            "text":        self.text,
            "source":      self.source,
            "source_type": self.source_type,
            "score":       self.score,
            "metadata":    self.metadata,
            # 兼容旧字段：eval 脚本直接用 r["doc_id"] 的地方
            "doc_id":      self.metadata.get("doc_id", self.source),
            "chunk_index": self.metadata.get("chunk_index", 0),
            "start_char":  self.metadata.get("start_char", 0),
            "end_char":    self.metadata.get("end_char", 0),
        }

    @classmethod
    def from_local_dict(cls, d: dict) -> "RetrievalResult":
        """从 VectorStore/BM25 返回的 dict 构建。"""
        return cls(
            text=d.get("text", ""),
            source=d.get("doc_id", ""),
            source_type="local",
            score=float(d.get("score", 0.0)),
            metadata={
                "doc_id":      d.get("doc_id", ""),
                "chunk_index": d.get("chunk_index", 0),
                "start_char":  d.get("start_char", 0),
                "end_char":    d.get("end_char", 0),
            },
        )

    @classmethod
    def from_web_dict(cls, d: dict) -> "RetrievalResult":
        """从 Tavily / Serper 返回的 dict 构建。"""
        return cls(
            text=d.get("content") or d.get("snippet") or "",
            source=d.get("url", ""),
            source_type="web",
            score=float(d.get("score", 0.0)),
            metadata={
                "title":          d.get("title", ""),
                "published_date": d.get("published_date", ""),
                "domain":         d.get("url", "").split("/")[2] if d.get("url") else "",
            },
        )
