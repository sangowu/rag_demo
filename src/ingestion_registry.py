"""
ingestion_registry.py
=====================
记录已摄入文档的注册表，避免重复写入 ChromaDB 和 BM25。

持久化方式：JSON 文件（data/ingestion_registry.json）
数据结构：{"doc_ids": ["doc_001", "doc_002", ...]}

Usage:
    from src.ingestion_registry import IngestionRegistry
    registry = IngestionRegistry()
    if not registry.is_registered("doc_001"):
        registry.register("doc_001")
"""

import json
from pathlib import Path

from src.config import config

_ROOT = Path(__file__).parent.parent
_REGISTRY_PATH = _ROOT / config.get("ingestion", {}).get(
    "registry_path", "data/ingestion_registry.json"
)


class IngestionRegistry:
    def __init__(self):
        self._doc_ids = set()
        if _REGISTRY_PATH.exists():
            with open(_REGISTRY_PATH, "r", encoding="UTF-8") as f:
                file = json.load(f)
            self._doc_ids = set(file.get("doc_ids", []))

    def is_registered(self, doc_id: str) -> bool:
        """检查 doc_id 是否已摄入。"""
        return doc_id in self._doc_ids

    def register(self, doc_id: str) -> None:
        """标记 doc_id 为已摄入，立即写盘。"""
        self._doc_ids.add(doc_id)
        self._save()

    def list_all(self) -> list[str]:
        """返回全部已摄入的 doc_id 列表。"""
        return sorted(self._doc_ids)

    def _save(self) -> None:
        """将当前注册表写入 JSON 文件。"""
        _REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(_REGISTRY_PATH, "w", encoding="UTF-8") as f:
            json.dump({"doc_ids": sorted(self._doc_ids)}, f, ensure_ascii=False)
