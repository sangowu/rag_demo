"""
ingest_finqa.py
===============
将 FinQA 文档批量摄入 ChromaDB 和 BM25 索引。

流程：
  1. 从 data/finqa/docs/ 读取所有 .md 文档
  2. 用 ChunkManager 切分
  3. 写入 VectorStore（ChromaDB）
  4. 写入 BM25Store
  5. 用 IngestionRegistry 记录已摄入文档，避免重复

Usage:
    python scripts/ingest_finqa.py [--batch-size 64]

注意：首次运行前请先执行数据下载：
    python src/data_loader.py
"""

import argparse
import sys
from pathlib import Path

from tqdm import tqdm

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from src.bm25_store import BM25Store
from src.chunk_manager import ChunkManager
from src.ingestion_registry import IngestionRegistry
from src.vector_store import VectorStore

DOCS_DIR = _ROOT / "data" / "finqa" / "docs"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=64,
                        help="每批写入 ChromaDB 的 chunk 数（默认 64）")
    return parser.parse_args()


def main():
    args = parse_args()

    vs       = VectorStore()
    bm25     = BM25Store()
    chunker  = ChunkManager()
    registry = IngestionRegistry()

    doc_paths = sorted(DOCS_DIR.glob("*.md"))
    if not doc_paths:
        print(f"[ERROR] No .md files found in {DOCS_DIR}")
        print("  Run: python src/data_loader.py")
        sys.exit(1)

    print(f"Found {len(doc_paths)} documents.")

    all_chunks   = []
    skipped      = 0
    new_docs     = 0

    for path in tqdm(doc_paths, desc="Ingesting docs"):
        doc_id = path.stem   # 文件名去掉 .md 后缀

        if registry.is_registered(doc_id):
            skipped += 1
            continue

        text   = path.read_text(encoding="utf-8")
        chunks = chunker.split(text, doc_id=doc_id)
        all_chunks.extend(chunks)
        registry.register(doc_id)
        new_docs += 1

        # 批量写入 ChromaDB
        if len(all_chunks) >= args.batch_size:
            vs.add_documents(all_chunks)
            all_chunks = []

    # 写入剩余 chunks
    if all_chunks:
        vs.add_documents(all_chunks)
        print(f"  Indexed {len(all_chunks)} remaining chunks.")

    # 重建 BM25 索引（一次性全量建）
    if new_docs > 0:
        print("Rebuilding BM25 index...")
        new_chunks_all = []
        for path in tqdm(doc_paths, desc="Building BM25"):
            doc_id = path.stem
            text   = path.read_text(encoding="utf-8")
            new_chunks_all.extend(chunker.split(text, doc_id=doc_id))
        bm25.build(new_chunks_all)
        print(f"BM25 index built with {len(new_chunks_all)} chunks.")

    print(f"\nDone. New docs: {new_docs} | Skipped: {skipped}")


if __name__ == "__main__":
    main()
