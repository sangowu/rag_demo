"""
ingest_finqa.py
===============
将 FinQA 文档批量摄入 ChromaDB 和 BM25 索引。

每个 chunk 前自动注入 "[TICKER | YEAR]" header，与 FinanceBench 摄入格式对齐。
header 直接从 doc_id 文件名解析（无需 LLM 调用）。

Usage:
    python scripts/ingest_finqa.py [--batch-size 64]
    python scripts/ingest_finqa.py --force   # 清除旧 FinQA 数据并重新摄入

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
from src.doc_metadata_extractor import build_chunk_header
from src.ingestion_registry import IngestionRegistry
from src.vector_store import VectorStore

DOCS_DIR = _ROOT / "data" / "finqa" / "docs"


def _parse_header(doc_id: str) -> str:
    """
    从 FinQA doc_id 解析 ticker 和 year，构造 chunk header。
    doc_id 格式: "ADI_2009_page_49.pdf"
    → "[ADI | 2009]"
    """
    parts = doc_id.replace(".pdf", "").split("_")
    ticker = parts[0] if parts else ""
    year = parts[1] if len(parts) > 1 and len(parts[1]) == 4 and parts[1].isdigit() else ""
    return build_chunk_header({"ticker": ticker, "year": year})


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=64,
                        help="每批写入 ChromaDB 的 chunk 数（默认 64）")
    parser.add_argument("--force", action="store_true",
                        help="清除 ChromaDB 中已有的 FinQA 数据并重新摄入（用于 header 格式变更后）")
    return parser.parse_args()


def main():
    args = parse_args()

    vs       = VectorStore()
    bm25     = BM25Store()
    chunker  = ChunkManager(semantic_embeddings=vs.langchain_dense_embeddings())
    registry = IngestionRegistry()

    doc_paths = sorted(DOCS_DIR.glob("*.md"))
    if not doc_paths:
        print(f"[ERROR] No .md files found in {DOCS_DIR}")
        print("  Run: python src/data_loader.py")
        sys.exit(1)

    print(f"Found {len(doc_paths)} documents.")

    # ------------------------------------------------------------------
    # --force: 清除旧的 FinQA 数据（ChromaDB + 注册表），以便重新摄入
    # ------------------------------------------------------------------
    if args.force:
        print("[--force] Clearing existing FinQA chunks from ChromaDB and registry...")
        finqa_doc_ids = [p.stem for p in doc_paths]   # "ADI_2009_page_49.pdf"
        deleted = 0
        for doc_id in tqdm(finqa_doc_ids, desc="Deleting old chunks"):
            try:
                vs.delete_by_doc_id(doc_id)
                deleted += 1
            except Exception:
                pass
        registry.unregister_many(finqa_doc_ids)
        print(f"  Cleared {deleted} doc_ids from ChromaDB and registry.")

    # ------------------------------------------------------------------
    # 摄入：读取 .md → chunk → 注入 header → 写 ChromaDB
    # ------------------------------------------------------------------
    all_chunks:  list[dict] = []
    header_map:  dict[str, str] = {}   # doc_id → header（供 BM25 build 使用）
    skipped   = 0
    new_docs  = 0

    for path in tqdm(doc_paths, desc="Ingesting docs"):
        doc_id = path.stem   # "ADI_2009_page_49.pdf"

        if registry.is_registered(doc_id):
            skipped += 1
            continue

        header = _parse_header(doc_id)
        text   = path.read_text(encoding="utf-8")
        chunks = chunker.split(text, doc_id=doc_id)
        if not chunks:
            continue

        all_chunks.extend(chunks)
        for c in chunks:
            header_map[c["doc_id"]] = header
        registry.register(doc_id)
        new_docs += 1

        # 批量写入 ChromaDB
        if len(all_chunks) >= args.batch_size:
            _flush(vs, all_chunks, header_map)
            all_chunks = []

    # 写入剩余 chunks
    if all_chunks:
        _flush(vs, all_chunks, header_map)
        print(f"  Indexed {len(all_chunks)} remaining chunks.")

    # ------------------------------------------------------------------
    # 重建 BM25（全量，保留 header）
    # ------------------------------------------------------------------
    if new_docs > 0:
        print("Rebuilding BM25 index...")
        bm25_chunks: list[dict] = []
        bm25_headers: dict[str, str] = {}
        for path in tqdm(doc_paths, desc="Building BM25"):
            doc_id = path.stem
            header = _parse_header(doc_id)
            text   = path.read_text(encoding="utf-8")
            cks    = chunker.split(text, doc_id=doc_id)
            bm25_chunks.extend(cks)
            for c in cks:
                bm25_headers[c["doc_id"]] = header
        bm25.build(bm25_chunks, header_override=bm25_headers)
        print(f"BM25 index built with {len(bm25_chunks)} chunks.")

    print(f"\nDone. New docs: {new_docs} | Skipped: {skipped}")


def _flush(vs: VectorStore, chunks: list[dict], header_map: dict[str, str]) -> None:
    """按 doc_id 分组，带各自 header 写入 ChromaDB。"""
    from collections import defaultdict
    groups: dict[str, list[dict]] = defaultdict(list)
    for c in chunks:
        groups[c["doc_id"]].append(c)
    for doc_id, doc_chunks in groups.items():
        header = header_map.get(doc_id)
        vs.add_documents(doc_chunks, header_override=header if header else None)


if __name__ == "__main__":
    main()
