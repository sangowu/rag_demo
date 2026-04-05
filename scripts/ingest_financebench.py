"""
ingest_financebench.py
======================
从 FinanceBench dataset 的 evidence_text_full_page 直接摄入 ChromaDB + BM25。

不需要下载 PDF，使用官方预处理的 evidence 页面文本。

流程：
  1. 从 HuggingFace 加载 FinanceBench
  2. 按 (doc_name, evidence_page_num) 去重，收集所有唯一 evidence 页
  3. 构造 doc_id = "{doc_name}_page_{page_num}.pdf"（与 evaluator gold_id 对齐）
  4. LLM 提取 ticker，构造 header
  5. chunk → embed → 写入 ChromaDB + BM25

doc_id 格式：{doc_name}_page_{evidence_page_num}.pdf
  例：3M_2018_10K_page_59.pdf

Usage:
    python scripts/ingest_financebench.py
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true",
                        help="清除已有 FinanceBench 数据并重新摄入")
    args = parser.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        print("请先安装: pip install datasets")
        sys.exit(1)

    from src.bm25_store import BM25Store
    from src.chunk_manager import ChunkManager
    from src.doc_metadata_extractor import build_chunk_header, extract_doc_metadata_llm
    from src.ingestion_registry import IngestionRegistry
    from src.vector_store import VectorStore

    print("Loading FinanceBench...")
    ds = load_dataset("PatronusAI/financebench", split="train")

    # 收集所有唯一 evidence 页：(doc_name, page_num) → page info
    pages = {}
    for row in ds:
        doc_name = row["doc_name"]
        company  = row["company"]
        period   = str(row["doc_period"])
        for ev in (row["evidence"] or []):
            page_num  = ev.get("evidence_page_num", -1)
            # evidence_text：官方标注的答案片段，与 FinQA 的 pre_text+table+post_text 逻辑一致
            # 比 evidence_text_full_page（整页）更短，噪声更少，不会触发 context 超限
            full_text = ev.get("evidence_text", "").strip()
            if page_num < 0 or not full_text:
                continue
            key = (doc_name, page_num)
            if key not in pages:
                pages[key] = {
                    "text":    full_text,
                    "company": company,
                    "period":  period,
                }

    print(f"Unique evidence pages: {len(pages)}")

    vs       = VectorStore()
    bm25     = BM25Store()
    chunker  = ChunkManager(semantic_embeddings=vs.langchain_dense_embeddings())
    registry = IngestionRegistry()

    # --force：清除旧的 FinanceBench 数据，重新摄入
    if args.force:
        fb_page_ids = [
            f"{doc_name}_page_{page_num}.pdf"
            for (doc_name, page_num) in pages.keys()
        ]
        print(f"[--force] Clearing {len(fb_page_ids)} FinanceBench page_ids...")
        for pid in fb_page_ids:
            try:
                vs.delete_by_doc_id(pid)
            except Exception:
                pass
        registry.unregister_many(fb_page_ids)
        print("Done. Re-ingesting...")

    all_chunks   = []
    header_cache = {}   # page_id → header str
    ticker_cache = {}   # doc_name → ticker（每份文档只调一次 LLM）
    new_pages    = 0
    skipped      = 0

    for (doc_name, page_num), info in pages.items():
        page_id = f"{doc_name}_page_{page_num}.pdf"

        if registry.is_registered(page_id):
            skipped += 1
            continue

        # LLM 提取 ticker（每份文档只提取一次）
        if doc_name not in ticker_cache:
            llm_meta = extract_doc_metadata_llm(info["text"])
            ticker_cache[doc_name] = llm_meta.get("ticker", "")
            if ticker_cache[doc_name]:
                print(f"  [{doc_name}] ticker: {ticker_cache[doc_name]}")

        meta = {
            "company_name": info["company"],
            "ticker":       ticker_cache[doc_name],
            "year":         info["period"],
        }
        header = build_chunk_header(meta)

        chunks = chunker.split(info["text"], doc_id=page_id)
        if not chunks:
            continue

        all_chunks.extend(chunks)
        for c in chunks:
            header_cache[c["doc_id"]] = header

        vs.add_documents(chunks, header_override=header if header else None)
        registry.register(page_id)
        new_pages += 1

    print(f"\nIngested {new_pages} new pages | Skipped {skipped}")

    # 重建 BM25
    if new_pages > 0:
        print(f"Rebuilding BM25 with {len(all_chunks)} new chunks...")
        existing = bm25._chunks or []
        header_map = {c["doc_id"]: header_cache[c["doc_id"]]
                      for c in all_chunks if c["doc_id"] in header_cache}
        bm25.build(existing + all_chunks, header_override=header_map)
        print("BM25 rebuilt.")

    print("\nRun evaluation:")
    print("  python src/evaluator.py --tag financebench_evidence "
          "--eval-path data/results/financebench_qa.jsonl --n 20 --n-retrieval 150")


if __name__ == "__main__":
    main()
