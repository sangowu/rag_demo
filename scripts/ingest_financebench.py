"""
ingest_financebench.py
======================
下载 FinanceBench PDF 并摄入 ChromaDB + BM25。

流程：
  1. 从 HuggingFace 加载 FinanceBench
  2. 按 doc_name 去重（150 条 QA 对应约 25 份文档）
  3. 下载 PDF → pdfplumber 提取文本
  4. LLM 提取元信息（company_name / year）
  5. chunk → embed → 写入 ChromaDB + BM25

注意：
  - PDF 下载需要网络，耗时较长（约 25 份文档）
  - 已下载的 PDF 缓存到 data/financebench/pdfs/，不重复下载
  - 已摄入的文档通过 IngestionRegistry 跳过

Usage:
    python scripts/ingest_financebench.py [--skip-download]
"""

import argparse
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

PDF_DIR = _ROOT / "data" / "financebench" / "pdfs"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-download", action="store_true",
                        help="跳过 PDF 下载，只处理已缓存的文件")
    return parser.parse_args()


def download_pdf(url: str, dest: Path) -> bool:
    """下载 PDF 到本地，返回是否成功。"""
    try:
        import requests
        resp = requests.get(url, timeout=30,
                            headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        dest.write_bytes(resp.content)
        return True
    except Exception as e:
        print(f"  [WARN] 下载失败: {e}")
        return False


def main():
    args = parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        print("请先安装: pip install datasets")
        sys.exit(1)

    from src.bm25_store import BM25Store
    from src.chunk_manager import ChunkManager
    from src.doc_metadata_extractor import build_chunk_header, extract_doc_metadata_llm
    from src.document_loader import DocumentLoader
    from src.ingestion_registry import IngestionRegistry
    from src.vector_store import VectorStore

    PDF_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading FinanceBench...")
    ds = load_dataset("PatronusAI/financebench", split="train")

    # 按 doc_name 去重，收集所有唯一文档
    docs = {}
    for row in ds:
        name = row["doc_name"]
        if name not in docs:
            docs[name] = {
                "doc_name": name,
                "doc_link": row["doc_link"],
                "company":  row["company"],
                "period":   str(row["doc_period"]),
            }
    print(f"Unique documents: {len(docs)}")

    vs       = VectorStore()
    bm25     = BM25Store()
    chunker  = ChunkManager()
    registry = IngestionRegistry()
    loader   = DocumentLoader()

    all_chunks  = []
    new_docs    = 0
    skipped     = 0

    for doc_name, info in docs.items():
        stored_id = f"{doc_name}.pdf"  # evaluator 会给 gold_id 加 .pdf，保持一致
        if registry.is_registered(stored_id):
            skipped += 1
            continue

        pdf_path = PDF_DIR / f"{doc_name}.pdf"

        # 下载 PDF
        if not pdf_path.exists():
            if args.skip_download:
                print(f"  [SKIP] {doc_name} — PDF not cached")
                continue
            print(f"  Downloading {doc_name}...")
            if not download_pdf(info["doc_link"], pdf_path):
                continue
            time.sleep(0.5)  # 避免请求过快

        # 提取文本
        print(f"  Processing {doc_name}...")
        try:
            doc = loader.load_pdf(str(pdf_path), doc_id=stored_id)
        except Exception as e:
            print(f"  [WARN] PDF 解析失败: {e}")
            continue

        # LLM 提取元信息（或直接用 dataset 里的 company/period）
        meta = {
            "company_name": info["company"],
            "ticker":       "",        # FinanceBench 不提供 ticker，LLM 可以推断
            "year":         info["period"],
            "doc_type":     "10-K",
        }
        header = build_chunk_header(meta)

        # Chunk
        chunks = chunker.split(doc["text"], doc_id=stored_id)
        if not chunks:
            continue

        all_chunks.extend(chunks)

        # 写入 ChromaDB
        vs.add_documents(chunks, header_override=header if header else None)
        registry.register(stored_id)
        new_docs += 1
        print(f"  → {len(chunks)} chunks, header: {header.strip()}")

    # 重建 BM25
    if new_docs > 0:
        print(f"\nRebuilding BM25 with {len(all_chunks)} new chunks...")
        existing = bm25._chunks or []
        # stored_id is "doc_name.pdf"; strip suffix to look up docs dict
        header_map = {c["doc_id"]: build_chunk_header({
            "company_name": docs[c["doc_id"][:-4]]["company"],
            "year":         docs[c["doc_id"][:-4]]["period"],
        }) for c in all_chunks if c["doc_id"][:-4] in docs}
        bm25.build(existing + all_chunks, header_override=header_map)

    print(f"\nDone. New: {new_docs} | Skipped: {skipped}")
    print("Run evaluation:")
    print("  python src/evaluator.py --tag financebench "
          "--eval-path data/results/financebench_qa.jsonl --n 20 --n-retrieval 150")


if __name__ == "__main__":
    main()
