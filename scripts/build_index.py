"""
build_index.py
==============
Unified index builder for chunking strategy experiments.

Supported strategies:
  baseline    — fixed-1024 chunks (same as ingest_finqa.py baseline)
  parent_child — embed child-256, return parent-1024 in search results
  contextual  — Gemini-generated context prefix prepended to each chunk
  summary     — Gemini-generated doc summary prepended to each chunk

Usage:
    python scripts/build_index.py --strategy parent_child
    python scripts/build_index.py --strategy contextual --sample 50
    python scripts/build_index.py --strategy summary --force
    python scripts/build_index.py --strategy baseline --force

Options:
  --strategy   {baseline,parent_child,contextual,summary}  (required)
  --force      Clear existing index before rebuilding
  --sample N   Only process N randomly sampled documents (for quick tests)
  --batch-size Chunks per PostgreSQL write batch (default 64)
"""

import argparse
import random
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

_DATASET_DOCS: dict[str, Path] = {
    "finqa":        _ROOT / "data" / "finqa"        / "docs",
    "financebench": _ROOT / "data" / "financebench" / "docs",
}

STRATEGIES = ("baseline", "parent_child", "contextual", "summary")


def _parse_header(doc_id: str) -> str:
    parts = doc_id.replace(".pdf", "").split("_")
    ticker = parts[0] if parts else ""
    year = parts[1] if len(parts) > 1 and len(parts[1]) == 4 and parts[1].isdigit() else ""
    return build_chunk_header({"ticker": ticker, "year": year})


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy",   required=True, choices=STRATEGIES)
    parser.add_argument("--force",      action="store_true", help="Clear existing index first")
    parser.add_argument("--sample",     type=int, default=200,
                        help="Docs per dataset (default 200, seed=42); 0 = all docs")
    parser.add_argument("--datasets",   type=str, default="finqa",
                        help="Comma-separated datasets: finqa,financebench (default: finqa)")
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


def _get_doc_paths(sample: int, datasets: list[str]) -> list[Path]:
    all_paths: list[Path] = []
    random.seed(42)
    for ds in datasets:
        docs_dir = _DATASET_DOCS.get(ds)
        if docs_dir is None:
            print(f"[WARN] Unknown dataset: {ds}")
            continue
        paths = sorted(docs_dir.glob("*.md"))
        if not paths:
            print(f"[ERROR] No .md files in {docs_dir} for dataset '{ds}'.")
            if ds == "finqa":
                print("  Run: python src/data_loader.py")
            elif ds == "financebench":
                print("  Run: python scripts/convert_financebench.py")
            continue
        if sample > 0:
            sampled = random.sample(paths, min(sample, len(paths)))
            print(f"[{ds}] Sampled {len(sampled)}/{len(paths)} docs (seed=42)")
            all_paths.extend(sampled)
        else:
            print(f"[{ds}] All {len(paths)} docs (full run)")
            all_paths.extend(paths)
    if not all_paths:
        print("[ERROR] No documents found across all datasets.")
        sys.exit(1)
    return all_paths


def _strategy_resources(strategy: str) -> tuple[VectorStore, BM25Store, IngestionRegistry]:
    """Return strategy-specific DB table, BM25 file, and registry."""
    table    = f"chunks_{strategy}"
    bm25_path    = _ROOT / "data" / f"bm25_{strategy}.pkl"
    registry_path = _ROOT / "data" / f"registry_{strategy}.json"
    return (
        VectorStore(table_name=table),
        BM25Store(index_path=bm25_path),
        IngestionRegistry(registry_path=registry_path),
    )


def _clear_index(vs: VectorStore, registry: IngestionRegistry) -> None:
    """TRUNCATE the strategy table and wipe its registry — unaffected by --sample."""
    print(f"[--force] Truncating table '{vs._table}' and clearing registry...")
    vs.clear_all()
    registry.clear()
    print("  Done.")


def _build_baseline(
    doc_paths: list[Path],
    vs: VectorStore,
    bm25: BM25Store,
    registry: IngestionRegistry,
    batch_size: int,
) -> None:
    """Fixed-1024 chunks — same as the original ingest_finqa.py behavior."""
    cm = ChunkManager(semantic_embeddings=vs.langchain_dense_embeddings())
    all_chunks: list[dict] = []

    for path in tqdm(doc_paths, desc="Chunking"):
        doc_id = path.stem
        if registry.is_registered(doc_id):
            continue
        text = path.read_text(encoding="utf-8")
        header = _parse_header(doc_id)
        chunks = cm.split(text, doc_id=doc_id)
        for c in chunks:
            c["_header"] = header
        all_chunks.extend(chunks)
        registry.register(doc_id)

    print(f"Indexing {len(all_chunks)} chunks...")
    for i in tqdm(range(0, len(all_chunks), batch_size), desc="Writing to DB"):
        batch = all_chunks[i : i + batch_size]
        vs.add_documents(batch)

    print("Rebuilding BM25...")
    _rebuild_bm25(bm25, doc_paths, registry)


def _build_parent_child(
    doc_paths: list[Path],
    vs: VectorStore,
    bm25: BM25Store,
    registry: IngestionRegistry,
    batch_size: int,
) -> None:
    """Embed child-256 chunks, store parent-1024 text as content."""
    cm = ChunkManager()  # parent_size from config (default 1024)
    all_chunks: list[dict] = []

    for path in tqdm(doc_paths, desc="Chunking (parent-child)"):
        doc_id = path.stem
        if registry.is_registered(doc_id):
            continue
        text = path.read_text(encoding="utf-8")
        chunks = cm.split_parent_child(text, doc_id=doc_id)
        all_chunks.extend(chunks)
        registry.register(doc_id)

    print(f"Indexing {len(all_chunks)} child chunks (content = parent text)...")
    for i in tqdm(range(0, len(all_chunks), batch_size), desc="Writing to DB"):
        batch = all_chunks[i : i + batch_size]
        vs.add_documents(batch)

    print("Rebuilding BM25 (child text for sparse retrieval)...")
    _rebuild_bm25(bm25, doc_paths, registry)


def _build_contextual(
    doc_paths: list[Path],
    vs: VectorStore,
    bm25: BM25Store,
    registry: IngestionRegistry,
    batch_size: int,
) -> None:
    """Embed (Gemini context prefix + chunk), store original chunk as content."""
    from src.contextual_chunker import ContextualChunker

    cm = ChunkManager()
    cc = ContextualChunker()
    all_chunks: list[dict] = []

    for path in tqdm(doc_paths, desc="Contextual chunking (Gemini)"):
        doc_id = path.stem
        if registry.is_registered(doc_id):
            continue
        text = path.read_text(encoding="utf-8")
        chunks = cm.split(text, doc_id=doc_id)
        enriched = cc.enrich(text, doc_id, chunks)
        all_chunks.extend(enriched)
        registry.register(doc_id)

    print(f"Indexing {len(all_chunks)} contextual chunks...")
    for i in tqdm(range(0, len(all_chunks), batch_size), desc="Writing to DB"):
        batch = all_chunks[i : i + batch_size]
        vs.add_documents(batch)

    print("Rebuilding BM25 (original chunk text)...")
    _rebuild_bm25(bm25, doc_paths, registry)


def _build_summary(
    doc_paths: list[Path],
    vs: VectorStore,
    bm25: BM25Store,
    registry: IngestionRegistry,
    batch_size: int,
) -> None:
    """
    Summary-augmented embedding: embed (Gemini doc summary + chunk), return chunk as content.

    One Gemini call per document to generate a 2-3 sentence summary.
    The summary is prepended to EVERY chunk in that document before embedding,
    giving each chunk global document context in its vector.
    Summaries are cached to data/doc_summaries.jsonl for re-use.
    """
    import json

    from langchain_core.messages import HumanMessage, SystemMessage

    from src.llm_factory import get_llm

    _SUMMARY_CACHE = _ROOT / "data" / "doc_summaries.jsonl"
    _SYSTEM = (
        "You are a financial document indexer. "
        "Write 2-3 sentences summarizing: the company name, fiscal year, "
        "and main financial topics (revenue, income, segments, ratios). "
        "Output only the summary, no extra text."
    )
    llm = get_llm(temperature=0.1, max_output_tokens=256)

    # Load cached summaries
    cache: dict[str, str] = {}
    if _SUMMARY_CACHE.exists():
        with open(_SUMMARY_CACHE, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                cache[rec["doc_id"]] = rec["summary"]
    print(f"Loaded {len(cache)} cached summaries.")

    cm = ChunkManager()
    all_chunks: list[dict] = []

    with open(_SUMMARY_CACHE, "a", encoding="utf-8") as cache_f:
        for path in tqdm(doc_paths, desc="Summary chunking (Gemini)"):
            doc_id = path.stem
            if registry.is_registered(doc_id):
                continue
            text = path.read_text(encoding="utf-8")

            # Generate or retrieve summary
            if doc_id not in cache:
                try:
                    resp = llm.invoke([
                        SystemMessage(_SYSTEM),
                        HumanMessage(f"doc_id: {doc_id}\n\n{text[:2000]}"),
                    ])
                    content = resp.content
                    if isinstance(content, list):
                        content = " ".join(
                            p.get("text", "") if isinstance(p, dict) else str(p)
                            for p in content
                        )
                    summary = content.strip()
                except Exception as e:
                    tqdm.write(f"[SKIP summary] {doc_id}: {e}")
                    summary = ""
                cache[doc_id] = summary
                cache_f.write(json.dumps({"doc_id": doc_id, "summary": summary}, ensure_ascii=False) + "\n")
                cache_f.flush()

            summary = cache.get(doc_id, "")
            chunks = cm.split(text, doc_id=doc_id)
            for chunk in chunks:
                embed_text = (summary + "\n\n" + chunk["text"]) if summary else chunk["text"]
                all_chunks.append({
                    **chunk,
                    "embed_text": embed_text,
                    "content": chunk["text"],
                })
            registry.register(doc_id)

    print(f"Indexing {len(all_chunks)} summary-augmented chunks...")
    for i in tqdm(range(0, len(all_chunks), batch_size), desc="Writing to DB"):
        batch = all_chunks[i : i + batch_size]
        vs.add_documents(batch)

    print("Rebuilding BM25 (original chunk text)...")
    _rebuild_bm25(bm25, doc_paths, registry)


def _rebuild_bm25(
    bm25: BM25Store,
    doc_paths: list[Path],
    registry: IngestionRegistry,
) -> None:
    """Rebuild BM25 index from registered docs using original chunk text."""
    cm = ChunkManager()
    all_bm25_chunks: list[dict] = []
    registered = registry.list_all()

    for path in doc_paths:
        doc_id = path.stem
        if doc_id not in registered:
            continue
        text = path.read_text(encoding="utf-8")
        chunks = cm.split(text, doc_id=doc_id)
        all_bm25_chunks.extend(chunks)

    bm25.build(all_bm25_chunks)
    print(f"BM25 index rebuilt: {len(all_bm25_chunks)} chunks.")


def main():
    args = parse_args()
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    doc_paths = _get_doc_paths(args.sample, datasets)

    vs, bm25, registry = _strategy_resources(args.strategy)

    if args.force:
        _clear_index(vs, registry)

    print(f"\n=== Strategy: {args.strategy}  (table: {vs._table}) ===\n")

    if args.strategy == "baseline":
        _build_baseline(doc_paths, vs, bm25, registry, args.batch_size)
    elif args.strategy == "parent_child":
        _build_parent_child(doc_paths, vs, bm25, registry, args.batch_size)
    elif args.strategy == "contextual":
        _build_contextual(doc_paths, vs, bm25, registry, args.batch_size)
    elif args.strategy == "summary":
        _build_summary(doc_paths, vs, bm25, registry, args.batch_size)

    print("\nDone.")


if __name__ == "__main__":
    main()
