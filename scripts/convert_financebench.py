"""
convert_financebench.py
=======================
Convert FinanceBench PDFs to Markdown and build eval.jsonl.

Steps:
  1. Build qa_filename -> local_pdf_stem mapping
  2. Convert matched PDFs to Markdown (data/financebench/docs/)
  3. Write data/financebench/eval.jsonl with doc_id = local PDF stem

Usage:
    python scripts/convert_financebench.py [--force]

Options:
  --force   Re-convert already-existing Markdown files
"""

import argparse
import json
import re
import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from src.document_loader import DocumentLoader

_PDFS_DIR  = _ROOT / "data" / "financebench" / "pdfs"
_DOCS_DIR  = _ROOT / "data" / "financebench" / "docs"
_QA_PATH   = _ROOT / "data" / "financebench" / "qa_pairs.jsonl"
_EVAL_PATH = _ROOT / "data" / "financebench" / "eval.jsonl"

# ------------------------------------------------------------------ #
# Company name normalization                                           #
# ------------------------------------------------------------------ #
_ABBR = {
    "activision blizzard": "activisionblizzard",
    "american express": "americanexpress",
    "american water works": "americanwaterworks",
    "aes corporation": "aes",
    "best buy": "bestbuy",
    "coca cola": "cocacola",
    "coca-cola": "cocacola",
    "cvs health": "cvshealth",
    "foot locker": "footlocker",
    "general mills": "generalmills",
    "johnson & johnson": "johnsonandjohnson",
    "johnson and johnson": "johnsonandjohnson",
    "jp morgan": "jpmorgan",
    "jpmorgan chase": "jpmorgan",
    "kraft heinz": "kraftheinz",
    "lockheed martin": "lockheedmartin",
    "mgm resorts": "mgmresorts",
    "ulta beauty": "ultabeauty",
}


def _norm_company(name: str) -> str:
    n = name.lower().strip()
    for k, v in _ABBR.items():
        if k in n:
            return v
    return re.sub(r"[^a-z0-9]", "", n)


def _norm_doctype(t: str) -> str:
    t = t.lower().strip()
    if t in ("10k", "10-k"):
        return "10k"
    if t in ("10q", "10-q"):
        return "10q"
    if t == "8k":
        return "8k"
    if t in ("earnings", "earnings release"):
        return "earnings"
    return re.sub(r"[^a-z0-9]", "", t)


def build_mapping(qa_records: list[dict]) -> dict[str, str | None]:
    """Return {qa_pdf_filename: local_pdf_stem_or_None}."""
    local_pdfs = {p.name: p for p in _PDFS_DIR.glob("*.pdf")}
    mapping: dict[str, str | None] = {}

    for rec in qa_records:
        qa_fn = rec["pdf_filename"]
        if qa_fn in mapping:
            continue

        nc = _norm_company(rec.get("company", ""))
        nt = _norm_doctype(rec.get("doc_type", ""))
        year = str(rec.get("doc_period", ""))

        candidates = []
        for local_name, local_path in local_pdfs.items():
            ln = local_name.upper()
            if nc.upper() in ln and year in ln:
                if nt.upper() in ln:
                    candidates.append(local_path.stem)

        if candidates:
            mapping[qa_fn] = candidates[0]  # pick first if ambiguous
        else:
            mapping[qa_fn] = None

    return mapping


def convert_pdfs(mapping: dict[str, str | None], force: bool) -> None:
    """Convert matched PDFs to Markdown in _DOCS_DIR."""
    _DOCS_DIR.mkdir(parents=True, exist_ok=True)
    loader = DocumentLoader()
    seen: set[str] = set()

    for qa_fn, stem in mapping.items():
        if stem is None or stem in seen:
            continue
        seen.add(stem)

        md_path = _DOCS_DIR / f"{stem}.md"
        if md_path.exists() and not force:
            print(f"[SKIP] {stem}.md already exists")
            continue

        pdf_path = _PDFS_DIR / f"{stem}.pdf"
        if not pdf_path.exists():
            # case-insensitive fallback
            matches = list(_PDFS_DIR.glob(f"{stem}.pdf"))
            if not matches:
                print(f"[WARN] PDF not found for stem: {stem}")
                continue
            pdf_path = matches[0]

        try:
            doc = loader.load_pdf(str(pdf_path), doc_id=stem)
            md_path.write_text(doc["text"], encoding="utf-8")
            print(f"[OK] {stem}.md  ({len(doc['text'])} chars)")
        except Exception as e:
            print(f"[ERROR] {stem}: {e}")


def build_eval_jsonl(qa_records: list[dict], mapping: dict[str, str | None]) -> None:
    """Write eval.jsonl with question / doc_id pairs for matched records."""
    written = 0
    with open(_EVAL_PATH, "w", encoding="utf-8") as f:
        for rec in qa_records:
            qa_fn = rec["pdf_filename"]
            stem = mapping.get(qa_fn)
            if stem is None:
                continue
            out = {
                "question": rec["question"],
                "answer": rec.get("answer", ""),
                "doc_id": stem,
                "source": "financebench",
                "financebench_id": rec.get("financebench_id", ""),
            }
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
            written += 1
    print(f"\nWrote {written} eval records -> {_EVAL_PATH}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Re-convert existing Markdown files")
    args = parser.parse_args()

    qa_records = [json.loads(l) for l in _QA_PATH.open(encoding="utf-8") if l.strip()]
    print(f"Loaded {len(qa_records)} QA pairs")

    mapping = build_mapping(qa_records)
    matched = {k: v for k, v in mapping.items() if v is not None}
    print(f"Mapped {len(matched)}/{len(mapping)} unique filenames -> local PDFs\n")

    print("=== Converting PDFs to Markdown ===")
    convert_pdfs(mapping, force=args.force)

    print("\n=== Building eval.jsonl ===")
    build_eval_jsonl(qa_records, mapping)


if __name__ == "__main__":
    main()
