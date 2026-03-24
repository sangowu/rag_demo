"""
Download FinanceBench open-source subset (150 QA pairs + source PDFs).

Output layout:
  data/financebench/
  ├── qa_pairs.jsonl          # 150 QA records (question, answer, evidence, metadata)
  ├── pdfs/                   # raw PDF files, named by company+period
  └── download_log.json       # success/failure per document
"""

import json
import time
import hashlib
import argparse
import logging
from pathlib import Path

import requests
from datasets import load_dataset

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).parent.parent / "data" / "financebench"
PDF_DIR = DATA_DIR / "pdfs"
QA_PATH = DATA_DIR / "qa_pairs.jsonl"
LOG_PATH = DATA_DIR / "download_log.json"

HF_DATASET = "PatronusAI/financebench"
REQUEST_TIMEOUT = 60          # seconds per PDF download attempt
RETRY_LIMIT = 3
RETRY_BACKOFF = 5             # seconds between retries

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_filename(company: str, doc_type: str, period: str) -> str:
    """Build a filesystem-safe PDF filename."""
    raw = f"{company}_{doc_type}_{period}".replace("/", "-").replace(" ", "_")
    return raw[:120] + ".pdf"          # guard against overly long names


def _url_hash(url: str) -> str:
    return hashlib.md5(url.encode()).hexdigest()[:8]


def download_pdf(url: str, dest: Path, retries: int = RETRY_LIMIT) -> bool:
    """Download a single PDF with retry logic. Returns True on success."""
    headers = {"User-Agent": "Mozilla/5.0 (research bot; contact: local)"}
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT, stream=True)
            resp.raise_for_status()
            content_type = resp.headers.get("Content-Type", "")
            if "pdf" not in content_type.lower() and not url.lower().endswith(".pdf"):
                log.warning("Unexpected content-type '%s' for %s", content_type, url)
            dest.write_bytes(resp.content)
            return True
        except Exception as exc:
            log.warning("Attempt %d/%d failed for %s: %s", attempt, retries, url, exc)
            if attempt < retries:
                time.sleep(RETRY_BACKOFF * attempt)
    return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(skip_pdfs: bool = False, force: bool = False) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    PDF_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Load QA pairs from HuggingFace ──────────────────────────────────
    log.info("Loading FinanceBench dataset from HuggingFace (%s) …", HF_DATASET)
    ds = load_dataset(HF_DATASET, split="train")
    log.info("Loaded %d records", len(ds))

    records = []
    url_to_filename: dict[str, str] = {}   # deduplicate PDFs across records

    for row in ds:
        doc_link = row.get("doc_link") or row.get("doc_url") or ""
        company  = row.get("company", "unknown")
        doc_type = row.get("doc_type", "doc")
        period   = row.get("doc_period") or row.get("year", "")

        # Determine target filename (share across rows with the same URL)
        if doc_link and doc_link not in url_to_filename:
            fname = _safe_filename(company, doc_type, period)
            # Avoid collision between different companies that hash to same name
            candidate = PDF_DIR / fname
            if candidate.exists() and not force:
                # File already there from a previous run — reuse name
                url_to_filename[doc_link] = fname
            else:
                url_to_filename[doc_link] = fname

        record = {
            "financebench_id": row.get("financebench_id", ""),
            "question":        row.get("question", ""),
            "answer":          row.get("answer", ""),
            "justification":   row.get("justification", ""),
            "evidence_text":   row.get("evidence_text", ""),
            "page_number":     row.get("page_number", ""),
            "company":         company,
            "doc_type":        doc_type,
            "doc_period":      period,
            "doc_link":        doc_link,
            "pdf_filename":    url_to_filename.get(doc_link, ""),
        }
        records.append(record)

    # Write QA pairs
    with QA_PATH.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    log.info("Saved %d QA pairs → %s", len(records), QA_PATH)

    # ── 2. Download PDFs ────────────────────────────────────────────────────
    if skip_pdfs:
        log.info("--skip-pdfs set, skipping PDF downloads.")
        return

    unique_docs = {r["doc_link"]: r["pdf_filename"]
                   for r in records if r["doc_link"]}
    log.info("Unique documents to download: %d", len(unique_docs))

    download_log: dict[str, dict] = {}

    for idx, (url, fname) in enumerate(unique_docs.items(), 1):
        dest = PDF_DIR / fname
        status_prefix = f"[{idx}/{len(unique_docs)}]"

        if dest.exists() and not force:
            log.info("%s SKIP (already exists): %s", status_prefix, fname)
            download_log[url] = {"status": "skipped", "file": fname}
            continue

        log.info("%s Downloading %s …", status_prefix, url)
        ok = download_pdf(url, dest)

        if ok:
            size_mb = dest.stat().st_size / 1_048_576
            log.info("  ✓ %.1f MB → %s", size_mb, fname)
            download_log[url] = {"status": "ok", "file": fname, "size_mb": round(size_mb, 2)}
        else:
            log.error("  ✗ Failed: %s", url)
            download_log[url] = {"status": "failed", "file": fname}

    # Write download log
    LOG_PATH.write_text(json.dumps(download_log, indent=2, ensure_ascii=False), encoding="utf-8")

    # ── 3. Summary ─────────────────────────────────────────────────────────
    ok_count   = sum(1 for v in download_log.values() if v["status"] == "ok")
    skip_count = sum(1 for v in download_log.values() if v["status"] == "skipped")
    fail_count = sum(1 for v in download_log.values() if v["status"] == "failed")

    log.info("─" * 50)
    log.info("Done.  ✓ downloaded: %d  ⊘ skipped: %d  ✗ failed: %d",
             ok_count, skip_count, fail_count)
    if fail_count:
        log.warning("Failed URLs logged in %s", LOG_PATH)


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download FinanceBench dataset")
    parser.add_argument(
        "--skip-pdfs", action="store_true",
        help="Only download QA pairs, skip PDF files"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-download files even if they already exist"
    )
    args = parser.parse_args()
    main(skip_pdfs=args.skip_pdfs, force=args.force)
