"""
data_loader.py
==============
Download FinQA (train + dev) from GitHub and convert to:
  - data/finqa/docs/<doc_id>.md   one Markdown file per source document
  - data/finqa/eval.jsonl         one record per QA pair
"""

import json
import logging
import urllib.request
from pathlib import Path

log = logging.getLogger(__name__)

# GitHub raw URLs for FinQA splits
URLS = {
    "train": "https://raw.githubusercontent.com/czyssrs/FinQA/main/dataset/train.json",
    "dev":   "https://raw.githubusercontent.com/czyssrs/FinQA/main/dataset/dev.json",
}

DOCS_DIR = Path("data/finqa/docs")
EVAL_PATH = Path("data/finqa/eval.jsonl")


def _fetch(url: str) -> list[dict]:
    """Fetch and parse a JSON array from a URL."""
    with urllib.request.urlopen(url, timeout=30) as response:
        return json.load(response)


def _doc_id(raw_id: str) -> str:
    """
    Extract doc_id from a FinQA record id.
    e.g. "ADI/2009/page_49.pdf-1"  ->  "ADI_2009_page_49.pdf"
    """
    return raw_id.rsplit("-", 1)[0].replace("/", "_")


def _table_to_markdown(table: list[list]) -> str:
    """Convert a 2D list to a Markdown table string."""
    table_md = []
    table_md.append("| " + " | ".join(table[0]) + " |")  # header
    table_md.append("| " + " | ".join(["---"] * len(table[0])) + " |")  # separator
    for row in table[1:]:
        table_md.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(table_md)


def _build_markdown(record: dict) -> str:
    """
    Assemble the Markdown document for one FinQA record.
    Order: pre_text paragraphs -> table -> post_text paragraphs
    """
    md_parts = []
    md_parts.extend(record["pre_text"])
    if record["table"]:
        md_parts.append(_table_to_markdown(record["table"]))
    md_parts.extend(record["post_text"])
    return "\n\n".join(md_parts)


def build(docs_dir: Path = DOCS_DIR, eval_path: Path = EVAL_PATH) -> None:
    """
    Main entry point.
    Downloads FinQA train+dev, writes .md docs and eval.jsonl.
    """
    docs_dir.mkdir(parents=True, exist_ok=True)
    eval_path.parent.mkdir(parents=True, exist_ok=True)

    seen_docs: set[str] = set()   # doc_ids already written this run
    eval_records: list[dict] = []
    new_docs = 0

    for split, url in URLS.items():
        log.info("Fetching %s split ...", split)
        records = _fetch(url)

        for record in records:
            did = _doc_id(record["id"])

            if did not in seen_docs:
                doc_path = docs_dir / f"{did}.md"
                if not doc_path.exists():
                    md_content = _build_markdown(record)
                    doc_path.write_text(md_content, encoding="utf-8")
                    new_docs += 1
                seen_docs.add(did)

            record_qa = record.get("qa", {})
            eval_records.append({
                "id": record["id"],
                "question": record_qa.get("question", ""),
                "program": record_qa.get("program", ""),
                "gold_inds": record_qa.get("gold_inds", []),
                "exe_ans": record_qa.get("exe_ans", ""),
                "program_re": record_qa.get("program_re", ""),
                "doc_id": did,
            })

    with open(eval_path, "w", encoding="utf-8") as f:
        for record in eval_records:
            f.write(json.dumps(record) + "\n")

    log.info("Done. New docs written: %d | Eval records: %d", new_docs, len(eval_records))

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    build()
