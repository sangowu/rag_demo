"""
doc_summarizer.py
=================
批量为 FinQA 文档生成 1-2 句 summary，保存到 data/doc_summaries.jsonl。
Summary 用于两阶段检索的 Stage 1：先找相关文档，再在文档内检索 chunk。

每个文档只调用一次 LLM，结果持久化，支持断点续跑（已有 summary 的文档跳过）。

Usage:
    python src/doc_summarizer.py [--batch 50] [--output data/doc_summaries.jsonl]
"""

import argparse
import json
import sys
from pathlib import Path

from tqdm import tqdm

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from langchain_core.messages import HumanMessage, SystemMessage

from src.llm_factory import get_llm

_DOCS_DIR = _ROOT / "data/finqa/docs"

_llm = get_llm(temperature=0.1, max_output_tokens=256)

_SYSTEM = (
    "You are a financial document indexer. "
    "Given a financial report excerpt, write exactly 1-2 sentences summarizing: "
    "the company name, fiscal year, and main financial topics covered. "
    "Be concise and factual. Output only the summary, no extra text."
)


def _summarize(doc_id: str, text: str) -> str:
    """调用 LLM 生成单个文档的 summary。"""
    snippet = text[:2000]  # 只取前 2000 字符，节省 token
    response = _llm.invoke([
        SystemMessage(_SYSTEM),
        HumanMessage(f"doc_id: {doc_id}\n\n{snippet}"),
    ])
    return response.content.strip()


def build_summaries(output_path: Path, batch_size: int = 50) -> None:
    """
    遍历所有 .md 文档，生成 summary 并追加写入 output_path。
    已有 summary 的文档（按 doc_id 去重）自动跳过，支持断点续跑。
    """
    # 加载已有结果（断点续跑）
    existing: dict[str, str] = {}
    if output_path.exists():
        with open(output_path, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                existing[rec["doc_id"]] = rec["summary"]
    print(f"Already summarized: {len(existing)} docs")

    doc_paths = sorted(_DOCS_DIR.glob("*.md"))
    pending = [p for p in doc_paths if p.name not in existing]
    print(f"Pending: {len(pending)} docs")

    if not pending:
        print("All docs already summarized.")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "a", encoding="utf-8") as out_f:
        for i, path in enumerate(tqdm(pending, desc="Summarizing")):
            doc_id = path.name
            text   = path.read_text(encoding="utf-8")
            try:
                summary = _summarize(doc_id, text)
            except Exception as e:
                tqdm.write(f"[SKIP] {doc_id}: {e}")
                continue

            out_f.write(json.dumps({"doc_id": doc_id, "summary": summary}, ensure_ascii=False) + "\n")

            # 每 batch_size 条 flush 一次，防止崩溃丢失
            if (i + 1) % batch_size == 0:
                out_f.flush()

    print(f"Done. Saved to {output_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch",  type=int, default=50,   help="flush 间隔（默认 50）")
    parser.add_argument("--output", type=str, default="data/doc_summaries.jsonl")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_summaries(_ROOT / args.output, batch_size=args.batch)
