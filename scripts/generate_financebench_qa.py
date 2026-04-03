"""
generate_financebench_qa.py
===========================
将 FinanceBench 数据集转换为 evaluator 兼容的 QA pairs 格式。

输出：data/results/financebench_qa.jsonl
格式：{"question": "...", "answer": "...", "doc_id": "3M_2018_10K"}

Usage:
    python scripts/generate_financebench_qa.py
"""

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

OUTPUT_PATH = _ROOT / "data" / "results" / "financebench_qa.jsonl"


def main():
    try:
        from datasets import load_dataset
    except ImportError:
        print("请先安装: pip install datasets")
        sys.exit(1)

    print("Loading FinanceBench...")
    ds = load_dataset("PatronusAI/financebench", split="train")
    print(f"Total examples: {len(ds)}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for row in ds:
            # 跳过没有答案的条目
            if not row["answer"] or not row["question"]:
                continue

            record = {
                "question":    row["question"],
                "answer":      row["answer"],
                "doc_id":      row["doc_name"],          # e.g. "3M_2018_10K"
                "company":     row["company"],
                "doc_period":  str(row["doc_period"]),
                "doc_type":    row["doc_type"],
                "question_type": row["question_type"],
                # 黄金 evidence（用于 Hit@K 的 ground truth）
                "evidence_text": row["evidence"][0]["evidence_text"] if row["evidence"] else "",
                "evidence_page": row["evidence"][0]["evidence_page_num"] if row["evidence"] else -1,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    print(f"Saved {written} QA pairs → {OUTPUT_PATH}")
    print("\nSample:")
    with open(OUTPUT_PATH) as f:
        sample = json.loads(f.readline())
    print(f"  question : {sample['question'][:80]}...")
    print(f"  answer   : {sample['answer']}")
    print(f"  doc_id   : {sample['doc_id']}")
    print(f"  company  : {sample['company']} ({sample['doc_period']})")


if __name__ == "__main__":
    main()
