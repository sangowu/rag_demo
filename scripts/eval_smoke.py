"""
eval_smoke.py
=============
CI smoke test：从 FinQA dev 集抽取少量样本，验证 retrieval 基本功能没有退化。

判定标准：
  - 对每个问题执行检索，检查 gold doc_id 是否出现在 top-k 结果中
  - recall@k = 命中数 / 总数
  - recall@k < THRESHOLD 时以非零退出码退出，触发 CI 失败

Usage:
    python scripts/eval_smoke.py [--n 10] [--top-k 5] [--threshold 0.6]
"""

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from src.bm25_store import BM25Store
from src.reranker import Reranker
from src.retriever import Retriever
from src.vector_store import VectorStore

_EVAL_PATH = _ROOT / "data/finqa/eval.jsonl"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",         type=int,   default=10,  help="评测样本数")
    parser.add_argument("--top-k",     type=int,   default=5,   help="检索 top-k")
    parser.add_argument("--threshold", type=float, default=0.6, help="最低 recall@k")
    return parser.parse_args()


def main():
    args = parse_args()

    with open(_EVAL_PATH, encoding="utf-8") as f:
        eval_records = [json.loads(line) for line in f if line.strip()]
    samples = eval_records[: args.n]

    # 初始化 retriever
    retriever = Retriever(VectorStore(), BM25Store(), Reranker())

    hits = 0
    for record in samples:
        query   = record.get("question", "")
        gold_id = record.get("doc_id", "")

        results = retriever.search(query, top_k=args.top_k)
        retrieved_ids = {r.get("doc_id", "") for r in results}

        if gold_id in retrieved_ids:
            hits += 1
        else:
            print(f"[MISS] doc_id={gold_id!r}  query={query[:60]!r}")

    recall = hits / len(samples) if samples else 0.0
    print(f"\nRecall@{args.top_k} = {hits}/{len(samples)} = {recall:.2%}")

    if recall < args.threshold:
        print(f"[FAIL] recall {recall:.2%} < threshold {args.threshold:.2%}")
        sys.exit(1)

    print(f"[PASS] recall {recall:.2%} >= threshold {args.threshold:.2%}")


if __name__ == "__main__":
    main()
