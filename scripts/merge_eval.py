"""
merge_eval.py
=============
合并 FinQA 和 FinanceBench 的标注文件，生成用于混合评估的 eval_mixed.jsonl。

采样策略：
  - FinanceBench：全量 150 条（官方 benchmark 共 150 条，全部使用）
  - FinQA       ：分层采样 150 条（按 question 长度分桶，保持分布均匀）
                  fixed seed=42，保证可复现

统一字段格式（与 evaluator.py 兼容）：
  question      : str   问题
  answer        : str   标准答案（FinQA 来自 exe_ans，FinanceBench 来自 answer）
  doc_id        : str   文档 ID
  evidence_page : int   FinanceBench 页码（FinQA 为 -1，evaluator 据此拼接 gold_id）
  dataset       : str   来源标识 "finqa" | "financebench"

关于评估正确性：
  1. Hit@K / MRR 仅对采样的 300 条 question 计算，结果代表所采样子集的性能
  2. 未采样的 question 不参与统计，不影响指标
  3. 未采样的 document 仍然存在于检索池（ChromaDB + BM25）中，作为 hard negative
     → 这使 Hit@K 比只测单数据集时更难，结果更真实可信，不会虚高

Usage:
    python scripts/merge_eval.py
    python scripts/merge_eval.py --finqa-n 200 --fb-n 100  # 自定义采样数量
    python scripts/merge_eval.py --seed 123                 # 自定义随机种子
"""

import argparse
import json
import random
from pathlib import Path

_ROOT = Path(__file__).parent.parent

_FINQA_PATH = _ROOT / "data" / "finqa" / "eval.jsonl"
_FB_PATH    = _ROOT / "data" / "results" / "financebench_qa.jsonl"
_OUT_PATH   = _ROOT / "data" / "eval_mixed.jsonl"


def _load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _stratified_sample(records: list[dict], n: int, seed: int) -> list[dict]:
    """
    对 FinQA 做分层采样：按 question 长度分成 3 个桶（短/中/长），
    各桶按比例取样，确保不同复杂度的问题均有覆盖。
    """
    rng = random.Random(seed)

    def bucket(r: dict) -> str:
        q_len = len(r.get("question", "").split())
        if q_len <= 12:
            return "short"
        elif q_len <= 20:
            return "medium"
        else:
            return "long"

    buckets: dict[str, list[dict]] = {"short": [], "medium": [], "long": []}
    for r in records:
        buckets[bucket(r)].append(r)

    total = len(records)
    result: list[dict] = []
    remainder = n

    bucket_names = list(buckets.keys())
    for i, name in enumerate(bucket_names):
        pool = buckets[name]
        if i == len(bucket_names) - 1:
            # 最后一桶取剩余数量，避免四舍五入误差
            k = remainder
        else:
            k = round(n * len(pool) / total)
        k = min(k, len(pool))
        result.extend(rng.sample(pool, k))
        remainder -= k

    rng.shuffle(result)
    return result


def _normalize_finqa(r: dict) -> dict:
    return {
        "question":      r.get("question", ""),
        "answer":        str(r.get("exe_ans", "") or ""),
        "doc_id":        r.get("doc_id", ""),
        "evidence_page": -1,   # FinQA doc_id 已含页码，无需额外拼接
        "dataset":       "finqa",
    }


def _normalize_financebench(r: dict) -> dict:
    return {
        "question":      r.get("question", ""),
        "answer":        str(r.get("answer", "") or ""),
        "doc_id":        r.get("doc_id", ""),
        "evidence_page": int(r.get("evidence_page", -1)),
        "dataset":       "financebench",
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--finqa-n", type=int, default=150,
                        help="从 FinQA 采样的 question 数量（默认 150）")
    parser.add_argument("--fb-n",    type=int, default=150,
                        help="从 FinanceBench 采样的 question 数量（默认 150，全量）")
    parser.add_argument("--seed",    type=int, default=42,
                        help="随机种子，保证可复现（默认 42）")
    parser.add_argument("--output",  type=str, default=str(_OUT_PATH),
                        help="输出路径（默认 data/eval_mixed.jsonl）")
    return parser.parse_args()


def main():
    args = parse_args()
    out_path = Path(args.output)

    # --- 加载 ---
    finqa_all = _load_jsonl(_FINQA_PATH)
    fb_all    = _load_jsonl(_FB_PATH)
    print(f"Loaded  FinQA:        {len(finqa_all):>5} records")
    print(f"Loaded  FinanceBench: {len(fb_all):>5} records")

    # --- 采样 ---
    finqa_n = min(args.finqa_n, len(finqa_all))
    fb_n    = min(args.fb_n,    len(fb_all))

    finqa_sampled = _stratified_sample(finqa_all, finqa_n, seed=args.seed)
    fb_sampled    = (
        fb_all if fb_n >= len(fb_all)
        else random.Random(args.seed).sample(fb_all, fb_n)
    )

    print(f"Sampled FinQA:        {len(finqa_sampled):>5}  (seed={args.seed}, stratified by question length)")
    print(f"Sampled FinanceBench: {len(fb_sampled):>5}  ({'full set' if fb_n >= len(fb_all) else f'seed={args.seed}'})")

    # --- 归一化 + 合并 ---
    mixed = (
        [_normalize_finqa(r) for r in finqa_sampled]
        + [_normalize_financebench(r) for r in fb_sampled]
    )
    random.Random(args.seed).shuffle(mixed)   # 打乱顺序，避免评估时先全 FinQA 后全 FB

    # --- 写出 ---
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in mixed:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(mixed)} records → {out_path}")

    # --- 统计摘要 ---
    fb_count    = sum(1 for r in mixed if r["dataset"] == "financebench")
    finqa_count = sum(1 for r in mixed if r["dataset"] == "finqa")
    print(f"\n  finqa:        {finqa_count} ({finqa_count/len(mixed)*100:.1f}%)")
    print(f"  financebench: {fb_count} ({fb_count/len(mixed)*100:.1f}%)")
    print(f"\n  Retrieval pool size is NOT affected by sampling.")
    print(f"  All ingested docs remain in ChromaDB+BM25 as hard negatives.")
    print(f"\nRun evaluation:")
    print(f"  python src/evaluator.py --tag mixed_eval --eval-path {out_path} --n 50 --n-retrieval {len(mixed)}")


if __name__ == "__main__":
    main()
