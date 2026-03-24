"""
eval_retrieval.py
=================
检索质量评估：对 FinQA eval 集计算 Hit@1/3/5、MRR@1/3/5。

结果保存至：
  - data/results/retrieval_metrics.json  — 数值统计
  - data/results/retrieval_metrics.png   — 对比图

Usage:
    python scripts/eval_retrieval.py [--n 200]
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from tqdm import tqdm

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from src.bm25_store import BM25Store
from src.reranker import Reranker
from src.retriever import Retriever
from src.vector_store import VectorStore

_EVAL_PATH   = _ROOT / "data/finqa/eval.jsonl"
_RESULTS_DIR = _ROOT / "data/results"
_KS          = [1, 3, 5]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=200, help="评测样本数（默认 200）")
    return parser.parse_args()


def compute_metrics(ranks: list[int | None], ks: list[int]) -> dict:
    """
    ranks: 每个样本 gold doc 的排名（1-based），未命中为 None
    返回 hit@k 和 mrr@k 字典
    """
    n = len(ranks)
    metrics = {}
    for k in ks:
        hits = sum(1 for r in ranks if r is not None and r <= k)
        rr   = sum(1.0 / r for r in ranks if r is not None and r <= k)
        metrics[f"hit@{k}"] = hits / n
        metrics[f"mrr@{k}"] = rr / n
    return metrics


def plot_metrics(metrics: dict, n: int, out_path: Path) -> None:
    ks        = _KS
    hit_vals  = [metrics[f"hit@{k}"] for k in ks]
    mrr_vals  = [metrics[f"mrr@{k}"] for k in ks]

    x     = range(len(ks))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 5))
    bars1 = ax.bar([i - width/2 for i in x], hit_vals, width, label="Hit@K",  color="#4C72B0")
    bars2 = ax.bar([i + width/2 for i in x], mrr_vals, width, label="MRR@K",  color="#DD8452")

    ax.set_xticks(list(x))
    ax.set_xticklabels([f"K={k}" for k in ks])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title(f"Retrieval Metrics (n={n})")
    ax.legend()
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))

    # 数值标注
    for bar in bars1 + bars2:
        h = bar.get_height()
        ax.annotate(f"{h:.2%}", xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Plot saved to {out_path}")


def main():
    args = parse_args()

    with open(_EVAL_PATH, encoding="utf-8") as f:
        eval_records = [json.loads(line) for line in f if line.strip()]
    samples = eval_records[: args.n]

    retriever = Retriever(VectorStore(), BM25Store(), Reranker())

    # 每个样本检索 top-max(K)，一次调用覆盖所有 K
    max_k  = max(_KS)
    ranks  = []   # gold doc 的排名，None 表示未命中
    misses = []

    for record in tqdm(samples, desc="Evaluating retrieval"):
        query   = record.get("question", "")
        gold_id = record.get("doc_id", "")

        results       = retriever.search(query, top_k=max_k)
        retrieved_ids = [r.get("doc_id", "") for r in results]

        if gold_id in retrieved_ids:
            ranks.append(retrieved_ids.index(gold_id) + 1)
        else:
            ranks.append(None)
            misses.append({"doc_id": gold_id, "query": query[:80]})

    metrics = compute_metrics(ranks, _KS)
    n       = len(samples)

    # 打印汇总
    print(f"\n{'='*42}")
    print(f"{'Metric':<12} {'Score':>8}")
    print(f"{'-'*42}")
    for k in _KS:
        print(f"Hit@{k:<8} {metrics[f'hit@{k}']:>8.2%}")
    print(f"{'-'*42}")
    for k in _KS:
        print(f"MRR@{k:<8} {metrics[f'mrr@{k}']:>8.4f}")
    print(f"{'='*42}")
    print(f"Missed {len(misses)}/{n} samples")

    if misses:
        print("\nFirst 5 misses:")
        for m in misses[:5]:
            print(f"  {m['doc_id']!r}  {m['query']!r}")

    # 保存 JSON
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = _RESULTS_DIR / "retrieval_metrics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"n": n, "metrics": metrics, "misses": misses}, f, indent=2, ensure_ascii=False)
    print(f"\nMetrics saved to {json_path}")

    # 保存 plot
    plot_metrics(metrics, n, _RESULTS_DIR / "retrieval_metrics.png")


if __name__ == "__main__":
    main()
