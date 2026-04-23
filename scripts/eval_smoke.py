"""
eval_smoke.py
=============
CI smoke test：从 FinQA dev 集抽取少量样本，验证 retrieval 基本功能没有退化。

判定标准：
  - 对每个问题执行检索，检查 gold doc_id 是否出现在 top-k 结果中
  - recall@k = 命中数 / 总数
  - recall@k < THRESHOLD 时以非零退出码退出，触发 CI 失败

每次评测结果追加写入 data/eval_log.jsonl，用于模型横向对比。

Usage:
    python scripts/eval_smoke.py [--n 100] [--top-k 5] [--threshold 0.6]
    python scripts/eval_smoke.py --tag parent_child  # 标记本次实验策略
"""

import argparse
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from src.bm25_store import BM25Store
from src.config import config
from src.reranker import Reranker
from src.retriever import Retriever
from src.vector_store import VectorStore

_EVAL_PATHS: dict[str, Path] = {
    "finqa":        _ROOT / "data" / "finqa"        / "eval.jsonl",
    "financebench": _ROOT / "data" / "financebench" / "eval.jsonl",
}
_LOG_PATH = _ROOT / "data/eval_log.jsonl"


STRATEGIES = ("baseline", "parent_child", "contextual", "summary")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",         type=int,   default=100,
                        help="每个数据集最大评测问题数（默认 100，seed=42 保证可复现）")
    parser.add_argument("--top-k",     type=int,   default=5,    help="检索 top-k")
    parser.add_argument("--threshold", type=float, default=0.6,  help="最低 recall@k")
    parser.add_argument("--tag",       type=str,   default=None, help="实验标签，写入 eval_log")
    parser.add_argument("--strategy",  type=str,   default=None, choices=STRATEGIES,
                        help="使用指定策略的独立索引表（默认使用 chunks 主表）")
    parser.add_argument("--datasets",  type=str,   default="finqa",
                        help="逗号分隔的数据集名称：finqa,financebench（默认：finqa）")
    return parser.parse_args()


def _read_current_config() -> dict:
    vs_cfg = config.get("vector_store", {})
    r_cfg  = config.get("retriever", {})
    return {
        "embedding_model":      vs_cfg.get("embedding_model", ""),
        "embedding_model_path": vs_cfg.get("embedding_model_path", ""),
        "embedding_server_url": vs_cfg.get("embedding_server_url", ""),
        "embed_max_token":      vs_cfg.get("embed_max_token", ""),
        "store_sparse":         vs_cfg.get("store_sparse", False),
        "retriever_mode":       r_cfg.get("mode", ""),
        "top_k":                r_cfg.get("top_k", ""),
        "chunking_strategy":    config.get("chunking", {}).get("strategy", ""),
        "chunk_size":           config.get("chunking", {}).get("chunk_size", ""),
    }


def _append_log(record: dict) -> None:
    _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _load_eval_records(datasets: list[str], indexed_doc_ids: set, n_per_dataset: int) -> list[dict]:
    """Load up to n_per_dataset eligible questions per dataset."""
    all_samples: list[dict] = []
    random.seed(42)
    for ds in datasets:
        eval_path = _EVAL_PATHS.get(ds)
        if eval_path is None or not eval_path.exists():
            print(f"[WARN] eval.jsonl not found for dataset '{ds}': {eval_path}")
            continue
        records = [json.loads(l) for l in eval_path.open(encoding="utf-8") if l.strip()]
        eligible = [r for r in records if r.get("doc_id", "") in indexed_doc_ids]
        sampled  = random.sample(eligible, min(n_per_dataset, len(eligible)))
        print(f"[{ds}] 可评测: {len(eligible)} | 本次抽样: {len(sampled)}")
        all_samples.extend(sampled)
    return all_samples


def main():
    args = parse_args()
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]

    # 按策略选择独立的 DB 表和 BM25 文件
    if args.strategy:
        table     = f"chunks_{args.strategy}"
        bm25_path = _ROOT / "data" / f"bm25_{args.strategy}.pkl"
        vs        = VectorStore(table_name=table)
        bm25      = BM25Store(index_path=bm25_path)
        print(f"Using strategy table: {table}")
    else:
        vs   = VectorStore()
        bm25 = BM25Store()

    indexed_ids     = vs.get_indexed_ids()
    indexed_doc_ids = {chunk_id.rsplit("_", 1)[0] for chunk_id in indexed_ids}

    samples = _load_eval_records(datasets, indexed_doc_ids, args.n)
    if not samples:
        print("[ERROR] 没有可评测的样本，请先运行 build_index.py 摄入文档。")
        sys.exit(1)

    print(f"已摄入文档: {len(indexed_doc_ids)} | 总抽样问题: {len(samples)}")

    retriever = Retriever(vs, bm25, Reranker())

    KS = [1, 3, 5]
    # 确保 top_k 至少取到最大的 K
    fetch_k = max(args.top_k, max(KS))

    hits_at   = {k: 0 for k in KS}
    rr_at     = {k: 0.0 for k in KS}
    total_latency_ms = 0.0

    pbar = tqdm(samples, desc="Evaluating", unit="q")
    for record in pbar:
        query   = record.get("question", "")
        gold_id = record.get("doc_id", "")

        t0 = time.perf_counter()
        results = retriever.search(query, top_k=fetch_k)
        total_latency_ms += (time.perf_counter() - t0) * 1000

        retrieved_ids = [r.get("doc_id", "") for r in results]

        # 找到 gold_id 在结果中的位置（1-indexed），没找到则 rank=None
        try:
            rank = retrieved_ids.index(gold_id) + 1
        except ValueError:
            rank = None

        if rank is None:
            tqdm.write(f"[MISS] doc_id={gold_id!r}  query={query[:60]!r}")

        for k in KS:
            if rank is not None and rank <= k:
                hits_at[k] += 1
                rr_at[k]   += 1.0 / rank

    n = len(samples)
    recall = {k: hits_at[k] / n for k in KS}
    mrr    = {k: rr_at[k]   / n for k in KS}
    avg_latency_ms = total_latency_ms / n if n else 0.0
    passed = recall[args.top_k] >= args.threshold

    print(f"\n{'Metric':<12}" + "".join(f"  @{k:<6}" for k in KS))
    print("-" * 30)
    print(f"{'Recall':<12}" + "".join(f"  {recall[k]:.4f}" for k in KS))
    print(f"{'MRR':<12}" + "".join(f"  {mrr[k]:.4f}"    for k in KS))
    print(f"\nAvg latency: {avg_latency_ms:.0f} ms/query  |  n={n}")

    log_entry = {
        "timestamp":      datetime.now().isoformat(timespec="seconds"),
        "tag":            args.tag or "default",
        "datasets":       datasets,
        "total":          n,
        "indexed_docs":   len(indexed_doc_ids),
        "avg_latency_ms": round(avg_latency_ms, 1),
        "passed":         passed,
        **{f"recall@{k}": round(recall[k], 4) for k in KS},
        **{f"mrr@{k}":    round(mrr[k],    4) for k in KS},
        "config":         _read_current_config(),
    }
    _append_log(log_entry)
    print(f"结果已写入 {_LOG_PATH}")

    if not passed:
        print(f"[FAIL] recall@{args.top_k} {recall[args.top_k]:.2%} < threshold {args.threshold:.2%}")
        sys.exit(1)

    print(f"[PASS] recall@{args.top_k} {recall[args.top_k]:.2%} >= threshold {args.threshold:.2%}")


if __name__ == "__main__":
    main()
