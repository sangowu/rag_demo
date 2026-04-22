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
    python scripts/eval_smoke.py [--n 10] [--top-k 5] [--threshold 0.6]
    python scripts/eval_smoke.py --tag bge-large  # 标记本次用的模型
"""

import argparse
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from src.bm25_store import BM25Store
from src.config import config
from src.reranker import Reranker
from src.retriever import Retriever
from src.vector_store import VectorStore

_EVAL_PATH = _ROOT / "data/finqa/eval.jsonl"
_LOG_PATH  = _ROOT / "data/eval_log.jsonl"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",         type=int,   default=10,   help="评测样本数")
    parser.add_argument("--top-k",     type=int,   default=5,    help="检索 top-k")
    parser.add_argument("--threshold", type=float, default=0.6,  help="最低 recall@k")
    parser.add_argument("--tag",       type=str,   default=None, help="模型标签，用于区分不同实验")
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


def main():
    args = parse_args()

    vs = VectorStore()

    indexed_ids     = vs.get_indexed_ids()
    indexed_doc_ids = {chunk_id.rsplit("_", 1)[0] for chunk_id in indexed_ids}

    with open(_EVAL_PATH, encoding="utf-8") as f:
        eval_records = [json.loads(line) for line in f if line.strip()]

    eligible = [r for r in eval_records if r.get("doc_id", "") in indexed_doc_ids]
    if not eligible:
        print("[ERROR] 没有可评测的样本，请先运行 ingest 脚本摄入文档。")
        sys.exit(1)

    random.seed(42)
    samples = random.sample(eligible, min(args.n, len(eligible)))
    print(f"已摄入文档: {len(indexed_doc_ids)} | 可评测问题: {len(eligible)} | 本次抽样: {len(samples)}")

    retriever = Retriever(vs, BM25Store(), Reranker())

    hits = 0
    total_latency_ms = 0.0
    for record in samples:
        query   = record.get("question", "")
        gold_id = record.get("doc_id", "")

        t0 = time.perf_counter()
        results = retriever.search(query, top_k=args.top_k)
        total_latency_ms += (time.perf_counter() - t0) * 1000

        retrieved_ids = {r.get("doc_id", "") for r in results}
        if gold_id in retrieved_ids:
            hits += 1
        else:
            print(f"[MISS] doc_id={gold_id!r}  query={query[:60]!r}")

    recall = hits / len(samples) if samples else 0.0
    avg_latency_ms = total_latency_ms / len(samples) if samples else 0.0
    passed = recall >= args.threshold

    print(f"\nRecall@{args.top_k} = {hits}/{len(samples)} = {recall:.2%}")
    print(f"Avg latency: {avg_latency_ms:.0f}ms/query")

    # 写入评测日志
    log_entry = {
        "timestamp":       datetime.now().isoformat(timespec="seconds"),
        "tag":             args.tag or "default",
        "recall_at_k":     round(recall, 4),
        "k":               args.top_k,
        "hits":            hits,
        "total":           len(samples),
        "avg_latency_ms":  round(avg_latency_ms, 1),
        "indexed_docs":    len(indexed_doc_ids),
        "passed":          passed,
        "config":          _read_current_config(),
    }
    _append_log(log_entry)
    print(f"结果已写入 {_LOG_PATH}")

    if not passed:
        print(f"[FAIL] recall {recall:.2%} < threshold {args.threshold:.2%}")
        sys.exit(1)

    print(f"[PASS] recall {recall:.2%} >= threshold {args.threshold:.2%}")


if __name__ == "__main__":
    main()
