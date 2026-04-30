"""
bad_case_analysis.py
====================
分析检索失败案例，找出系统薄弱点。

输出：
  data/results/bad_cases.jsonl  — 每行一个失败案例
  终端打印失败模式统计

Usage:
    python scripts/bad_case_analysis.py
    python scripts/bad_case_analysis.py --datasets finqa,legalbench --n 100 --top-k 5
    python scripts/bad_case_analysis.py --datasets finqa --n 200 --strategy baseline
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path
from collections import Counter

from tqdm import tqdm

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from src.bm25_store import BM25Store
from src.reranker import Reranker
from src.retriever import Retriever
from src.vector_store import VectorStore

_EVAL_PATHS = {
    "finqa":      _ROOT / "data/finqa/eval.jsonl",
    "legalbench": _ROOT / "data/legalbench/eval.jsonl",
}
_RESULTS_DIR = _ROOT / "data/results"


def _overlaps_char(chunk: dict, snippet: dict) -> bool:
    cs, ce = chunk.get("start_char", 0), chunk.get("end_char", 0)
    return cs < snippet["char_end"] and ce > snippet["char_start"]


def _extract_table_cells(gold_text: str) -> list[str]:
    import re
    return [v.strip() for v in re.findall(r' is ([^;]+)', gold_text) if v.strip()]


def _chunk_hits_gold(chunk_text: str, gold_inds: dict) -> bool:
    chunk_lower = chunk_text.lower()
    for key, gold_text in gold_inds.items():
        gold_clean = gold_text.strip().lower()
        if len(gold_clean) < 15:
            continue
        if key.startswith("table_"):
            cells = _extract_table_cells(gold_clean)
            if not cells:
                continue
            threshold = min(2, len(cells))
            if sum(1 for c in cells if c in chunk_lower) >= threshold:
                return True
        else:
            if gold_clean[:60] in chunk_lower:
                return True
    return False


def _classify_failure(record: dict, results: list[dict], top_k: int) -> str:
    """
    将失败案例分类：
      no_doc     — top-k 中没有任何来自 gold doc 的 chunk
      wrong_chunk — gold doc 被检索到但 chunk 不包含证据
      rank_low   — gold chunk 存在但排名 > top_k
    """
    gold_id = record.get("doc_id", "")
    gold_snippets = record.get("snippets")
    gold_inds = record.get("gold_inds", {})
    if isinstance(gold_inds, list):
        gold_inds = {str(i): v for i, v in enumerate(gold_inds)}

    retrieved_doc_ids = [r.get("doc_id", "") for r in results]
    has_gold_doc_in_top = gold_id in retrieved_doc_ids

    if not has_gold_doc_in_top:
        return "no_doc_retrieved"

    # gold doc 进来了，但 chunk 不对
    if gold_snippets:
        any_overlap = any(
            r.get("doc_id") == gold_id and any(_overlaps_char(r, s) for s in gold_snippets)
            for r in results
        )
    else:
        any_overlap = any(
            _chunk_hits_gold(r.get("text", ""), gold_inds)
            for r in results
            if r.get("doc_id") == gold_id
        )

    return "wrong_chunk" if not any_overlap else "rank_too_low"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", type=str, default="finqa,legalbench")
    parser.add_argument("--n",        type=int, default=100)
    parser.add_argument("--seed",     type=int, default=42)
    parser.add_argument("--top-k",    type=int, default=5)
    parser.add_argument("--strategy", type=str, default="baseline",
                        choices=["baseline", "contextual"])
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]

    table  = f"chunks_{args.strategy}"
    bm25_path = _ROOT / f"data/bm25_{args.strategy}.pkl"

    print(f"[init] 连接 VectorStore({table})...", flush=True)
    vs = VectorStore(table_name=table)

    indexed_ids = vs.get_indexed_ids()
    indexed_doc_ids = {row.rsplit("_", 1)[0] for row in indexed_ids}

    # 采样
    samples_by_ds: dict[str, list[dict]] = {}
    for ds in datasets:
        path = _EVAL_PATHS.get(ds)
        if not path or not path.exists():
            print(f"[WARN] {ds} eval.jsonl not found, skipping.")
            continue
        records  = [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]
        eligible = [r for r in records if r.get("doc_id", "") in indexed_doc_ids]
        sampled  = random.sample(eligible, min(args.n, len(eligible)))
        samples_by_ds[ds] = sampled
        print(f"[{ds}] eligible={len(eligible)}, sampled={len(sampled)}")

    print(f"\n[init] 加载 BM25 ({bm25_path.stat().st_size/1024/1024:.0f} MB)...", flush=True)
    bm25 = BM25Store(index_path=bm25_path)
    bm25.load()
    print("[init] BM25 加载完成。", flush=True)

    print("[init] 加载 Reranker...", flush=True)
    reranker = Reranker()
    print("[init] Reranker 加载完成。", flush=True)

    retriever = Retriever(vs, bm25, reranker)

    # ── 分数据集收集 bad cases ─────────────────────────────────────────
    all_bad_cases: list[dict] = []
    ds_stats: dict[str, dict] = {}

    for ds, samples in samples_by_ds.items():
        fail_types: list[str] = []
        hit_count = 0

        for record in tqdm(samples, desc=f"[{ds}]", ncols=88):
            query       = record.get("question", "")
            gold_id     = record.get("doc_id", "")
            gold_snippets = record.get("snippets")
            gold_inds   = record.get("gold_inds", {})
            if isinstance(gold_inds, list):
                gold_inds = {str(i): v for i, v in enumerate(gold_inds)}

            results = [r.to_dict() for r in retriever.search(query, top_k=args.top_k, rewrite=False)]

            # 判断 hit
            if gold_snippets:
                hit = any(
                    r.get("doc_id") == gold_id and any(_overlaps_char(r, s) for s in gold_snippets)
                    for r in results
                )
            else:
                hit = gold_id in [r.get("doc_id") for r in results] and any(
                    _chunk_hits_gold(r.get("text", ""), gold_inds)
                    for r in results if r.get("doc_id") == gold_id
                )

            if hit:
                hit_count += 1
                continue

            # 失败案例
            fail_type = _classify_failure(record, results, args.top_k)
            fail_types.append(fail_type)

            bad_case = {
                "dataset":    ds,
                "fail_type":  fail_type,
                "query":      query,
                "gold_doc_id": gold_id,
                "retrieved_doc_ids": [r.get("doc_id") for r in results],
                "retrieved_texts":   [r.get("text", "")[:200] for r in results],
            }
            if gold_snippets:
                bad_case["gold_snippets"] = gold_snippets[:3]
            else:
                bad_case["gold_inds"] = dict(list(gold_inds.items())[:3])
            all_bad_cases.append(bad_case)

        n = len(samples)
        ds_stats[ds] = {
            "total":   n,
            "hit":     hit_count,
            "miss":    n - hit_count,
            f"hit@{args.top_k}": round(hit_count / n, 3),
            "fail_types": dict(Counter(fail_types)),
        }

    # ── 打印统计 ──────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"Bad Case 分析  (strategy={args.strategy}, top_k={args.top_k})")
    print(f"{'='*55}")
    for ds, stat in ds_stats.items():
        print(f"\n[{ds}]  Hit@{args.top_k}={stat[f'hit@{args.top_k}']:.3f}  "
              f"miss={stat['miss']}/{stat['total']}")
        for ft, cnt in sorted(stat["fail_types"].items(), key=lambda x: -x[1]):
            pct = cnt / stat["miss"] * 100 if stat["miss"] else 0
            print(f"  {ft:<22} {cnt:>4} ({pct:.0f}%)")

    print(f"\n总 bad cases: {len(all_bad_cases)}")

    # ── 保存 ─────────────────────────────────────────────────────────
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = _RESULTS_DIR / "bad_cases.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for case in all_bad_cases:
            f.write(json.dumps(case, ensure_ascii=False) + "\n")
    print(f"已保存 → {out}")

    # ── 抽样展示 ─────────────────────────────────────────────────────
    print(f"\n{'─'*55}")
    print("Bad Case 抽样（每类最多 2 条）")
    print(f"{'─'*55}")
    shown: Counter = Counter()
    for case in all_bad_cases:
        ft = case["fail_type"]
        if shown[ft] >= 2:
            continue
        shown[ft] += 1
        print(f"\n[{case['dataset']}] {ft}")
        print(f"  Query : {case['query'][:120]}")
        print(f"  Gold  : {case['gold_doc_id']}")
        print(f"  Got   : {case['retrieved_doc_ids']}")
        if "gold_snippets" in case:
            s = case["gold_snippets"][0]
            print(f"  Snippet char [{s['char_start']},{s['char_end']}]")
        elif "gold_inds" in case:
            first_val = next(iter(case["gold_inds"].values()), "")
            print(f"  Gold text: {first_val[:80]}")


if __name__ == "__main__":
    main()
