"""
eval_e2e.py
===========
端到端 Agent Pipeline 评测。

对 FinQA eval.jsonl 中的样本，完整跑 LangGraph agent，评测：
  - Exact Match      : 数值答案精确匹配（去单位、去逗号）
  - LLM-as-Judge     : Gemini 对答案语义质量评分（1-5）
  - Refuse Rate      : grounding check 触发拒答的比例
  - Avg Latency      : 端到端延迟（含 LLM 生成）

结果保存至 data/results/e2e_results.json

Usage:
    python scripts/eval_e2e.py
    python scripts/eval_e2e.py --n 50 --seed 42
    python scripts/eval_e2e.py --n 50 --no-judge   # 跳过 LLM-as-Judge，仅计算 EM + Refuse
"""

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

from tqdm import tqdm

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from src.agent.graph import graph
from src.llm_judge import LLMJudge
from src.tracing import record_agent_trace

_EVAL_PATH   = _ROOT / "data/finqa/eval.jsonl"
_RESULTS_DIR = _ROOT / "data/results"


# ── 数值归一化（FinQA gold answer 多为数值字符串）─────────────────────────────
def _normalize(text: str) -> str:
    """去除货币符号、逗号、百分号、多余空格，统一小写。"""
    text = text.strip().lower()
    text = re.sub(r"[$,%]", "", text)
    text = re.sub(r",", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _exact_match(pred: str, gold: str) -> bool:
    return _normalize(pred) == _normalize(gold)


def _contains_match(pred: str, gold: str) -> bool:
    """gold 值出现在 pred 中即算命中（应对答案含解释的情况）。"""
    return _normalize(gold) in _normalize(pred)


# ── 初始 state ────────────────────────────────────────────────────────────────
def _init_state(query: str) -> dict:
    return {
        "query": query,
        "retry_count": 0,
        "retrieved_chunks": [],
        "sources": [],
        "messages": [],
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "trace": [],
    }


# ── 主评测循环 ────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",        type=int, default=50,   help="评测样本数（默认 50）")
    parser.add_argument("--seed",     type=int, default=42)
    parser.add_argument("--no-judge", action="store_true",    help="跳过 LLM-as-Judge（更快）")
    args = parser.parse_args()

    # ── 加载 & 采样 ──────────────────────────────────────────────────────────
    records = [json.loads(l) for l in _EVAL_PATH.open(encoding="utf-8") if l.strip()]
    # gold answer 字段为 exe_ans
    records = [r for r in records if r.get("exe_ans") is not None]
    random.seed(args.seed)
    samples = random.sample(records, min(args.n, len(records)))
    print(f"Loaded {len(samples)} samples (seed={args.seed})")
    if not samples:
        print("ERROR: no samples found. Check eval.jsonl path and exe_ans field.")
        sys.exit(1)

    judge = LLMJudge() if not args.no_judge else None

    # ── 结果收集 ─────────────────────────────────────────────────────────────
    em_hits = 0
    contains_hits = 0
    refused_count = 0
    judge_scores: list[int] = []
    total_latency = 0.0
    rows: list[dict] = []

    for rec in tqdm(samples, desc="eval_e2e", ncols=88):
        query      = rec["question"]
        gold       = str(rec["exe_ans"])

        t0     = time.perf_counter()
        result = graph.invoke(_init_state(query))
        record_agent_trace(query, result, run_name=f"eval:{query[:40]}")
        latency_ms = (time.perf_counter() - t0) * 1000
        total_latency += latency_ms

        final_answer = result.get("final_answer", "")
        refused      = result.get("refuse", False)
        grounding    = result.get("grounding_score", None)
        chunks       = result.get("retrieved_chunks", [])

        if refused:
            refused_count += 1
            em = False
            contains = False
            judge_result = None
        else:
            em       = _exact_match(final_answer, gold)
            contains = _contains_match(final_answer, gold)
            if em:
                em_hits += 1
            if contains:
                contains_hits += 1

            if judge is not None:
                contexts = [c.get("text", "") for c in chunks[:5]]
                try:
                    judge_result = judge.evaluate(query, final_answer, contexts)
                    judge_scores.append(judge_result["score"])
                except Exception as e:
                    print(f"\n[WARN] Judge failed: {e}")
                    judge_result = None
            else:
                judge_result = None

        rows.append({
            "question":       query,
            "gold":           gold,
            "answer":         final_answer,
            "refused":        refused,
            "grounding_score": round(grounding, 4) if grounding is not None else None,
            "exact_match":    em,
            "contains_match": contains,
            "judge":          judge_result,
            "latency_ms":     round(latency_ms),
        })

    # ── 汇总指标 ─────────────────────────────────────────────────────────────
    n = len(samples)
    n_answered = n - refused_count
    summary = {
        "n":               n,
        "seed":            args.seed,
        "refuse_rate":     round(refused_count / n, 4),
        "exact_match":     round(em_hits / n, 4),
        "contains_match":  round(contains_hits / n, 4),
        "avg_judge_score": round(sum(judge_scores) / len(judge_scores), 3) if judge_scores else None,
        "judge_pass_rate": round(sum(1 for s in judge_scores if s >= 4) / len(judge_scores), 4) if judge_scores else None,
        "avg_latency_ms":  round(total_latency / n, 1),
        "n_answered":      n_answered,
        "n_refused":       refused_count,
    }

    # ── 打印结果 ─────────────────────────────────────────────────────────────
    print(f"\n{'='*50}")
    print(f"  End-to-End Evaluation — FinQA (n={n})")
    print(f"{'='*50}")
    print(f"  Refuse Rate      : {summary['refuse_rate']:.1%}  ({refused_count}/{n})")
    print(f"  Exact Match      : {summary['exact_match']:.1%}  (over all {n})")
    print(f"  Contains Match   : {summary['contains_match']:.1%}  (over all {n})")
    if summary["avg_judge_score"] is not None:
        print(f"  Judge Score      : {summary['avg_judge_score']:.2f} / 5.0  (pass≥4: {summary['judge_pass_rate']:.1%})")
    print(f"  Avg Latency      : {summary['avg_latency_ms']:.0f} ms/query")
    print(f"{'='*50}\n")

    # ── 保存 ─────────────────────────────────────────────────────────────────
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = _RESULTS_DIR / "e2e_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "rows": rows}, f, indent=2, ensure_ascii=False)
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
