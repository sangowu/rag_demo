"""
compare_evals.py
================
读取 data/eval_log.jsonl，打印多版本指标对比表。

Usage:
    python scripts/compare_evals.py
    python scripts/compare_evals.py --log data/eval_log.jsonl
"""

import argparse
import json
from pathlib import Path

_ROOT = Path(__file__).parent.parent


def load_log(log_path: Path) -> list[dict]:
    if not log_path.exists():
        print(f"Log not found: {log_path}")
        return []
    with open(log_path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _f(val, fmt=".3f"):
    if val is None or val == "":
        return "-"
    try:
        return format(float(val), fmt)
    except (TypeError, ValueError):
        return str(val)


def print_table(entries: list[dict]) -> None:
    # --- RAGAS 指标 ---
    print("\n┌─ RAGAS Metrics " + "─" * 52 + "┐")
    hdr = (
        f"  {'tag':<18} {'n':>4}  {'faith':>6}  {'c_prec':>6}  {'a_relev':>7}"
        f"  {'alpha':>5}  {'ck':>3}  {'strategy':<10}  {'size':>4}  {'ovlp':>4}"
    )
    print(hdr)
    print("  " + "─" * (len(hdr) - 2))
    for e in entries:
        print(
            f"  {e.get('tag',''):<18} {e.get('n','-'):>4}"
            f"  {_f(e.get('faithfulness')):>6}"
            f"  {_f(e.get('context_precision')):>6}"
            f"  {_f(e.get('answer_relevancy')):>7}"
            f"  {_f(e.get('alpha'),''):>5}"
            f"  {str(e.get('candidate_k',''))[:3]:>3}"
            f"  {str(e.get('chunk_strategy','')):<10}"
            f"  {str(e.get('chunk_size',''))[:4]:>4}"
            f"  {str(e.get('chunk_overlap',''))[:4]:>4}"
        )

    # --- 检索指标 ---
    print("\n┌─ Retrieval Metrics (Hit@K / MRR@K) " + "─" * 31 + "┐")
    hdr2 = f"  {'tag':<18}  {'hit@1':>5}  {'hit@3':>5}  {'hit@5':>5}  {'mrr@1':>5}  {'mrr@3':>5}  {'mrr@5':>5}"
    print(hdr2)
    print("  " + "─" * (len(hdr2) - 2))
    for e in entries:
        print(
            f"  {e.get('tag',''):<18}"
            f"  {_f(e.get('hit@1')):>5}"
            f"  {_f(e.get('hit@3')):>5}"
            f"  {_f(e.get('hit@5')):>5}"
            f"  {_f(e.get('mrr@1')):>5}"
            f"  {_f(e.get('mrr@3')):>5}"
            f"  {_f(e.get('mrr@5')):>5}"
        )

    # --- 耗时 & Token ---
    print("\n┌─ Timing & Tokens " + "─" * 50 + "┐")
    hdr3 = (
        f"  {'tag':<18}  {'t_retr':>6}  {'t_gen':>5}  {'t_ragas':>7}"
        f"  {'t_tot':>5}  {'p_tok':>6}  {'c_tok':>6}  {'avg_p':>5}  {'avg_c':>5}"
    )
    print(hdr3)
    print("  " + "─" * (len(hdr3) - 2))
    for e in entries:
        print(
            f"  {e.get('tag',''):<18}"
            f"  {_f(e.get('t_retrieval_s'),'.1f'):>6}s"
            f"  {_f(e.get('t_generation_s'),'.1f'):>5}s"
            f"  {_f(e.get('t_ragas_s'),'.1f'):>7}s"
            f"  {_f(e.get('t_total_s'),'.1f'):>5}s"
            f"  {_f(e.get('prompt_tokens'),'.0f'):>6}"
            f"  {_f(e.get('completion_tokens'),'.0f'):>6}"
            f"  {_f(e.get('avg_prompt_tokens'),'.0f'):>5}"
            f"  {_f(e.get('avg_completion_tokens'),'.0f'):>5}"
        )
    print()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=str, default="data/eval_log.jsonl")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    entries = load_log(_ROOT / args.log)
    if entries:
        print_table(entries)
