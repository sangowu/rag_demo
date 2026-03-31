"""
compare_evals.py
================
读取 data/eval_log.jsonl，打印多版本 RAGAS 指标对比表。

Usage:
    python scripts/compare_evals.py
    python scripts/compare_evals.py --log data/eval_log.jsonl
"""

import argparse
import json
from pathlib import Path

_ROOT = Path(__file__).parent.parent
_METRICS = ["faithfulness", "context_precision", "answer_relevancy"]


def load_log(log_path: Path) -> list[dict]:
    if not log_path.exists():
        print(f"Log not found: {log_path}")
        return []
    with open(log_path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def print_table(entries: list[dict]) -> None:
    col_w = {"tag": 18, "time": 16, "n": 4, "mode": 8, "alpha": 6, "ck": 4,
             "chunk": 6, "faith": 7, "prec": 7, "relev": 7}

    header = (
        f"{'tag':<{col_w['tag']}} {'time':<{col_w['time']}} {'n':>{col_w['n']}} "
        f"{'mode':<{col_w['mode']}} {'alpha':>{col_w['alpha']}} {'ck':>{col_w['ck']}} "
        f"{'chunk':>{col_w['chunk']}} {'faith':>{col_w['faith']}} "
        f"{'c_prec':>{col_w['prec']}} {'a_relev':>{col_w['relev']}}"
    )
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)

    for e in entries:
        faith = e.get("faithfulness", float("nan"))
        prec  = e.get("context_precision", float("nan"))
        relev = e.get("answer_relevancy", float("nan"))
        print(
            f"{e.get('tag',''):<{col_w['tag']}} "
            f"{e.get('time',''):<{col_w['time']}} "
            f"{e.get('n', ''):>{col_w['n']}} "
            f"{e.get('retriever_mode',''):>{col_w['mode']}} "
            f"{str(e.get('alpha',''))[:5]:>{col_w['alpha']}} "
            f"{str(e.get('candidate_k',''))[:4]:>{col_w['ck']}} "
            f"{str(e.get('chunk_size',''))[:6]:>{col_w['chunk']}} "
            f"{faith:>{col_w['faith']}.3f} "
            f"{prec:>{col_w['prec']}.3f} "
            f"{relev:>{col_w['relev']}.3f}"
        )
    print(sep)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=str, default="data/eval_log.jsonl")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    entries = load_log(_ROOT / args.log)
    if entries:
        print_table(entries)
