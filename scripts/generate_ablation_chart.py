"""
generate_ablation_chart.py
==========================
读取 ablation_results.json，生成消融实验对比图。

Usage:
    python scripts/generate_ablation_chart.py
"""

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

_RESULTS_PATH = _ROOT / "data/results/ablation_results.json"
_OUT_PATH     = _ROOT / "data/results/ablation_chart.png"


def main():
    try:
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker
        import numpy as np
    except ImportError:
        print("请先安装: pip install matplotlib numpy")
        sys.exit(1)

    with open(_RESULTS_PATH, encoding="utf-8") as f:
        data = json.load(f)

    results = data["results"]
    labels  = list(results.keys())
    n       = data["n"]

    metrics_to_plot = [
        ("hit@5",        "Hit@5"),
        ("mrr@5",        "MRR@5"),
        ("chunk_prec@5", "ChunkPrec@5"),
    ]

    x     = np.arange(len(labels))
    width = 0.25
    colors = ["#4C72B0", "#DD8452", "#55A868"]

    fig, ax = plt.subplots(figsize=(11, 5))

    for i, (key, display) in enumerate(metrics_to_plot):
        vals = [results[lb][key] for lb in labels]
        offset = (i - 1) * width
        bars = ax.bar(x + offset, vals, width, label=display, color=colors[i], alpha=0.88)
        for bar in bars:
            h = bar.get_height()
            ax.annotate(
                f"{h:.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 3), textcoords="offset points",
                ha="center", va="bottom", fontsize=7.5,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=12, ha="right", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_ylabel("Score")
    ax.set_title(f"Ablation Study — Retrieval Component Contribution (n={n}, FinQA)", fontsize=11)
    ax.legend(loc="lower right")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(_OUT_PATH, dpi=150)
    plt.close()
    print(f"Chart saved → {_OUT_PATH}")


if __name__ == "__main__":
    main()
