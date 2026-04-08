"""
generate_eval_charts.py
=======================
生成实验结果对比图，输出至 docs/eval_results.png。

Usage:
    python scripts/generate_eval_charts.py
"""

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

_ROOT = Path(__file__).parent.parent
_OUT  = _ROOT / "docs" / "eval_results.png"

# ── 颜色 ───────────────────────────────────────────────────
C_FIXED   = "#4A90D9"
C_REC     = "#7B68EE"
C_SEM     = "#2ECC71"
C_BG      = "#F8F9FA"
C_GRID    = "#DEE2E6"

# ── 实验数据（混合评估 n=50, n_retrieval=300）─────────────
# 选取有代表性的实验组
experiments = [
    # label              strategy   alpha  Hit@1   Hit@3   Hit@5   MRR@5  Faith  CtxPrec AnsRel
    ("fixed-512\nα=0.7", "fixed",  0.7, 0.5433, 0.7333, 0.8133, 0.6473, 0.790, 0.285, 0.358),
    ("fixed-1024\nα=0.7","fixed",  0.7, 0.5300, 0.7333, 0.8000, 0.6364, 0.797, 0.386, 0.399),
    ("recur-1024\nα=0.7","recur",  0.7, 0.5367, 0.7367, 0.8067, 0.6423, 0.810, 0.455, 0.518),
    ("sem-1024\nα=0.7",  "sem",    0.7, 0.5133, 0.7300, 0.7900, 0.6237, 0.815, 0.337, 0.416),
    ("fixed-1024\nα=0.4","fixed",  0.4, 0.5333, 0.7200, 0.7933, 0.6340, 0.876, 0.435, 0.436),
    ("recur-1024\nα=0.4","recur",  0.4, 0.5300, 0.7200, 0.7967, 0.6339, 0.817, 0.560, 0.384),
    ("recur-1024\nα=0.3","recur",  0.3, 0.5167, 0.7400, 0.8233, 0.6362, 0.767, 0.427, 0.442),
]

labels   = [e[0] for e in experiments]
strategy = [e[1] for e in experiments]
hit1     = [e[3] for e in experiments]
hit3     = [e[4] for e in experiments]
hit5     = [e[5] for e in experiments]
mrr5     = [e[6] for e in experiments]
faith    = [e[7] for e in experiments]
ctxprec  = [e[8] for e in experiments]
ansrel   = [e[9] for e in experiments]

n = len(experiments)
x = np.arange(n)
bar_w = 0.22

# 颜色映射
colors = [C_FIXED if s == "fixed" else C_REC if s == "recur" else C_SEM
          for s in strategy]

def make_bar_chart(ax, title, datasets, ds_labels, ds_colors, ylim=(0, 1.0), yline=None):
    bar_width = 0.22
    offsets = np.linspace(-(len(datasets)-1)/2, (len(datasets)-1)/2, len(datasets)) * bar_width
    for vals, lbl, col, off in zip(datasets, ds_labels, ds_colors, offsets):
        bars = ax.bar(x + off, vals, bar_width, label=lbl, color=col, alpha=0.85,
                      edgecolor="white", linewidth=0.6)
    if yline is not None:
        ax.axhline(yline, color="#E74C3C", lw=1.2, linestyle="--", alpha=0.7)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylim(*ylim)
    ax.yaxis.grid(True, color=C_GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top","right"]].set_visible(False)
    ax.legend(fontsize=7.5, loc="lower right")


fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
fig.patch.set_facecolor(C_BG)
for ax in axes:
    ax.set_facecolor(C_BG)

fig.suptitle("Agentic RAG — Experiment Results (Mixed Eval: FinQA + FinanceBench, n=50)",
             fontsize=13, fontweight="bold", y=1.01, color="#2C3E50")

# ── 图1：Hit@K ─────────────────────────────────────────────
make_bar_chart(
    axes[0],
    "Retrieval — Hit@K & MRR@5",
    [hit1, hit3, hit5, mrr5],
    ["Hit@1", "Hit@3", "Hit@5", "MRR@5"],
    [C_FIXED, C_REC, C_SEM, "#E67E22"],
    ylim=(0.3, 1.0),
)
axes[0].set_ylabel("Score", fontsize=9)

# ── 图2：RAGAS ─────────────────────────────────────────────
make_bar_chart(
    axes[1],
    "Generation Quality — RAGAS",
    [faith, ctxprec, ansrel],
    ["Faithfulness", "Context Precision", "Answer Relevancy"],
    [C_FIXED, C_REC, C_SEM],
    ylim=(0, 1.0),
)
axes[1].set_ylabel("Score", fontsize=9)

# ── 图3：Faithfulness vs Hit@5 散点 ───────────────────────
ax3 = axes[2]
ax3.set_facecolor(C_BG)
scatter_colors = [C_FIXED if s == "fixed" else C_REC if s == "recur" else C_SEM
                  for s in strategy]
sc = ax3.scatter(hit5, faith, c=scatter_colors, s=120, zorder=3,
                 edgecolors="white", linewidths=0.8)
for i, lbl in enumerate(labels):
    ax3.annotate(lbl.replace("\n", " "),
                 (hit5[i], faith[i]),
                 textcoords="offset points", xytext=(6, 3),
                 fontsize=6.5, color="#555555")

ax3.set_xlabel("Hit@5", fontsize=9)
ax3.set_ylabel("Faithfulness", fontsize=9)
ax3.set_title("Hit@5 vs Faithfulness Trade-off", fontsize=11, fontweight="bold", pad=8)
ax3.yaxis.grid(True, color=C_GRID, linewidth=0.8, zorder=0)
ax3.xaxis.grid(True, color=C_GRID, linewidth=0.8, zorder=0)
ax3.set_axisbelow(True)
ax3.spines[["top","right"]].set_visible(False)

legend_items = [
    mpatches.Patch(color=C_FIXED, label="fixed"),
    mpatches.Patch(color=C_REC,   label="recursive"),
    mpatches.Patch(color=C_SEM,   label="semantic"),
]
ax3.legend(handles=legend_items, fontsize=7.5, loc="lower right")

# ── 最优标注 ───────────────────────────────────────────────
best_idx = 4  # fixed-1024 α=0.4
axes[1].axhline(faith[best_idx], color="#E74C3C", lw=1.0, linestyle=":", alpha=0.6)
axes[1].text(n - 0.5, faith[best_idx] + 0.01, f"best faith={faith[best_idx]:.3f}",
             fontsize=7, color="#E74C3C", ha="right")

plt.tight_layout(pad=2.0)
_OUT.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(_OUT, dpi=160, bbox_inches="tight", facecolor=C_BG)
plt.close()
print(f"Saved → {_OUT}")
