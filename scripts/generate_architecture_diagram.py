"""
generate_architecture_diagram.py
=================================
生成项目架构图，输出至 docs/architecture.png。

Usage:
    python scripts/generate_architecture_diagram.py
"""

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

_ROOT = Path(__file__).parent.parent
_OUT  = _ROOT / "docs" / "architecture.png"

# ── 颜色方案 ──────────────────────────────────────────────
C_USER     = "#4A90D9"   # 蓝：用户 / API
C_AGENT    = "#7B68EE"   # 紫：LangGraph Agent 节点
C_RETRIEVAL= "#2ECC71"   # 绿：检索层
C_STORAGE  = "#E67E22"   # 橙：存储
C_INFRA    = "#95A5A6"   # 灰：基础设施
C_TEXT     = "#FFFFFF"
C_BG       = "#F8F9FA"
C_BORDER   = "#DEE2E6"

def box(ax, x, y, w, h, label, sublabel="", color=C_USER, fontsize=9, subsize=7.5):
    """画一个圆角矩形 + 标签。"""
    rect = FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle="round,pad=0.03",
        linewidth=1.2,
        edgecolor="#FFFFFF",
        facecolor=color,
        zorder=3,
    )
    ax.add_patch(rect)
    if sublabel:
        ax.text(x, y + 0.07, label, ha="center", va="center",
                fontsize=fontsize, fontweight="bold", color=C_TEXT, zorder=4)
        ax.text(x, y - 0.13, sublabel, ha="center", va="center",
                fontsize=subsize, color="#FFFFFFCC", zorder=4, style="italic")
    else:
        ax.text(x, y, label, ha="center", va="center",
                fontsize=fontsize, fontweight="bold", color=C_TEXT, zorder=4)

def arrow(ax, x1, y1, x2, y2, label="", color="#555555", lw=1.5):
    """画一条带箭头的连线。"""
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle="-|>",
            color=color,
            lw=lw,
            connectionstyle="arc3,rad=0.0",
        ),
        zorder=2,
    )
    if label:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mx + 0.05, my, label, fontsize=7, color="#555555",
                ha="left", va="center", zorder=5)

def hline(ax, x1, x2, y, color="#CCCCCC", lw=1.0, style="--"):
    ax.plot([x1, x2], [y, y], color=color, lw=lw, linestyle=style, zorder=1)


def main():
    fig, ax = plt.subplots(figsize=(14, 11))
    fig.patch.set_facecolor(C_BG)
    ax.set_facecolor(C_BG)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 11)
    ax.axis("off")

    # ── 标题 ─────────────────────────────────────────────
    ax.text(7, 10.6, "Structured RAG — System Architecture",
            ha="center", va="center", fontsize=14, fontweight="bold", color="#2C3E50")

    # ── 区域背景 ──────────────────────────────────────────
    def zone(x, y, w, h, label, color):
        r = FancyBboxPatch((x, y), w, h,
                           boxstyle="round,pad=0.1",
                           linewidth=1, edgecolor=color,
                           facecolor=color + "18", zorder=0)
        ax.add_patch(r)
        ax.text(x + 0.18, y + h - 0.18, label,
                fontsize=7.5, color=color, fontweight="bold", va="top", zorder=1)

    zone(0.3,  7.5, 13.4, 2.7,  "① Client & API Layer",    C_USER)
    zone(0.3,  4.2, 13.4, 3.0,  "② LangGraph Agent",       C_AGENT)
    zone(0.3,  0.5, 13.4, 3.4,  "③ Hybrid Retrieval Layer", C_RETRIEVAL)

    # ══════════════════════════════════════════════════════
    # 区域 1 — Client & API
    # ══════════════════════════════════════════════════════
    # User
    box(ax, 2.2, 9.8, 1.8, 0.55, "User Query", color=C_USER)
    # FastAPI
    box(ax, 5.5, 9.8, 2.4, 0.55, "FastAPI /query", "SSE Streaming", color=C_USER)
    # Demo UI
    box(ax, 9.5, 9.8, 2.2, 0.55, "Demo UI", "HTML/JS · Gradio", color=C_USER)
    # LangSmith
    box(ax, 12.2, 9.8, 1.8, 0.55, "LangSmith", "Tracing", color=C_INFRA)

    arrow(ax, 3.1, 9.8, 4.3, 9.8)
    arrow(ax, 6.7, 9.8, 8.4, 9.8, "stream")
    arrow(ax, 6.7, 9.8, 11.3, 9.8)

    # Guardrails
    box(ax, 3.5, 8.6, 2.4, 0.55, "Guardrails", "input relevance · output grounding", color=C_INFRA)
    # Semantic Cache
    box(ax, 7.5, 8.6, 2.6, 0.55, "Semantic Cache", "BGE-M3 cosine · threshold 0.92", color=C_INFRA)

    arrow(ax, 5.5, 9.52, 3.5, 8.88)
    arrow(ax, 5.5, 9.52, 7.5, 8.88)
    ax.text(6.2, 9.25, "cache\nhit →", fontsize=7, color="#888", ha="center")

    # ══════════════════════════════════════════════════════
    # 区域 2 — LangGraph Agent
    # ══════════════════════════════════════════════════════
    nodes = [
        (1.5,  5.7, "Planner",   "decide retrieve\n& rewrite"),
        (3.8,  5.7, "Tool",      "search_internal\nhybrid retrieval"),
        (6.1,  5.7, "Generator", "answer +\ninline citations"),
        (8.4,  5.7, "Reflector", "self-evaluate\nretry ≤ 2×"),
        (10.7, 5.7, "Final",     "format output\n+ attribution"),
    ]
    for nx, ny, nl, ns in nodes:
        box(ax, nx, ny, 1.9, 0.85, nl, ns, color=C_AGENT, fontsize=8.5, subsize=7)

    # 节点间箭头
    for i in range(len(nodes) - 1):
        x1 = nodes[i][0]   + 0.95
        x2 = nodes[i+1][0] - 0.95
        arrow(ax, x1, 5.7, x2, 5.7, color="#7B68EE", lw=1.8)

    # Reflector → Tool 重试回路
    ax.annotate("", xy=(3.8, 5.28), xytext=(8.4, 5.28),
                arrowprops=dict(arrowstyle="-|>", color="#E74C3C", lw=1.4,
                                connectionstyle="arc3,rad=0.0"), zorder=2)
    ax.text(6.1, 5.1, "retry (quality low)", fontsize=7, color="#E74C3C",
            ha="center", va="center")

    # API → Planner
    arrow(ax, 5.5, 9.52, 1.5, 6.13, color="#4A90D9", lw=1.5)

    # Final → API response
    arrow(ax, 10.7, 6.13, 5.5, 9.52, color="#4A90D9", lw=1.5)

    # ══════════════════════════════════════════════════════
    # 区域 3 — Hybrid Retrieval
    # ══════════════════════════════════════════════════════
    # Query Rewriter
    box(ax, 2.5, 3.2, 2.4, 0.65, "Query Rewriter",
        "ticker → company name\n(ADI → Analog Devices)", color="#27AE60", fontsize=8, subsize=7)

    # VectorStore
    box(ax, 6.0, 3.2, 2.4, 0.65, "VectorStore",
        "ChromaDB · BGE-M3\ndense vectors", color=C_RETRIEVAL, fontsize=8, subsize=7)

    # BM25
    box(ax, 9.0, 3.2, 2.0, 0.65, "BM25Store",
        "rank-bm25\nsparse index", color=C_RETRIEVAL, fontsize=8, subsize=7)

    # Reranker
    box(ax, 11.5, 3.2, 2.0, 0.65, "Reranker",
        "BGE-reranker-v2-m3\ncross-encoder", color=C_RETRIEVAL, fontsize=8, subsize=7)

    # Storage 层
    box(ax, 4.5, 1.45, 2.2, 0.6, "ChromaDB", "data/chroma/", color=C_STORAGE, fontsize=8, subsize=7)
    box(ax, 7.5, 1.45, 2.2, 0.6, "BM25 Index", "data/bm25_index.pkl", color=C_STORAGE, fontsize=8, subsize=7)
    box(ax, 10.8, 1.45, 2.6, 0.6, "FinQA · FinanceBench", "docs + eval sets", color=C_STORAGE, fontsize=8, subsize=7)

    # Tool → Rewriter / VectorStore / BM25
    arrow(ax, 3.8, 5.28, 2.5, 3.53, color="#2ECC71", lw=1.4)
    arrow(ax, 3.8, 5.28, 6.0, 3.53, color="#2ECC71", lw=1.4)
    arrow(ax, 3.8, 5.28, 9.0, 3.53, color="#2ECC71", lw=1.4)

    # VectorStore + BM25 → Reranker
    arrow(ax, 7.2, 3.2, 10.5, 3.2, color="#27AE60", lw=1.4)
    arrow(ax, 10.0, 3.2, 10.5, 3.2, color="#27AE60", lw=1.4)

    # Storage 连线
    arrow(ax, 6.0, 2.87, 4.5, 1.75, color="#E67E22", lw=1.2)
    arrow(ax, 9.0, 2.87, 7.5, 1.75, color="#E67E22", lw=1.2)
    arrow(ax, 11.5, 2.87, 10.8, 1.75, color="#E67E22", lw=1.2)

    # ── 图例 ────────────────────────────────────────────
    legend_items = [
        mpatches.Patch(color=C_USER,      label="API / Client"),
        mpatches.Patch(color=C_AGENT,     label="LangGraph Agent"),
        mpatches.Patch(color=C_RETRIEVAL, label="Retrieval"),
        mpatches.Patch(color=C_STORAGE,   label="Storage"),
        mpatches.Patch(color=C_INFRA,     label="Infrastructure"),
    ]
    ax.legend(handles=legend_items, loc="lower left",
              bbox_to_anchor=(0.02, 0.01), fontsize=8,
              framealpha=0.85, edgecolor=C_BORDER)

    # ── 输出 ────────────────────────────────────────────
    _OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(pad=0.5)
    plt.savefig(_OUT, dpi=180, bbox_inches="tight", facecolor=C_BG)
    plt.close()
    print(f"Saved → {_OUT}")


if __name__ == "__main__":
    main()
