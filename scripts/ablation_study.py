"""
ablation_study.py
=================
消融实验：逐组件量化每一步对检索质量的贡献。

配置（逐步叠加）：
  1. Dense only        — 仅 BGE-M3 dense 向量检索
  2. + BM25 hybrid     — Dense + BM25 RRF 融合
  3. + Reranker        — 加入 BGE cross-encoder reranker
  4. + Query Rewriter  — LLM 展开 ticker / 公司名
  5. + Header Inject   — 切换至 contextual chunking 索引

评测指标：
  - Hit@1/3/5, MRR@5          (doc-level)
  - ChunkPrec@5               (chunk-level，利用 FinQA gold_inds 证据文本)
  - avg_latency_ms            (端到端检索延迟)

数据集：FinQA eval.jsonl，seed=42 随机抽取 n 条（默认 200）

结果保存至：data/results/ablation_results.json

Usage:
    python scripts/ablation_study.py
    python scripts/ablation_study.py --n 50 --configs 1,2,3
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

from tqdm import tqdm

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from src.bm25_store import BM25Store
from src.query_rewriter import QueryRewriter
from src.reranker import Reranker
from src.retriever import Retriever
from src.vector_store import VectorStore

_EVAL_PATH   = _ROOT / "data/finqa/eval.jsonl"
_RESULTS_DIR = _ROOT / "data/results"
_KS          = [1, 3, 5]


# ──────────────────────────── stub components ────────────────────────────────

class _NullBM25:
    """BM25 空实现，返回空列表 → RRF 仅依赖 dense 结果。"""
    def search(self, query: str, top_k: int = 50) -> list:
        return []


class _PassThroughReranker:
    """Reranker 空实现，直接返回 RRF 候选的 top-k，不做二次打分。"""
    def rerank(self, query: str, candidates: list, top_k: int = 5) -> list:
        return candidates[:top_k]

    def offload(self) -> None:
        pass


# ──────────────────────────── gold_inds matching ─────────────────────────────

def _chunk_hits_gold(chunk_text: str, gold_inds: dict) -> bool:
    """
    判断检索到的 chunk 是否包含 gold_inds 中的证据文本。
    匹配策略：取每条证据文本的前 60 个字符做子串匹配，
    容忍原文中轻微的格式差异（换行/空格）。
    """
    if not gold_inds:
        return False
    chunk_lower = chunk_text.lower()
    for gold_text in gold_inds.values():
        gold_clean = gold_text.strip().lower()
        if len(gold_clean) < 15:
            continue
        prefix = gold_clean[:60]
        if prefix in chunk_lower:
            return True
    return False


# ──────────────────────────── metrics ────────────────────────────────────────

def _compute_metrics(ranks: list[int | None], chunk_hits: list[bool]) -> dict:
    n = len(ranks)
    metrics: dict = {}
    for k in _KS:
        hits = sum(1 for r in ranks if r is not None and r <= k)
        rr   = sum(1.0 / r for r in ranks if r is not None and r <= k)
        metrics[f"hit@{k}"]  = round(hits / n, 4)
        metrics[f"mrr@{k}"]  = round(rr   / n, 4)
    metrics["chunk_prec@5"] = round(sum(chunk_hits) / n, 4)
    return metrics


# ──────────────────────────── single-config eval ─────────────────────────────

def _evaluate(
    retriever: Retriever,
    samples: list[dict],
    use_rewrite: bool = False,
    desc: str = "",
) -> dict:
    ranks: list[int | None] = []
    chunk_hits: list[bool]  = []
    total_ms = 0.0

    for record in tqdm(samples, desc=desc, ncols=88):
        query     = record.get("question", "")
        gold_id   = record.get("doc_id", "")
        gold_inds = record.get("gold_inds", {})
        # gold_inds 在 eval.jsonl 里可能存为 list（旧格式兼容）
        if isinstance(gold_inds, list):
            gold_inds = {str(i): v for i, v in enumerate(gold_inds)}

        t0 = time.perf_counter()
        results = retriever.search(query, top_k=max(_KS), rewrite=use_rewrite)
        total_ms += (time.perf_counter() - t0) * 1000

        retrieved_ids = [r.get("doc_id", "") for r in results]

        # doc-level rank（1-based），未命中为 None
        try:
            rank = retrieved_ids.index(gold_id) + 1
        except ValueError:
            rank = None
        ranks.append(rank)

        # chunk-level：top-5 中是否有 chunk 包含 gold 证据文本
        hit = any(
            _chunk_hits_gold(r.get("text", ""), gold_inds)
            for r in results[:5]
        )
        chunk_hits.append(hit)

    metrics = _compute_metrics(ranks, chunk_hits)
    metrics["avg_latency_ms"] = round(total_ms / len(samples), 1)
    return metrics


# ──────────────────────────── main ───────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",       type=int, default=200,
                        help="评测样本数（默认 200）")
    parser.add_argument("--seed",    type=int, default=42)
    parser.add_argument("--configs", type=str, default="1,2,3,4,5",
                        help="要运行的配置编号，逗号分隔（默认全部）")
    return parser.parse_args()


def main():
    args = parse_args()
    run_configs = {int(c) for c in args.configs.split(",")}

    # ── 加载并采样 eval 数据 ──────────────────────────────────────────────
    with open(_EVAL_PATH, encoding="utf-8") as f:
        all_records = [json.loads(line) for line in f if line.strip()]

    random.seed(args.seed)
    samples = random.sample(all_records, min(args.n, len(all_records)))
    print(f"\nLoaded {len(samples)} samples (seed={args.seed})\n")

    # ── 配置定义 ─────────────────────────────────────────────────────────
    # (label, use_rewrite)
    CONFIGS: dict[int, tuple[str, bool]] = {
        1: ("Dense only",       False),
        2: ("+ BM25 hybrid",    False),
        3: ("+ Reranker",       False),
        4: ("+ Query Rewriter", True),
        5: ("+ Header Inject",  True),
    }

    all_results: dict[str, dict] = {}

    # Reranker 在 config 3-5 共享，避免重复加载模型权重
    shared_reranker: Reranker | None = None

    for cfg_id, (label, use_rewrite) in CONFIGS.items():
        if cfg_id not in run_configs:
            continue

        print(f"{'='*60}")
        print(f"Config {cfg_id}: {label}")
        print(f"{'='*60}")

        if cfg_id == 1:
            retriever = Retriever(
                VectorStore(),
                _NullBM25(),
                _PassThroughReranker(),
            )

        elif cfg_id == 2:
            retriever = Retriever(
                VectorStore(),
                BM25Store(),
                _PassThroughReranker(),
            )

        elif cfg_id == 3:
            shared_reranker = shared_reranker or Reranker()
            retriever = Retriever(
                VectorStore(),
                BM25Store(),
                shared_reranker,
            )

        elif cfg_id == 4:
            shared_reranker = shared_reranker or Reranker()
            retriever = Retriever(
                VectorStore(),
                BM25Store(),
                shared_reranker,
                query_rewriter=QueryRewriter(),
            )

        else:  # cfg_id == 5
            shared_reranker = shared_reranker or Reranker()
            retriever = Retriever(
                VectorStore(table_name="chunks_contextual"),
                BM25Store(index_path=_ROOT / "data/bm25_contextual.pkl"),
                shared_reranker,
                query_rewriter=QueryRewriter(),
            )

        metrics = _evaluate(
            retriever, samples,
            use_rewrite=use_rewrite,
            desc=f"[{cfg_id}/5] {label}",
        )
        all_results[label] = metrics

        print(
            f"  Hit@5={metrics['hit@5']:.3f}  "
            f"MRR@5={metrics['mrr@5']:.4f}  "
            f"ChunkPrec@5={metrics['chunk_prec@5']:.3f}  "
            f"Latency={metrics['avg_latency_ms']:.0f}ms\n"
        )

        # Config 切换前释放 GPU 显存（embedding server 模式下为空操作）
        if hasattr(retriever._vs, "offload"):
            retriever._vs.offload()

    # ── 汇总对比表 ───────────────────────────────────────────────────────
    header = f"{'Configuration':<22} {'Hit@1':>6} {'Hit@3':>6} {'Hit@5':>6} {'MRR@5':>7} {'CPrec@5':>8} {'ms/q':>6}"
    print(f"\n{'─'*len(header)}")
    print(header)
    print(f"{'─'*len(header)}")
    for label, m in all_results.items():
        print(
            f"{label:<22} "
            f"{m['hit@1']:>6.3f} "
            f"{m['hit@3']:>6.3f} "
            f"{m['hit@5']:>6.3f} "
            f"{m['mrr@5']:>7.4f} "
            f"{m['chunk_prec@5']:>8.3f} "
            f"{m['avg_latency_ms']:>6.0f}"
        )
    print(f"{'─'*len(header)}\n")

    # ── 保存 JSON ────────────────────────────────────────────────────────
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = _RESULTS_DIR / "ablation_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {"n": len(samples), "seed": args.seed, "results": all_results},
            f, indent=2, ensure_ascii=False,
        )
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
