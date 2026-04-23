"""
generate_qa.py
==============
从 FinQA 文档批量生成 QA pairs，每个文档生成 1 条。

流程：
  1. 按 --sample-mode 选取文档（顺序或按 ticker 分层轮询）
  2. 按 doc_id 稳定哈希选择选段策略（叙述 / 平衡 / 数值），对表格行做惩罚
  3. LLM 生成 question + answer_span + answer
  4. 规则校验后写入 JSONL

Usage:
    python scripts/generate_qa.py [--n 100] [--sample-mode stratified] [--seed 42]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from langchain_core.messages import HumanMessage, SystemMessage

from src.config import config
from src.llm_factory import get_llm

logger = logging.getLogger(__name__)

# ── Chunk / sampling constants (no magic numbers in logic) ───────────────────
MIN_PARA_LEN = 80
MAX_CHUNK_CHARS = 1500
TABLE_PENALTY_WEIGHT = 1.5
BALANCED_TABLE_PENALTY = 0.5
NARRATIVE_TABLE_FALLBACK_THRESHOLD = 50
NUM_STRATEGIES = 3
STRATEGY_LABELS = ("narrative", "balanced", "numeric")

# ── LLM ──────────────────────────────────────────────────────────────────────
_llm_qa_cfg = config.get("llm_qa_gen", {})
_llm = get_llm(
    temperature=_llm_qa_cfg.get("temperature", 0.7),
    max_output_tokens=_llm_qa_cfg.get("max_tokens", 512),
)

_DOCS_DIR = _ROOT / "data/finqa/docs"
_RESULTS_DIR = _ROOT / "data/results"

# ── Prompts ───────────────────────────────────────────────────────────────────
_GEN_SYSTEM = """You are a financial QA dataset creator.
Given a financial report excerpt, generate exactly ONE high-quality question-answer pair.

Rules:
- The question MUST include the company name and fiscal year (extract from doc_id or text)
- The answer_span MUST be copied verbatim from the text, word for word
- The answer MUST be directly readable from answer_span (no calculation needed)
- Vary question types: factual (what/who/where), trend (increased/decreased), comparison, risk factors, business strategy — not just numbers

Output valid JSON only, no extra text:
{
  "question": "...",
  "answer_span": "exact sentence copied from text",
  "answer": "specific value or fact"
}"""

_GEN_HUMAN = """doc_id: {doc_id}

Text:
{chunk_text}"""


# ── Sampling & chunk selection ────────────────────────────────────────────────
def stratify_key(path: Path) -> str:
    """
    Group documents for stratified sampling (e.g. ticker = segment before first '_').

    Args:
        path: Path to a .md file.

    Returns:
        Stratification bucket key; never empty.
    """
    stem = path.stem
    if "_" in stem:
        return stem.split("_", 1)[0]
    return stem or "_other"


def stable_strategy_bucket(doc_id: str, num_strategies: int = NUM_STRATEGIES) -> int:
    """
    Deterministic strategy index per doc_id (stable across runs and processes).

    Args:
        doc_id: Document stem id.
        num_strategies: Number of chunk strategies (default 3).

    Returns:
        Integer in [0, num_strategies).
    """
    digest = hashlib.md5(doc_id.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % num_strategies


def _digit_count(paragraph: str) -> int:
    return len(re.findall(r"\d", paragraph))


def _table_weight(paragraph: str) -> int:
    return paragraph.count("|")


def pick_chunk(doc_text: str, doc_id: str) -> tuple[str, str]:
    """
    Select a chunk from document text using strategy derived from doc_id.

    Strategies:
        narrative — prefer long prose, low digit density, penalize markdown tables
        balanced — compromise between narrative and numeric cues
        numeric — prefer paragraphs rich in digits (financial tables)

    Args:
        doc_text: Full markdown body.
        doc_id: Document id (used for stable strategy selection).

    Returns:
        Tuple of (chunk_text truncated to MAX_CHUNK_CHARS, strategy label).
    """
    paragraphs = [
        p.strip() for p in doc_text.split("\n\n") if len(p.strip()) > MIN_PARA_LEN
    ]
    if not paragraphs:
        return doc_text[:MAX_CHUNK_CHARS], STRATEGY_LABELS[0]

    strategy = stable_strategy_bucket(doc_id, NUM_STRATEGIES)

    if strategy == 0:

        def score_narrative(p: str) -> float:
            ln = max(len(p), 1)
            return ln * (1 - _digit_count(p) / ln) - TABLE_PENALTY_WEIGHT * _table_weight(p)

        ranked = sorted(paragraphs, key=score_narrative, reverse=True)
        chosen = ranked[0]
        if (
            _table_weight(chosen) > NARRATIVE_TABLE_FALLBACK_THRESHOLD
            and len(ranked) > 1
            and _table_weight(ranked[1]) < _table_weight(chosen)
        ):
            chosen = ranked[1]
        label = STRATEGY_LABELS[0]
    elif strategy == 1:

        def score_balanced(p: str) -> float:
            ln = max(len(p), 1)
            mix = 0.5 + 0.5 * (1 - _digit_count(p) / ln)
            return ln * mix - BALANCED_TABLE_PENALTY * _table_weight(p)

        chosen = sorted(paragraphs, key=score_balanced, reverse=True)[0]
        label = STRATEGY_LABELS[1]
    else:
        chosen = sorted(paragraphs, key=_digit_count, reverse=True)[0]
        label = STRATEGY_LABELS[2]

    return chosen[:MAX_CHUNK_CHARS], label


def sample_doc_paths(all_paths: list[Path], n: int, seed: int, mode: str) -> list[Path]:
    """
    Select up to n document paths.

    sequential: sorted by path, first n.
    stratified: group by stratify_key(), shuffle within group, round-robin across groups.

    Args:
        all_paths: All candidate .md paths.
        n: Target count.
        seed: RNG seed for stratified shuffle (ignored for sequential).
        mode: 'sequential' or 'stratified'.

    Returns:
        Selected paths (length <= min(n, len(all_paths))).
    """
    if not all_paths:
        return []
    if n <= 0:
        return []

    if mode == "sequential":
        return sorted(all_paths)[:n]

    rng = random.Random(seed)
    buckets: dict[str, list[Path]] = defaultdict(list)
    for p in sorted(all_paths):
        buckets[stratify_key(p)].append(p)
    for key in buckets:
        rng.shuffle(buckets[key])

    ordered_keys = sorted(buckets.keys())
    selected: list[Path] = []
    while len(selected) < n:
        progressed = False
        for key in ordered_keys:
            if len(selected) >= n:
                break
            if buckets[key]:
                selected.append(buckets[key].pop(0))
                progressed = True
        if not progressed:
            break
    return selected


def _parse_json(content: str) -> dict | None:
    """从 LLM 输出中提取 JSON，容忍 think 标签和 markdown 代码块包裹。"""
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
    content = re.sub(r"```json|```", "", content).strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                return None
    return None


def generate(doc_id: str, chunk_text: str) -> dict | None:
    """第一次调用：生成 QA pair。"""
    try:
        resp = _llm.invoke(
            [
                SystemMessage(_GEN_SYSTEM),
                HumanMessage(_GEN_HUMAN.format(doc_id=doc_id, chunk_text=chunk_text)),
            ]
        )
        return _parse_json(resp.content)
    except Exception as e:
        logger.warning("LLM generate failed for %s: %s", doc_id, e)
        tqdm.write(f"  [GEN ERROR] {e}")
        return None


def validate(qa: dict, chunk_text: str) -> bool:
    """规则验证 QA pair 质量，无需额外 LLM 调用。"""
    span = qa.get("answer_span", "").strip()
    answer = qa.get("answer", "").strip()

    if not span or span not in chunk_text:
        tqdm.write("  [INVALID] answer_span not found in text")
        return False

    if not answer:
        tqdm.write("  [INVALID] empty answer")
        return False

    if len(span) < 15:
        tqdm.write("  [INVALID] answer_span too short")
        return False

    return True


# ── Main ──────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=100, help="处理文档数（默认 100）")
    parser.add_argument(
        "--output",
        type=str,
        default="data/results/qa_pairs.jsonl",
        help="输出路径",
    )
    parser.add_argument(
        "--sample-mode",
        choices=("sequential", "stratified"),
        default="stratified",
        help="文档抽样：sequential=排序取前 n；stratified=按 ticker 分层轮询（默认）",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="stratified 模式下各层内打乱与轮询顺序的随机种子",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()

    all_md = list(_DOCS_DIR.glob("*.md"))
    doc_paths = sample_doc_paths(all_md, args.n, args.seed, args.sample_mode)
    if not doc_paths:
        print(f"[ERROR] No documents found in {_DOCS_DIR}")
        sys.exit(1)

    logger.info(
        "sample_mode=%s seed=%s picked=%s of %s total md",
        args.sample_mode,
        args.seed,
        len(doc_paths),
        len(all_md),
    )

    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = _ROOT / args.output

    passed = 0
    failed = 0

    with open(out_path, "w", encoding="utf-8") as out_f:
        for path in tqdm(doc_paths, desc="Generating QA"):
            doc_id = path.name
            doc_text = path.read_text(encoding="utf-8")
            chunk_text, strategy_label = pick_chunk(doc_text, doc_id)

            qa = generate(doc_id, chunk_text)
            if not qa:
                failed += 1
                continue

            if not validate(qa, chunk_text):
                failed += 1
                tqdm.write(
                    f"  [FAIL] {doc_id} | span_in_text={qa.get('answer_span', '')[:50]!r}"
                )
                continue

            record = {
                "doc_id": doc_id,
                "question": qa["question"],
                "answer_span": qa["answer_span"],
                "answer": qa["answer"],
                "chunk_text": chunk_text,
                "chunk_strategy": strategy_label,
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_f.flush()
            passed += 1

    print(f"\nDone. Passed: {passed} | Failed/Invalid: {failed} | Total: {len(doc_paths)}")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
