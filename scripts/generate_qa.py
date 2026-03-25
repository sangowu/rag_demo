"""
generate_qa.py
==============
从 FinQA 文档批量生成 QA pairs，每个文档生成 1 条。

流程：
  1. 读取文档，选取最具代表性的 chunk（含数字最多的）
  2. 第一次 LLM 调用：生成 question + answer_span + answer
  3. 第二次 LLM 调用：验证 QA pair 质量
  4. 通过验证的写入 data/results/qa_pairs.jsonl

Usage:
    python scripts/generate_qa.py [--n 100] [--output data/results/qa_pairs.jsonl]
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

from tqdm import tqdm

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src.config import config

# ── LLM ──────────────────────────────────────────────────────────────────────
_llm_cfg     = config.get("llm", {})
_llm_qa_cfg  = config.get("llm_qa_gen", {})
_llm = ChatOpenAI(
    model=_llm_cfg.get("model", "Qwen/Qwen3-8B"),
    base_url=_llm_cfg.get("base_url", "http://localhost:8001/v1"),
    api_key=os.environ.get(_llm_cfg.get("api_key_env", "MODELSCOPE_API_KEY"), "local"),
    temperature=_llm_qa_cfg.get("temperature", 0.7),
    max_tokens=_llm_qa_cfg.get("max_tokens", 512),
    model_kwargs={"extra_body": {
        "top_p": 0.8,
        "top_k": 20,
        "min_p": 0.0,
        "chat_template_kwargs": {"enable_thinking": False},
    }},
)

_DOCS_DIR    = _ROOT / "data/finqa/docs"
_RESULTS_DIR = _ROOT / "data/results"

# ── Prompts ───────────────────────────────────────────────────────────────────
_GEN_SYSTEM = """You are a financial QA dataset creator.
Given a financial report excerpt, generate exactly ONE high-quality question-answer pair.

Rules:
- The question MUST include the company name and fiscal year (extract from doc_id or text)
- The answer_span MUST be copied verbatim from the text, word for word
- The answer MUST be directly readable from answer_span (no calculation needed)
- Focus on specific financial figures (revenue, expenses, income, margins, etc.)

Output valid JSON only, no extra text:
{
  "question": "...",
  "answer_span": "exact sentence copied from text",
  "answer": "specific value or fact"
}"""

_GEN_HUMAN = """doc_id: {doc_id}

Text:
{chunk_text}"""

_VAL_SYSTEM = """You are a QA quality checker for financial datasets.
Evaluate whether the QA pair meets ALL of these criteria:
1. answer_span is copied verbatim from the text (not paraphrased)
2. answer can be read directly from answer_span (no calculation required)
3. question includes a company name and year

Output valid JSON only:
{"valid": true/false, "reason": "one sentence"}"""

_VAL_HUMAN = """Text:
{chunk_text}

Question: {question}
Answer Span: {answer_span}
Answer: {answer}"""


# ── Helpers ───────────────────────────────────────────────────────────────────
def _pick_chunk(doc_text: str) -> str:
    """选取含数字最多的段落作为生成 QA 的来源。"""
    paragraphs = [p.strip() for p in doc_text.split("\n\n") if len(p.strip()) > 80]
    if not paragraphs:
        return doc_text[:1500]
    scored = [(len(re.findall(r"\d", p)), p) for p in paragraphs]
    scored.sort(reverse=True)
    return scored[0][1][:1500]


def _parse_json(content: str) -> dict | None:
    """从 LLM 输出中提取 JSON，容忍 think 标签和 markdown 代码块包裹。"""
    # 去除 <think>...</think> 块
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
        resp = _llm.invoke([
            SystemMessage(_GEN_SYSTEM),
            HumanMessage(_GEN_HUMAN.format(doc_id=doc_id, chunk_text=chunk_text)),
        ])
        return _parse_json(resp.content)
    except Exception as e:
        print(f"  [GEN ERROR] {e}")
        return None


def validate(qa: dict, chunk_text: str) -> bool:
    """第二次调用：验证 QA pair 质量。"""
    try:
        resp = _llm.invoke([
            SystemMessage(_VAL_SYSTEM),
            HumanMessage(_VAL_HUMAN.format(
                chunk_text=chunk_text,
                question=qa.get("question", ""),
                answer_span=qa.get("answer_span", ""),
                answer=qa.get("answer", ""),
            )),
        ])
        result = _parse_json(resp.content)
        if result and result.get("valid"):
            return True
        reason = result.get("reason", "unknown") if result else "parse error"
        print(f"  [INVALID] {reason}")
        return False
    except Exception as e:
        print(f"  [VAL ERROR] {e}")
        return False


# ── Main ──────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",      type=int, default=100,
                        help="处理文档数（默认 100）")
    parser.add_argument("--output", type=str, default="data/results/qa_pairs.jsonl",
                        help="输出路径")
    return parser.parse_args()


def main():
    args = parse_args()

    doc_paths = sorted(_DOCS_DIR.glob("*.md"))[: args.n]
    if not doc_paths:
        print(f"[ERROR] No documents found in {_DOCS_DIR}")
        sys.exit(1)

    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = _ROOT / args.output

    passed = 0
    failed = 0

    with open(out_path, "w", encoding="utf-8") as out_f:
        for path in tqdm(doc_paths, desc="Generating QA"):
            doc_id     = path.stem
            doc_text   = path.read_text(encoding="utf-8")
            chunk_text = _pick_chunk(doc_text)

            # 第一次调用：生成
            qa = generate(doc_id, chunk_text)
            if not qa:
                failed += 1
                continue

            # 第二次调用：验证
            if not validate(qa, chunk_text):
                failed += 1
                continue

            record = {
                "doc_id":      doc_id,
                "question":    qa["question"],
                "answer_span": qa["answer_span"],
                "answer":      qa["answer"],
                "chunk_text":  chunk_text,
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_f.flush()
            passed += 1

    print(f"\nDone. Passed: {passed} | Failed/Invalid: {failed} | Total: {len(doc_paths)}")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
