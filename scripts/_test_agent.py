"""
_test_agent.py
==============
验证 Planner 动态路由 + Grounding Check 的端到端行为。

测试场景：
  1. 正常金融问题        → should_retrieve=True，grounding ok，生成答案
  2. 问候 / 常识问题     → should_retrieve=False，跳过检索，直接生成
  3. 超出知识库的问题    → should_retrieve=True，grounding weak，触发拒答

Usage:
    python scripts/_test_agent.py
    python scripts/_test_agent.py --case 1      # 只跑指定场景
    python scripts/_test_agent.py --verbose     # 打印完整 state
"""

import argparse
import sys
import textwrap
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from src.agent.graph import graph
from src.agent.nodes import _GROUNDING_THRESHOLD
from src.tracing import record_agent_trace

# ── 初始 state 模板 ────────────────────────────────────────────────────────────
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


# ── 测试用例 ───────────────────────────────────────────────────────────────────
CASES = [
    {
        "id": 1,
        "name": "正常金融问题（应检索 + 生成）",
        "query": "What was ADI's total revenue in 2009?",
        "expect_retrieve": True,
        "expect_refuse": False,
    },
    {
        "id": 2,
        "name": "问候 / 常识（应跳过检索）",
        "query": "Hello, what can you help me with?",
        "expect_retrieve": False,
        "expect_refuse": False,
    },
    {
        "id": 3,
        "name": "超出知识库范围（TAVILY 可用时 web search fallback，否则拒答）",
        "query": "What is the current stock price of Apple right now?",
        "expect_retrieve": True,
        "expect_refuse": False,    # TAVILY 可用时走 web_search → 不拒答
    },
    {
        "id": 4,
        "name": "完全无法回答的问题（Planner 跳过检索，LLM 直接说无法回答）",
        "query": "xyzzy qwerty nonsense financial question 12345",
        "expect_retrieve": False,
        "expect_refuse": False,
    },
]


# ── 单场景运行 ─────────────────────────────────────────────────────────────────
def run_case(case: dict, verbose: bool = False) -> bool:
    print(f"\n{'='*60}")
    print(f"[Case {case['id']}] {case['name']}")
    print(f"Query: {case['query']}")
    print("─" * 60)

    state = _init_state(case["query"])
    result = graph.invoke(state)
    record_agent_trace(case["query"], result, run_name=f"test_case_{case['id']}")

    # ── 取关键字段 ──────────────────────────────────────────────────────────
    did_retrieve  = result.get("should_retrieve", True)
    refused       = result.get("refuse", False)
    grounding     = result.get("grounding_score")
    final_answer  = result.get("final_answer", "")
    metrics       = result.get("metrics", {})

    # ── 打印摘要 ────────────────────────────────────────────────────────────
    print(f"should_retrieve : {did_retrieve}")
    grounding_str = f"{grounding:.3f}" if grounding is not None else "N/A"
    print(f"grounding_score : {grounding_str}  (threshold={_GROUNDING_THRESHOLD})")
    print(f"refused         : {refused}")
    print(f"retry_count     : {metrics.get('retry_count', 0)}")
    print(f"latency_ms      : {metrics.get('latency_ms', 'N/A')}")
    print(f"\nAnswer:\n{textwrap.fill(final_answer[:400], width=72)}")

    trace = metrics.get("trace", result.get("trace", []))
    if trace:
        print("\nTrace:")
        for entry in trace:
            step = entry.pop("step", "?")
            details = "  ".join(f"{k}={v}" for k, v in entry.items() if v is not None)
            print(f"  [{step}] {details}")

    if verbose:
        print(f"\n[Full state keys]: {list(result.keys())}")

    # ── 断言 ────────────────────────────────────────────────────────────────
    passed = True
    checks = [
        ("should_retrieve", did_retrieve, case["expect_retrieve"]),
        ("refuse",          refused,      case["expect_refuse"]),
    ]
    for field, got, want in checks:
        ok = (got == want)
        status = "✓" if ok else "✗"
        print(f"  {status} {field}: expected={want}, got={got}")
        if not ok:
            passed = False

    print(f"\n{'PASS' if passed else 'FAIL'}")
    return passed


# ── 主入口 ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case",    type=int, default=None, help="只运行指定 case id")
    parser.add_argument("--verbose", action="store_true",    help="打印完整 state")
    args = parser.parse_args()

    cases = [c for c in CASES if args.case is None or c["id"] == args.case]
    if not cases:
        print(f"Case {args.case} not found.")
        sys.exit(1)

    results = [run_case(c, verbose=args.verbose) for c in cases]

    print(f"\n{'='*60}")
    passed = sum(results)
    print(f"Result: {passed}/{len(results)} passed")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
