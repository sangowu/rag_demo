"""
guardrails.py
=============
输入/输出双层校验，拦截无关 query 和无依据答案。

Input guardrail  — query 是否与金融文档相关
Output guardrail — answer 中的关键数字是否有 chunk 支撑

Usage:
    from src.guardrails import Guardrails
    g = Guardrails()
    ok, reason = g.check_input("What was ADI revenue in 2009?")
    ok, reason = g.check_output(answer, chunks)
"""

import os
import re

from langchain_openai import ChatOpenAI
from src.config import config

_llm_cfg = config.get("llm", {})
_is_modelscope = "modelscope" in _llm_cfg.get("base_url", "")

_INPUT_PROMPT = """\
/no_think
Is the following query related to financial documents, company reports, \
revenue, earnings, or business metrics? Reply with ONLY "yes" or "no".

Query: {query}"""


class Guardrails:
    def __init__(self):
        # 输入检查用轻量 LLM，max_tokens=4 足够输出 yes/no
        self._llm = ChatOpenAI(
            model=_llm_cfg.get("model", "Qwen/Qwen3-8B"),
            base_url=_llm_cfg.get("base_url", "http://localhost:8000/v1"),
            api_key=os.environ.get(_llm_cfg.get("api_key_env", "MODELSCOPE_API_KEY"), "local"),
            temperature=0.0,
            max_tokens=16,
            extra_body={"enable_thinking": False},  # 本地和 ModelScope 都禁用 thinking
        )

    def check_input(self, query: str) -> tuple[bool, str]:
        """
        检查 query 是否与金融文档相关。

        Returns:
            (True, "")            — 通过，正常走检索
            (False, reason)       — 拦截，直接返回 reason 给用户
        """
        # 空 query 快速拒绝
        if not query or not query.strip():
            return False, "Query is empty."

        # 过长 query 拒绝（防止 prompt injection）
        if len(query) > 500:
            return False, "Query is too long (max 500 characters)."

        try:
            result = self._llm.invoke(_INPUT_PROMPT.format(query=query))
            # 去掉 Qwen3 thinking block（<think>...</think>），再找 yes/no
            raw = re.sub(r"<think>.*?</think>", "", result.content, flags=re.DOTALL)
            answer = raw.strip().lower()
            has_yes = "yes" in answer
            has_no  = "no"  in answer
            # 明确包含 no 且不含 yes → 拦截；其余情况放行（优先避免误拦截）
            if has_no and not has_yes:
                return False, "This system only answers questions about financial documents and company reports."
            return True, ""
        except Exception:
            # LLM 调用失败时放行，避免误拦截
            return True, ""

    def check_output(self, answer: str, chunks: list[dict]) -> tuple[bool, str]:
        """
        检查 answer 中的关键数字是否出现在 chunks 里（规则检查，不调 LLM）。

        逻辑：
          - 提取 answer 中所有数字（金额、百分比、年份）
          - 检查这些数字中至少有一个出现在某个 chunk 的文本里
          - 若 answer 没有数字，视为通过（描述性回答难以用数字核验）

        Returns:
            (True, "")          — 通过
            (False, reason)     — 答案数字无法在 context 中找到支撑
        """
        if not answer or not chunks:
            return True, ""

        # 提取 answer 里的数字（忽略单独的年份，重点看金融数字）
        numbers = re.findall(r'\b\d[\d,\.]*\d\b', answer)
        if not numbers:
            return True, ""

        # 合并所有 chunk 文本
        context = " ".join(c.get("text", "") for c in chunks)

        # 至少有一个数字能在 context 里找到
        for num in numbers:
            # 去掉千分位逗号后比较
            normalized = num.replace(",", "")
            if num in context or normalized in context:
                return True, ""

        return False, "Answer may not be grounded in retrieved documents."
