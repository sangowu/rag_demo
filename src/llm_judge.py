"""
llm_judge.py
============
LLM-as-Judge：对单条 RAG 结果进行逐条评分，输出分数和评估理由。

评分维度（1-5 分）：
  - 答案是否准确回答了问题
  - 答案是否有检索内容支撑
  - 答案是否简洁清晰

输出结构：
  {"score": 4, "reason": "...", "passed": True}

Usage:
    from src.llm_judge import LLMJudge
    judge = LLMJudge()
    result = judge.evaluate(question, answer, contexts)
"""

import json
import os
import re

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src.config import config

_llm_cfg = config.get("llm", {})

_llm = ChatOpenAI(
    model=_llm_cfg.get("model", "Qwen/Qwen3-8B"),
    base_url=_llm_cfg.get("base_url", "https://api-inference.modelscope.cn/v1"),
    api_key=os.environ.get(_llm_cfg.get("api_key_env", "MODELSCOPE_API_KEY"), ""),
    temperature=0.0,   # 评分要稳定，temperature=0
    max_tokens=512,
)

_SYSTEM_PROMPT = """You are an expert evaluator for RAG (Retrieval-Augmented Generation) systems.
Given a question, a generated answer, and the retrieved context, evaluate the answer quality.

Score the answer on a scale of 1-5:
  5 - Excellent: accurate, well-supported by context, clear and concise
  4 - Good: mostly accurate, supported by context, minor issues
  3 - Acceptable: partially correct, somewhat supported, noticeable gaps
  2 - Poor: inaccurate or unsupported by context
  1 - Bad: completely wrong or irrelevant

Respond in JSON format only:
{"score": <1-5>, "reason": "<brief explanation>"}"""


class LLMJudge:
    def evaluate(
        self,
        question: str,
        answer: str,
        contexts: list[str],
    ) -> dict:
        """
        对单条 RAG 结果进行 LLM 评分。

        Args:
            question : 用户问题
            answer   : RAG 生成的答案
            contexts : 检索到的 chunk 文本列表

        Returns:
            {"score": int, "reason": str, "passed": bool}
        """
        context = "\n\n".join(f"[{i+1}] {c}" for i , c in enumerate(contexts))
        human_prompt = (
            f"Question: {question}\n\n"
            f"Answer: {answer}\n\n"
            f"Context: \n{context}"
        )

        response = _llm.invoke([SystemMessage(_SYSTEM_PROMPT), HumanMessage(human_prompt)])
        return self._parse_response(response.content)

    def _parse_response(self, content: str) -> dict:
        """
        从 LLM 输出中提取 JSON，容错处理非标准输出。

        Returns:
            {"score": int, "reason": str, "passed": bool}
        """
        try:
            res = json.loads(content)
            score, reason = res["score"], res["reason"]
        except (json.JSONDecodeError, KeyError):
            match = re.search(r'"score"\s*:\s*(\d)', content)
            score = int(match.group(1)) if match else 1
            reason = "parse error"
        score = max(1, min(5, int(score)))
        return {"score": score, "reason": reason, "passed": score >= 3}
        
