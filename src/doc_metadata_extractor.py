"""
doc_metadata_extractor.py
=========================
从文档正文（前 2000 字）用 LLM 提取结构化元信息。

适用于任意来源文档（网页、PDF、Markdown），
不依赖文件名格式（与 metadata_extractor.py 的文件名解析互补）。

提取字段：
    company_name : 公司全称（如 "Analog Devices Inc."）
    ticker       : 股票代码（如 "ADI"，可能为空）
    year         : 财报年份（如 "2009"，可能为空）
    doc_type     : 文档类型（如 "annual report"、"earnings release"）

Usage:
    from src.doc_metadata_extractor import extract_doc_metadata_llm
    meta = extract_doc_metadata_llm("Analog Devices reported revenue of $2.1B in fiscal 2009...")
    # → {"company_name": "Analog Devices", "ticker": "ADI", "year": "2009", "doc_type": "annual report"}
"""

import os
import re

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from src.config import config

_llm_cfg = config.get("llm", {})
_is_modelscope = "modelscope" in _llm_cfg.get("base_url", "")

_PROMPT = """\
/no_think
Read the following text excerpt from a financial document and extract structured metadata.
Reply ONLY with a JSON object, no explanation.

Text:
{text}

Extract:
{{
  "company_name": "<full company name, or empty string if unknown>",
  "ticker": "<stock ticker symbol, or empty string if not mentioned>",
  "year": "<fiscal year as 4-digit string, or empty string if unknown>",
  "doc_type": "<document type, e.g. annual report / earnings release / 10-K / news article>"
}}"""


class _DocMeta(BaseModel):
    company_name: str = Field(default="")
    ticker:       str = Field(default="")
    year:         str = Field(default="")
    doc_type:     str = Field(default="")


_llm = ChatOpenAI(
    model=_llm_cfg.get("model", "Qwen/Qwen3-8B"),
    base_url=_llm_cfg.get("base_url", "http://localhost:8000/v1"),
    api_key=os.environ.get(_llm_cfg.get("api_key_env", "MODELSCOPE_API_KEY"), "local"),
    temperature=0.0,
    max_tokens=128,
    extra_body={"enable_thinking": False},
)

_structured_llm = _llm.with_structured_output(_DocMeta)


def extract_doc_metadata_llm(text: str, max_chars: int = 2000) -> dict:
    """
    从文档正文提取元信息。只读前 max_chars 字（通常是封面/开头页）。

    Args:
        text      : 文档全文
        max_chars : 送给 LLM 的最大字符数（默认 2000，降低 token 消耗）

    Returns:
        dict: company_name, ticker, year, doc_type
              提取失败时返回全空 dict，不抛异常
    """
    excerpt = text[:max_chars].strip()
    if not excerpt:
        return {"company_name": "", "ticker": "", "year": "", "doc_type": ""}

    try:
        meta = _structured_llm.invoke(_PROMPT.format(text=excerpt))
        return {
            "company_name": meta.company_name.strip(),
            "ticker":       meta.ticker.strip().upper(),
            "year":         re.sub(r"\D", "", meta.year)[:4],  # 只保留数字，最多4位
            "doc_type":     meta.doc_type.strip(),
        }
    except Exception:
        return {"company_name": "", "ticker": "", "year": "", "doc_type": ""}


def build_chunk_header(meta: dict) -> str:
    """
    根据提取到的元信息构造 chunk header。
    company_name 优先于 ticker，有什么用什么，无法提取时返回空字符串。

    Examples:
        {"company_name": "Analog Devices", "year": "2009"} → "[Analog Devices | 2009]"
        {"ticker": "ADI", "year": "2009"}                  → "[ADI | 2009]"
        {}                                                  → ""
    """
    name = meta.get("company_name") or meta.get("ticker") or ""
    year = meta.get("year") or ""

    if name and year:
        return f"[{name} | {year}]\n"
    elif name:
        return f"[{name}]\n"
    return ""
