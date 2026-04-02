"""
metadata_extractor.py
=====================
从 FinQA doc_id 和 query 中提取结构化元信息，用于 ChromaDB 预过滤以缩小检索范围。

Doc metadata（从 doc_id 解析，无需 LLM）：
  ticker : 公司代码，如 "ADI"
  year   : 财年，如 "2009"

Query metadata（正则 + ticker 词典匹配）：
  ticker : 从 query 中识别公司名或代码
  year   : 从 query 中识别四位年份

Usage:
    from src.metadata_extractor import extract_doc_metadata, extract_query_metadata

    meta = extract_doc_metadata("ADI_2009_page_49.pdf")
    # → {"ticker": "ADI", "year": "2009"}

    meta = extract_query_metadata("What was ADI revenue in 2009?", known_tickers)
    # → {"ticker": "ADI", "year": "2009"}
"""

import re
from pathlib import Path


def extract_doc_metadata(doc_id: str) -> dict:
    """
    从 doc_id 解析结构化元信息。

    doc_id 格式：{TICKER}_{YEAR}_page_{N}.pdf
    示例：ADI_2009_page_49.pdf → {"ticker": "ADI", "year": "2009"}

    Args:
        doc_id: 文档 ID 字符串（含或不含 .pdf 后缀均可）

    Returns:
        dict with keys: ticker, year（解析失败时对应值为空字符串）
    """
    stem = Path(doc_id).stem  # 去掉 .pdf
    parts = stem.split("_")

    ticker = parts[0] if len(parts) >= 1 else ""
    year   = parts[1] if len(parts) >= 2 and parts[1].isdigit() else ""

    return {"ticker": ticker, "year": year}


_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")


def extract_query_metadata(query: str, known_tickers: set[str]) -> dict:
    """
    从 query 文本中提取年份和公司代码。

    Args:
        query         : 用户查询文本
        known_tickers : 已知公司代码集合（从索引中获取）

    Returns:
        dict，只包含成功识别的字段（ticker 和/或 year），
        识别不到则不包含该 key，避免误过滤。
    """
    meta = {}

    # 提取年份
    years = _YEAR_RE.findall(query)
    if len(years) == 1:
        meta["year"] = years[0]

    # 提取 ticker：先按词边界精确匹配大写词，再与已知 ticker 集合取交
    tokens = re.findall(r"\b[A-Z]{2,6}\b", query)
    matched = [t for t in tokens if t in known_tickers]
    if len(matched) == 1:
        meta["ticker"] = matched[0]

    return meta


def build_chroma_filter(meta: dict) -> dict | None:
    """
    将提取到的元信息转换为 ChromaDB where 过滤条件。

    单字段：{"ticker": "ADI"}
    多字段：{"$and": [{"ticker": {"$eq": "ADI"}}, {"year": {"$eq": "2009"}}]}

    Args:
        meta: extract_query_metadata 的返回值

    Returns:
        ChromaDB where dict，无可用字段时返回 None
    """
    conditions = [
        {k: {"$eq": v}} for k, v in meta.items() if k in ("ticker", "year")
    ]
    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}
