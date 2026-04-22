"""
mcp_server.py
=============
MCP Server，将 RAG 系统暴露为 Claude Code 可调用的工具。

工具：
  - search_knowledge_base : 检索 FinQA 财务文档
  - calculate             : 安全执行财务数学计算
  - ingest_document       : 摄入新文档到向量数据库

Usage:
    python src/mcp_server.py

接入 Claude Code：在 .claude/settings.json 配置 mcpServers 后自动加载。
"""

import math
import re
import signal
import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from fastmcp import FastMCP

mcp = FastMCP("rag-demo")

# 懒加载，避免 import 时触发模型加载
_retriever = None


def _get_retriever():
    global _retriever
    if _retriever is None:
        from src.bm25_store import BM25Store
        from src.reranker import Reranker
        from src.retriever import Retriever
        from src.vector_store import VectorStore
        _retriever = Retriever(VectorStore(), BM25Store(), Reranker())
    return _retriever


# ------------------------------------------------------------------
# Tool 1: search_knowledge_base
# ------------------------------------------------------------------

@mcp.tool()
def search_knowledge_base(
    query: str,
    ticker: str = "",
    top_k: int = 5,
) -> str:
    """
    在 FinQA 财务文档库中检索与问题相关的段落。

    当用户询问任何涉及财务数据、公司业绩、收入、利润、现金流等问题时调用。
    返回最相关的文档段落及来源信息，用于生成准确答案。

    Args:
        query:  用户问题或检索关键词，使用原始问题效果最好
        ticker: 公司股票代码（如 "ADI"、"INTC"），提供后只在该公司文档中搜索
        top_k:  返回结果数量，默认 5，复杂问题可适当增大
    """
    import concurrent.futures

    where = {"ticker": ticker.upper()} if ticker.strip() else None

    try:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(_get_retriever().search, query, top_k, where)
            results = future.result(timeout=10)
    except concurrent.futures.TimeoutError:
        return "[错误] 检索超时（>10s），请稍后重试。"
    except Exception as e:
        return f"[错误] 检索失败：{e}"

    if not results:
        return "未找到相关文档。请尝试换用其他关键词，或确认该公司文档已摄入。"

    lines = []
    for i, r in enumerate(results, 1):
        doc_id = r.get("doc_id", "unknown")
        score  = r.get("score", 0.0)
        text   = r.get("text", "").strip()
        lines.append(f"[{i}] 来源: {doc_id}  相关度: {score:.3f}\n{text}")

    return "\n\n---\n\n".join(lines)


# ------------------------------------------------------------------
# Tool 2: calculate
# ------------------------------------------------------------------

_ALLOWED_EXPR = re.compile(r'^[\d\s\+\-\*\/\(\)\.\,]+$')
_PCT_TOKEN    = re.compile(r'(\d+(?:\.\d+)?)\s*%')


def _safe_eval(expression: str) -> float:
    """白名单过滤后用受限 eval 执行，只暴露 math 函数。"""
    # 将 50% 替换为 0.5
    expr = _PCT_TOKEN.sub(lambda m: str(float(m.group(1)) / 100), expression)
    expr = expr.replace(",", "")  # 去除千分位逗号

    if not _ALLOWED_EXPR.match(expr):
        raise ValueError(f"表达式含不允许的字符，只支持数字和 + - * / ( ) 运算符")

    result = eval(expr, {"__builtins__": {}}, {
        "sqrt": math.sqrt,
        "abs":  abs,
        "round": round,
    })
    return float(result)


@mcp.tool()
def calculate(
    expression: str,
    description: str = "",
) -> str:
    """
    安全执行财务数学计算。

    当需要计算增长率、占比、差值、比率等数值时调用，不要直接心算。
    只支持 + - * / ( ) 和数字，百分号（%）会自动转换为小数。

    Args:
        expression:  数学表达式，如 "(2100 - 1850) / 1850 * 100"
        description: 本次计算的说明，如 "ADI 2008→2009 营收增长率"（可选）
    """
    try:
        result = _safe_eval(expression)
    except Exception as e:
        return f"[错误] 计算失败：{e}\n表达式：{expression}"

    # 格式化输出
    if abs(result) >= 1e6:
        formatted = f"{result:,.2f}"
    elif abs(result) < 0.01 and result != 0:
        formatted = f"{result:.6f}"
    else:
        formatted = f"{result:.4f}".rstrip("0").rstrip(".")

    lines = [f"计算: {expression}", f"结果: {formatted}"]
    if description:
        lines.append(f"说明: {description}")
    return "\n".join(lines)


# ------------------------------------------------------------------
# Tool 3: ingest_document
# ------------------------------------------------------------------

@mcp.tool()
def ingest_document(
    text: str,
    doc_id: str,
    source: str = "",
) -> str:
    """
    将新文档摄入到向量数据库，使其可被检索。

    当用户上传或提供新的财务文档时调用。
    文档摄入后立即可用于 search_knowledge_base 检索。

    Args:
        text:   文档全文（Markdown 或纯文本）
        doc_id: 文档唯一标识，如 "TSLA_2024_annual_report"
        source: 文档来源说明（可选），如 "用户上传" 或 URL
    """
    import concurrent.futures

    from src.bm25_store import BM25Store
    from src.chunk_manager import ChunkManager
    from src.ingestion_registry import IngestionRegistry
    from src.vector_store import VectorStore

    try:
        vs       = VectorStore()
        chunker  = ChunkManager()
        bm25     = BM25Store()
        registry = IngestionRegistry()

        if registry.is_registered(doc_id):
            return f"文档 {doc_id!r} 已存在，跳过摄入。如需更新请先删除旧数据。"

        chunks = chunker.split(text, doc_id=doc_id)
        if not chunks:
            return f"[错误] 文档 {doc_id!r} 切分后为空，请检查文本内容。"

        header = f"[{source}]\n" if source else ""

        def _do_ingest():
            vs.add_documents(chunks, header_override=header or None)
            bm25.add_documents(chunks, header_override={c["doc_id"]: header for c in chunks} if header else {})
            registry.register(doc_id)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(_do_ingest)
            future.result(timeout=10)

    except concurrent.futures.TimeoutError:
        return f"[错误] 摄入超时（>10s），文档 {doc_id!r} 可能未完整写入。"
    except Exception as e:
        return f"[错误] 摄入失败：{e}"

    return (
        f"文档摄入成功\n"
        f"doc_id : {doc_id}\n"
        f"chunks : {len(chunks)}\n"
        f"source : {source or '未指定'}"
    )


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

if __name__ == "__main__":
    mcp.run()
