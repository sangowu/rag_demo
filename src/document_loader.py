"""
document_loader.py
==================
多来源文档加载器，将不同格式统一转换为纯文本。

支持来源：
  - URL      : 抓取网页正文（requests + BeautifulSoup）
  - PDF      : 提取文字层（pdfplumber）
  - Markdown : 直接读取
  - Text     : 直接读取

所有 loader 输出统一格式：
    {
        "text"       : str,   # 纯文本内容
        "doc_id"     : str,   # 唯一标识（由调用方提供或自动生成）
        "source_type": str,   # "url" | "pdf" | "markdown" | "text"
        "title"      : str,   # 文档标题（可能为空）
        "source"     : str,   # 原始来源路径或 URL
    }

Usage:
    from src.document_loader import DocumentLoader
    loader = DocumentLoader()
    doc = loader.load_url("https://example.com/annual-report")
    doc = loader.load_pdf("/path/to/report.pdf")
"""

import hashlib
import re
from pathlib import Path


class DocumentLoader:

    # ------------------------------------------------------------------
    # URL
    # ------------------------------------------------------------------

    def load_url(self, url: str, doc_id: str = None) -> dict:
        """
        抓取网页并提取正文文本。
        去除导航栏、广告等噪音，保留 <article>/<main>/<p> 内容。

        Args:
            url    : 目标网页 URL
            doc_id : 自定义 ID，默认用 URL 的 MD5 前 8 位

        Returns:
            统一文档 dict
        """
        try:
            import requests
            from bs4 import BeautifulSoup
        except ImportError:
            raise ImportError("请安装依赖: pip install requests beautifulsoup4")

        resp = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")

        # 去除噪音标签
        for tag in soup(["script", "style", "nav", "header", "footer",
                         "aside", "form", "iframe"]):
            tag.decompose()

        # 优先取语义正文区域
        body = (
            soup.find("article") or
            soup.find("main") or
            soup.find(id=re.compile(r"content|main|article", re.I)) or
            soup.find(class_=re.compile(r"content|main|article|post", re.I)) or
            soup.body
        )

        title = soup.title.string.strip() if soup.title else ""
        text  = body.get_text(separator="\n", strip=True) if body else ""
        text  = re.sub(r"\n{3,}", "\n\n", text)  # 折叠多余空行

        return {
            "text":        text,
            "doc_id":      doc_id or _url_id(url),
            "source_type": "url",
            "title":       title,
            "source":      url,
        }

    # ------------------------------------------------------------------
    # PDF
    # ------------------------------------------------------------------

    def load_pdf(self, path: str, doc_id: str = None) -> dict:
        """
        提取 PDF 文字层。逐页提取，页间用换行分隔。

        Args:
            path   : PDF 文件路径
            doc_id : 自定义 ID，默认用文件名（不含扩展名）

        Returns:
            统一文档 dict
        """
        try:
            import pdfplumber
        except ImportError:
            raise ImportError("请安装依赖: pip install pdfplumber")

        path = Path(path)
        pages = []
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                if page_text.strip():
                    pages.append(page_text)

        text = "\n\n".join(pages)
        text = text.replace("\x00", "")  # strip NUL bytes that PostgreSQL rejects

        return {
            "text":        text,
            "doc_id":      doc_id or path.stem,
            "source_type": "pdf",
            "title":       path.stem,
            "source":      str(path),
        }

    # ------------------------------------------------------------------
    # Markdown / plain text
    # ------------------------------------------------------------------

    def load_file(self, path: str, doc_id: str = None) -> dict:
        """
        读取 Markdown 或纯文本文件。

        Args:
            path   : 文件路径
            doc_id : 自定义 ID，默认用文件名（不含扩展名）

        Returns:
            统一文档 dict
        """
        path = Path(path)
        text = path.read_text(encoding="utf-8")
        suffix = path.suffix.lower()
        source_type = "markdown" if suffix in (".md", ".markdown") else "text"

        return {
            "text":        text,
            "doc_id":      doc_id or path.stem,
            "source_type": source_type,
            "title":       path.stem,
            "source":      str(path),
        }


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _url_id(url: str) -> str:
    """用 URL 的 MD5 前 8 位生成唯一 doc_id。"""
    return "web_" + hashlib.md5(url.encode()).hexdigest()[:8]
