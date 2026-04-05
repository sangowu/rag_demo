"""
chunk_manager.py
================
Split a Markdown document into overlapping chunks.

Usage:
    from src.chunk_manager import ChunkManager
    cm = ChunkManager(chunk_size=512, overlap=64, strategy="recursive")
    chunks = cm.split(text, doc_id="ADI_2009_page_49.pdf")
"""

from typing import Any, Literal, Optional

from src.config import config

_cfg = config["chunking"]

def _estimate_tokens(text: str) -> int:
    """Rough token count estimate: words * 1.3"""
    return int(len(text.split()) * 1.3)


def _extract_blocks(text: str) -> list[str]:
    """
    Split document text into blocks, keeping Markdown tables intact.
    A table block = consecutive lines starting with '|'.
    Returns a list of text blocks (paragraphs or whole tables).
    """
    blocks = []
    current_lines = []
    table_lines = []
    
    for line in text.splitlines():
        if line.startswith('|'):
            if current_lines:
                blocks.append('\n'.join(current_lines))
                current_lines = []
            table_lines.append(line)
        else:
            if table_lines:
                blocks.append('\n'.join(table_lines))
                table_lines = []
            current_lines.append(line)
    
    if table_lines:
        blocks.append('\n'.join(table_lines))
    if current_lines:
        blocks.append('\n'.join(current_lines))

    return blocks

class ChunkManager:
    def __init__(
        self,
        chunk_size: Optional[int] = None,
        overlap: Optional[int] = None,
        strategy: Optional[Literal["fixed", "recursive", "semantic"]] = None,
        semantic_embeddings: Optional[Any] = None,
    ):
        """
        Args:
            chunk_size: Token window size for fixed/recursive strategies.
            overlap:  Overlap tokens for fixed/recursive strategies.
            strategy: Splitting strategy name from config or override.
            semantic_embeddings: Optional LangChain Embeddings (e.g. from
                VectorStore.langchain_dense_embeddings()). When set and strategy
                is semantic, reuses the same BGE-M3 as ingestion instead of
                loading HuggingFaceEmbeddings separately.
        """
        self.chunk_size = chunk_size if chunk_size is not None else _cfg["chunk_size"]
        self.overlap = overlap if overlap is not None else _cfg["overlap"]
        self.strategy = strategy if strategy is not None else _cfg["strategy"]
        self._injected_semantic_embeddings = semantic_embeddings
        self._lazy_hf_semantic_embeddings = None

    def split(self, text: str, doc_id: str) -> list[dict]:
        """
        Split document text into chunks.

        Args:
            text:   Full document text (Markdown).
            doc_id: Source document identifier.

        Returns:
            List of dicts with keys: text, doc_id, chunk_index, start_char, end_char.
        """
        if self.strategy == "fixed":
            raw_chunks = self._split_fixed(text)
        elif self.strategy == "recursive":
            raw_chunks = self._split_recursive(text)
        elif self.strategy == "semantic":
            raw_chunks = self._split_semantic(text)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy!r}")    

        chunks = []
        last_pos = 0

        for i, chunk_text in enumerate(raw_chunks):
            start_char = text.find(chunk_text, last_pos)
            if start_char == -1:
                start_char = last_pos
            end_char = start_char + len(chunk_text)
            chunks.append({
                "text": chunk_text,
                "doc_id": doc_id,
                "chunk_index": i,
                "start_char": start_char,
                "end_char": end_char,
            })
            last_pos = start_char

        return chunks
        
    def _split_fixed(self, text: str) -> list[str]:
        """
        Fixed-size split: slide a window of chunk_size tokens with overlap.
        Respect table blocks — never cut inside one.
        """
        from langchain_text_splitters import TokenTextSplitter
        blocks = _extract_blocks(text)
        result = []
        splitter = TokenTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.overlap
        )

        for block in blocks:
            if block.strip().startswith('|'):
                # 表格 block 整体保留
                result.append(block)
            else:
                # 普通文本 block 传给 TokenTextSplitter 切分
                result.extend(splitter.split_text(block))

        return result

    def _split_recursive(self, text: str) -> list[str]:
        """
        Recursive split: try splitting on \\n\\n, then \\n, then space.
        Respect table blocks — never cut inside one.
        Uses tiktoken for token-based chunk_size (same unit as fixed strategy).
        """
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        blocks = _extract_blocks(text)
        result = []
        # from_tiktoken_encoder 使 chunk_size 单位为 token，与 fixed 策略一致
        splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=self.chunk_size,
            chunk_overlap=self.overlap,
            separators=["\n\n", "\n", " "]
        )

        for block in blocks:
            if block.strip().startswith('|'):
                result.append(block)
            else:
                # 普通文本 block 传给 RecursiveCharacterTextSplitter 切分
                result.extend(splitter.split_text(block))

        return result

    def _split_semantic(self, text: str) -> list[str]:
        """
        Semantic split: use a language model to identify natural boundaries.
        Respect table blocks — never cut inside one.
        """
        from langchain_experimental.text_splitter import SemanticChunker
        from langchain_community.embeddings import HuggingFaceEmbeddings

        blocks = _extract_blocks(text)
        result = []

        if self._injected_semantic_embeddings is not None:
            emb = self._injected_semantic_embeddings
        else:
            if self._lazy_hf_semantic_embeddings is None:
                self._lazy_hf_semantic_embeddings = HuggingFaceEmbeddings(
                    model_name="BAAI/bge-m3"
                )
            emb = self._lazy_hf_semantic_embeddings
        # breakpoint_threshold_amount=95: 只在相似度最低的 5% 处断句（保守切分）
        # min_chunk_size: 防止金融短句被切成碎片（单位：字符，~100 tokens）
        splitter = SemanticChunker(
            embeddings=emb,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=95,
            min_chunk_size=500,
        )

        for block in blocks:
            if block.strip().startswith('|'):
                result.append(block)
            else:
                # 普通文本 block 传给 SemanticChunker 切分
                result.extend(splitter.split_text(block))

        return result
