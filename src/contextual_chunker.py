"""
contextual_chunker.py
=====================
Contextual Retrieval preprocessor (Anthropic, 2024).

For each chunk, calls Gemini to generate a brief situating context:
  "This passage is from [Company] [Year] annual report, covering [topic]."

The embedded text becomes: context_prefix + "\\n\\n" + chunk_text
The stored content (returned to LLM) remains the original chunk_text,
so the LLM sees clean source text, not the indexing prefix.

Usage:
    from src.contextual_chunker import ContextualChunker
    from src.chunk_manager import ChunkManager

    cm = ChunkManager()
    cc = ContextualChunker()
    chunks = cm.split(text, doc_id)
    enriched = cc.enrich(text, doc_id, chunks)
    # each chunk now has embed_text = "<context>\\n\\n<original chunk>"
"""

import logging
from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage

from src.llm_factory import get_llm

_logger = logging.getLogger(__name__)

_SYSTEM = (
    "You are a document indexing assistant. "
    "Given a financial document excerpt and a specific passage from it, "
    "write 1-2 sentences that situate the passage within the document. "
    "Include: the company name, fiscal year, and the financial topic or section. "
    "Output ONLY the context sentences — no preamble, no quotes, no extra text."
)

_HUMAN_TEMPLATE = """\
Document ID: {doc_id}

Document excerpt (first 3000 chars):
{doc_snippet}

Passage to situate:
{chunk_text}

Context:"""


class ContextualChunker:
    def __init__(self, llm=None):
        self._llm = llm or get_llm(temperature=0.0, max_output_tokens=128)

    def _generate_context(self, doc_id: str, doc_snippet: str, chunk_text: str) -> str:
        prompt = _HUMAN_TEMPLATE.format(
            doc_id=doc_id,
            doc_snippet=doc_snippet,
            chunk_text=chunk_text[:800],  # cap chunk preview to save tokens
        )
        try:
            response = self._llm.invoke([
                SystemMessage(_SYSTEM),
                HumanMessage(prompt),
            ])
            content = response.content
            if isinstance(content, list):
                content = " ".join(
                    p.get("text", "") if isinstance(p, dict) else str(p)
                    for p in content
                )
            return content.strip()
        except Exception as e:
            _logger.warning("Context generation failed for %s: %s", doc_id, e)
            return ""

    def enrich(
        self,
        full_text: str,
        doc_id: str,
        chunks: list[dict],
        doc_snippet_chars: int = 3000,
    ) -> list[dict]:
        """
        Enrich chunks with contextual prefix for embedding.

        Args:
            full_text : full document text (used for context generation)
            doc_id    : document identifier
            chunks    : output of ChunkManager.split()

        Returns:
            Same chunks with added keys:
              embed_text : context_prefix + "\\n\\n" + chunk["text"]
              content    : original chunk["text"] (unchanged)
        """
        doc_snippet = full_text[:doc_snippet_chars]
        enriched = []
        for chunk in chunks:
            ctx = self._generate_context(doc_id, doc_snippet, chunk["text"])
            embed_text = (ctx + "\n\n" + chunk["text"]) if ctx else chunk["text"]
            enriched.append({
                **chunk,
                "embed_text": embed_text,
                "content": chunk["text"],
            })
        return enriched
