"""
late_chunking.py
================
Late Chunking embedder for BGE-M3.

原理：对全文做一次 forward pass，获取 token 级别的隐状态，
然后按 chunk 的字符边界对对应 token 做 mean-pool + normalize，
使每个 chunk 的向量包含完整文档上下文。

与标准分块 embedding 的区别：
  标准：每个 chunk 独立 encode → 只有 chunk 内部语境
  Late Chunking：全文 encode → 按边界池化 → 每个 chunk 含全文语境

Usage:
    from src.late_chunking import LateChunkingEmbedder
    embedder = LateChunkingEmbedder()
    embeddings = embedder.embed(doc_text, chunks)  # chunks 需含 start_char/end_char
"""

import logging
from pathlib import Path

import torch
import torch.nn.functional as F

from src.config import config

_logger = logging.getLogger(__name__)
_vs_cfg = config["vector_store"]
_ROOT = Path(__file__).parent.parent

# BGE-M3 最大 token 数
_MAX_DOC_TOKENS = 8192


class LateChunkingEmbedder:
    """
    BGE-M3 Late Chunking embedder。

    懒加载 tokenizer + model，首次调用 embed() 时初始化。
    同一进程内复用，避免重复加载。
    """

    def __init__(self):
        self._tokenizer = None
        self._model = None

    def _load(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoModel, AutoTokenizer

        model_path = _vs_cfg.get("embedding_model_path") or _vs_cfg["embedding_model"]
        _logger.info("Loading BGE-M3 for late chunking from %s", model_path)

        self._tokenizer = AutoTokenizer.from_pretrained(model_path)
        self._model = AutoModel.from_pretrained(model_path, torch_dtype=torch.float16)
        self._model.eval()
        if torch.cuda.is_available():
            self._model.cuda()

    def _embedding_server_url(self) -> str:
        import os
        return os.environ.get("EMBEDDING_SERVER_URL", "") or _vs_cfg.get("embedding_server_url", "")

    def _use_remote(self) -> bool:
        return bool(self._embedding_server_url().strip())

    def _remote_embed(self, texts: list[str]) -> list[list[float]]:
        import requests
        url = self._embedding_server_url().rstrip("/") + "/embed"
        resp = requests.post(url, json={"texts": texts}, timeout=120)
        resp.raise_for_status()
        return resp.json()["embeddings"]

    def embed(self, text: str, chunks: list[dict]) -> list[list[float]]:
        """
        对一篇文档做 late chunking embedding。

        Args:
            text:   完整文档原文（不含 header）
            chunks: ChunkManager.split() 的输出，每项需含 start_char / end_char

        Returns:
            list[list[float]]，长度与 chunks 相同，每项为 1024 维向量。
            若某 chunk 超出截断范围，回退到对该 chunk 文本单独 encode。
        """
        # 远程模式：直接发 chunk 文本，不做 token 级池化（退化为标准 embedding）
        if self._use_remote():
            texts = [chunk["text"] for chunk in chunks]
            return self._remote_embed(texts)

        self._load()

        encoding = self._tokenizer(
            text,
            return_tensors="pt",
            return_offsets_mapping=True,
            truncation=True,
            max_length=_MAX_DOC_TOKENS,
            padding=False,
        )
        offset_mapping = encoding.pop("offset_mapping")[0]  # [seq_len, 2]

        device = next(self._model.parameters()).device
        encoding = {k: v.to(device) for k, v in encoding.items()}

        with torch.no_grad():
            outputs = self._model(**encoding)
            token_embeddings = outputs.last_hidden_state[0]  # [seq_len, 1024]

        # 截断后文档覆盖的最大字符位置
        valid_char_end = int(offset_mapping[-1][1])

        results: list[list[float]] = []
        fallback_texts: list[tuple[int, str]] = []  # (chunk_idx, text) 超出截断范围的 chunk

        for idx, chunk in enumerate(chunks):
            c_start = chunk["start_char"]
            c_end = chunk["end_char"]

            if c_start >= valid_char_end:
                # chunk 完全落在截断范围外，稍后单独 encode
                fallback_texts.append((idx, chunk["text"]))
                results.append(None)
                continue

            mask = self._token_mask(offset_mapping, c_start, c_end)
            selected = token_embeddings[mask]

            if selected.shape[0] == 0:
                # 无对应 token（如纯空白 chunk），用 CLS 向量兜底
                emb = token_embeddings[0]
            else:
                emb = selected.mean(dim=0)

            emb = F.normalize(emb, dim=0)
            results[idx] = emb.cpu().float().tolist()

        # 对超出截断范围的 chunk 单独 encode（退化为标准 embedding）
        if fallback_texts:
            _logger.warning(
                "Document truncated at %d chars; %d chunk(s) will be encoded individually.",
                valid_char_end,
                len(fallback_texts),
            )
            fb_embeddings = self._encode_texts([t for _, t in fallback_texts])
            for (idx, _), emb in zip(fallback_texts, fb_embeddings):
                results[idx] = emb

        return results

    @staticmethod
    def _token_mask(offset_mapping: torch.Tensor, c_start: int, c_end: int) -> torch.Tensor:
        """返回属于 [c_start, c_end) 字符范围的 token 的布尔 mask（排除特殊 token）。"""
        tok_starts = offset_mapping[:, 0]
        tok_ends = offset_mapping[:, 1]
        # 特殊 token 的 offset 为 (0, 0)，通过 tok_ends > 0 排除
        return (tok_starts >= c_start) & (tok_ends <= c_end) & (tok_ends > 0)

    def _encode_texts(self, texts: list[str]) -> list[list[float]]:
        """单独 encode 一批短文本，用于 fallback。返回归一化向量列表。"""
        encoding = self._tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            max_length=_MAX_DOC_TOKENS,
            padding=True,
        )
        device = next(self._model.parameters()).device
        encoding = {k: v.to(device) for k, v in encoding.items()}

        with torch.no_grad():
            outputs = self._model(**encoding)
            # CLS pooling
            cls_embeddings = outputs.last_hidden_state[:, 0, :]
            cls_embeddings = F.normalize(cls_embeddings, dim=1)

        return cls_embeddings.cpu().float().tolist()

    def offload(self) -> None:
        """释放 GPU 显存。"""
        if self._model is not None:
            del self._model
            self._model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
