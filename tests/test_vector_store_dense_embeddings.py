"""Unit tests for VectorStoreDenseLangchainEmbeddings batching."""

import unittest
from unittest.mock import MagicMock

from src.vector_store import VectorStoreDenseLangchainEmbeddings


class TestVectorStoreDenseLangchainEmbeddings(unittest.TestCase):
    """Tests embed_documents batching and embed_query delegation."""

    def test_embed_documents_empty(self) -> None:
        """Empty input returns empty list without calling the store."""
        vs = MagicMock()
        emb = VectorStoreDenseLangchainEmbeddings(vs, batch_size=4)
        self.assertEqual(emb.embed_documents([]), [])
        vs._load_model.assert_not_called()

    def test_embed_documents_batches(self) -> None:
        """Long input is split into batches of the configured size."""
        vs = MagicMock()

        def _fake_embed(batch: list[str]) -> dict:
            return {"dense_vecs": [[float(len(t))] for t in batch]}

        vs._embed_documents.side_effect = _fake_embed
        emb = VectorStoreDenseLangchainEmbeddings(vs, batch_size=2)
        out = emb.embed_documents(["a", "bb", "ccc"])
        self.assertEqual(out, [[1.0], [2.0], [3.0]])
        self.assertEqual(vs._embed_documents.call_count, 2)
        vs._load_model.assert_called()

    def test_embed_query(self) -> None:
        """embed_query forwards to VectorStore._embed_query dense_vec."""
        vs = MagicMock()
        vs._embed_query.return_value = {"dense_vec": [0.1, 0.2]}
        emb = VectorStoreDenseLangchainEmbeddings(vs, batch_size=8)
        self.assertEqual(emb.embed_query("q"), [0.1, 0.2])
        vs._embed_query.assert_called_once_with("q")


if __name__ == "__main__":
    unittest.main()
