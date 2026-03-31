"""
Unit tests for scripts/generate_qa.py sampling and chunk helpers (no LLM calls).
"""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_SPEC = importlib.util.spec_from_file_location(
    "generate_qa", _ROOT / "scripts" / "generate_qa.py",
)
assert _SPEC and _SPEC.loader
_gq = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_gq)


class TestStratifyKey(unittest.TestCase):
    def test_ticker_prefix(self) -> None:
        p = Path("/tmp/AAL_2013_page_15.pdf.md")
        self.assertEqual(_gq.stratify_key(p), "AAL")

    def test_no_underscore_uses_stem(self) -> None:
        p = Path("/tmp/single.md")
        self.assertEqual(_gq.stratify_key(p), "single")


class TestStableStrategyBucket(unittest.TestCase):
    def test_deterministic(self) -> None:
        self.assertEqual(_gq.stable_strategy_bucket("doc_a"), _gq.stable_strategy_bucket("doc_a"))
        self.assertTrue(0 <= _gq.stable_strategy_bucket("any_id") < _gq.NUM_STRATEGIES)

    def test_empty_input_bounded(self) -> None:
        v = _gq.stable_strategy_bucket("")
        self.assertTrue(0 <= v < _gq.NUM_STRATEGIES)


class TestPickChunk(unittest.TestCase):
    def test_short_doc_returns_fallback(self) -> None:
        text = "x" * 50
        chunk, label = _gq.pick_chunk(text, "id1")
        self.assertLessEqual(len(chunk), _gq.MAX_CHUNK_CHARS)
        self.assertIn(label, _gq.STRATEGY_LABELS)

    def test_extreme_empty(self) -> None:
        chunk, _ = _gq.pick_chunk("", "stable-id")
        self.assertEqual(chunk, "")


class TestSampleDocPaths(unittest.TestCase):
    def test_sequential_order(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            tmp_path = Path(d)
            (tmp_path / "b.md").write_text("b")
            (tmp_path / "a.md").write_text("a")
            paths = list(tmp_path.glob("*.md"))
            out = _gq.sample_doc_paths(paths, n=10, seed=0, mode="sequential")
            self.assertEqual([p.name for p in out], ["a.md", "b.md"])

    def test_stratified_round_robin(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            tmp_path = Path(d)
            for name in ("A_1.md", "A_2.md", "B_1.md", "B_2.md"):
                (tmp_path / name).write_text("x")
            paths = list(tmp_path.glob("*.md"))
            out = _gq.sample_doc_paths(paths, n=4, seed=99, mode="stratified")
            names = [p.name for p in out]
            self.assertEqual(len(names), 4)
            self.assertEqual(set(names), {"A_1.md", "A_2.md", "B_1.md", "B_2.md"})
            self.assertNotEqual(names[0][0], names[1][0])

    def test_invalid_n_returns_empty(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            tmp_path = Path(d)
            (tmp_path / "z.md").write_text("z")
            self.assertEqual(
                _gq.sample_doc_paths(list(tmp_path.glob("*.md")), n=0, seed=1, mode="sequential"),
                [],
            )

    def test_empty_paths(self) -> None:
        self.assertEqual(_gq.sample_doc_paths([], n=5, seed=1, mode="stratified"), [])


if __name__ == "__main__":
    unittest.main()
