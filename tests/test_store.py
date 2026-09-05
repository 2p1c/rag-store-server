import tempfile
import unittest
from pathlib import Path

import numpy as np

from rag_store.store import VectorStore, format_bytes, pack_results


class PackResultsTest(unittest.TestCase):
    def test_drops_low_scores_and_empty_text(self):
        hits = [
            ("keep", "a.md", 0.9),
            ("", "b.md", 0.99),
            ("   ", "c.md", 0.99),
            ("nope", "d.md", 0.2),
        ]
        out = pack_results(hits, min_score=0.35, max_chars=6000)
        self.assertEqual(out, [{"text": "keep", "source": "a.md", "score": 0.9}])

    def test_fills_until_char_budget(self):
        hits = [
            ("aa", "a", 0.9),
            ("bb", "b", 0.8),
            ("ccc", "c", 0.7),
        ]
        out = pack_results(hits, min_score=0.35, max_chars=4)
        self.assertEqual([r["source"] for r in out], ["a", "b"])

    def test_includes_first_hit_even_if_over_budget(self):
        hits = [("toolong", "a", 0.9), ("short", "b", 0.8)]
        out = pack_results(hits, min_score=0.35, max_chars=3)
        self.assertEqual([r["source"] for r in out], ["a"])

    def test_all_below_threshold_is_empty(self):
        self.assertEqual(pack_results([("x", "a", 0.1)], min_score=0.35), [])


class FormatBytesTest(unittest.TestCase):
    def test_units(self):
        self.assertEqual(format_bytes(512), "512 B")
        self.assertEqual(format_bytes(2048), "2.0 KB")


class VectorStoreTest(unittest.TestCase):
    def _store(self, texts: list[str], vectors: np.ndarray) -> VectorStore:
        table = {t: v for t, v in zip(texts, vectors)}

        def encode_fn(batch: list[str]) -> np.ndarray:
            return np.stack([table[t] for t in batch]).astype(np.float32)

        sources = [f"doc:{i}" for i in range(len(texts))]
        return VectorStore.build(texts, sources, encode_fn=encode_fn, model_id="dummy")

    def test_inner_product_after_l2_ranks_cosine(self):
        texts = ["apple", "car"]
        vectors = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        store = self._store(texts, vectors)
        hits = store.top_k("apple", k=2)
        self.assertEqual([h[0] for h in hits], ["apple", "car"])
        self.assertAlmostEqual(hits[0][2], 1.0, places=5)
        self.assertAlmostEqual(hits[1][2], 0.0, places=5)

    def test_search_applies_min_score(self):
        texts = ["apple", "car"]
        vectors = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        store = self._store(texts, vectors)
        results = store.search("apple")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["text"], "apple")
        self.assertEqual(results[0]["source"], "doc:0")

    def test_search_translates_chinese_before_encode(self):
        texts = ["apple", "car"]
        vectors = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        table = {t: v for t, v in zip(texts, vectors)}

        def encode_fn(batch: list[str]) -> np.ndarray:
            return np.stack([table[t] for t in batch]).astype(np.float32)

        store = VectorStore.build(
            texts,
            ["doc:0", "doc:1"],
            encode_fn=encode_fn,
            model_id="dummy",
            translate_fn=lambda q: "apple" if "苹" in q else q,
        )
        results = store.search("苹果")
        self.assertEqual(results[0]["text"], "apple")

    def test_save_and_load_roundtrip(self):
        texts = ["apple", "car"]
        vectors = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        table = {t: v for t, v in zip(texts, vectors)}

        def encode_fn(batch: list[str]) -> np.ndarray:
            return np.stack([table[t] for t in batch]).astype(np.float32)

        store = self._store(texts, vectors)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            store.save(path)
            self.assertGreater(store.size_info()["disk_bytes"], 0)
            loaded = VectorStore.load(path, encode_fn=encode_fn, model_id="dummy")
            self.assertEqual(loaded.size_info()["ntotal"], 2)
            self.assertEqual(loaded.search("apple")[0]["text"], "apple")


if __name__ == "__main__":
    unittest.main()
