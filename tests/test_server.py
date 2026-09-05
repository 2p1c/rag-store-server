import unittest

import numpy as np
from fastapi.testclient import TestClient

from rag_store.server import create_app
from rag_store.store import VectorStore


class SearchApiTest(unittest.TestCase):
    def setUp(self) -> None:
        texts = ["apple fruit", "car engine"]
        table = {
            "apple fruit": np.array([1.0, 0.0], dtype=np.float32),
            "car engine": np.array([0.0, 1.0], dtype=np.float32),
            "apple": np.array([1.0, 0.0], dtype=np.float32),
        }

        def encode_fn(batch: list[str]) -> np.ndarray:
            return np.stack([table[t] for t in batch])

        store = VectorStore.build(
            texts,
            ["a.md", "b.md"],
            encode_fn=encode_fn,
            model_id="dummy",
        )
        self._cm = TestClient(create_app(store))
        self.client = self._cm.__enter__()

    def tearDown(self) -> None:
        self._cm.__exit__(None, None, None)

    def test_search_returns_contract_payload(self):
        res = self.client.post("/search", json={"query": "apple"})
        self.assertEqual(res.status_code, 200)
        body = res.json()
        self.assertIn("results", body)
        self.assertEqual(len(body["results"]), 1)
        hit = body["results"][0]
        self.assertEqual(hit["text"], "apple fruit")
        self.assertEqual(hit["source"], "a.md")
        self.assertGreater(hit["score"], 0.9)

    def test_empty_query_is_error(self):
        res = self.client.post("/search", json={"query": "   "})
        self.assertEqual(res.status_code, 400)

    def test_chinese_query_is_translated_then_searched(self):
        texts = ["apple fruit", "car engine"]
        table = {
            "apple fruit": np.array([1.0, 0.0], dtype=np.float32),
            "car engine": np.array([0.0, 1.0], dtype=np.float32),
            "apple": np.array([1.0, 0.0], dtype=np.float32),
        }

        def encode_fn(batch: list[str]) -> np.ndarray:
            return np.stack([table[t] for t in batch])

        store = VectorStore.build(
            texts,
            ["a.md", "b.md"],
            encode_fn=encode_fn,
            model_id="dummy",
            translate_fn=lambda q: "apple" if "苹" in q else q,
        )
        with TestClient(create_app(store)) as client:
            res = client.post("/search", json={"query": "苹果"})
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["results"][0]["text"], "apple fruit")


if __name__ == "__main__":
    unittest.main()
