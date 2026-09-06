from __future__ import annotations

from pathlib import Path

import faiss
import numpy as np


class VectorIndex:
    """Thin wrapper: search returns `(row_id, score)` pairs."""

    def __init__(self, index: faiss.Index) -> None:
        self._index = index

    @property
    def ntotal(self) -> int:
        return int(self._index.ntotal)

    @property
    def d(self) -> int:
        return int(self._index.d)

    def add(self, vecs: np.ndarray) -> None:
        self._index.add(vecs)

    def search(self, vec: np.ndarray, k: int) -> list[tuple[int, float]]:
        k = min(k, self.ntotal)
        scores, ids = self._index.search(vec, k)
        hits: list[tuple[int, float]] = []
        for score, idx in zip(scores[0], ids[0]):
            if idx < 0:
                continue
            hits.append((int(idx), float(score)))
        return hits

    def write(self, path: str | Path) -> None:
        faiss.write_index(self._index, str(path))

    @classmethod
    def read(cls, path: str | Path) -> VectorIndex:
        return cls(faiss.read_index(str(path)))

    @classmethod
    def flat_ip(cls, dim: int) -> VectorIndex:
        return cls(faiss.IndexFlatIP(dim))
