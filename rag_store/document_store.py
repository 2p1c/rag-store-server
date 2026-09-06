from __future__ import annotations

import json
from pathlib import Path

import numpy as np

OFFSETS_FILE = "docs.offsets"


class DocumentStore:
    """Look up `{text, source}` by FAISS row id.

    Built from in-memory lists (ingest / v1 load) or jsonl + uint64 offsets (v2).
    File-backed `get()` opens the jsonl per call so concurrent searches cannot
    race on a shared file offset.
    """

    def __init__(
        self,
        *,
        records: list[dict] | None = None,
        docs_path: Path | None = None,
        offsets: np.ndarray | None = None,
    ) -> None:
        if records is not None:
            self._records = records
            self._docs_path: Path | None = None
            self._offsets: np.ndarray | None = None
            return
        if docs_path is None or offsets is None:
            raise ValueError("need records, or docs_path and offsets")
        self._records = None
        self._docs_path = Path(docs_path)
        self._offsets = np.asarray(offsets, dtype=np.uint64)

    @classmethod
    def from_lists(cls, texts: list[str], sources: list[str]) -> DocumentStore:
        if len(texts) != len(sources):
            raise ValueError("texts and sources length mismatch")
        records = [{"text": text, "source": source} for text, source in zip(texts, sources)]
        return cls(records=records)

    @classmethod
    def from_files(cls, docs_path: str | Path, offsets_path: str | Path) -> DocumentStore:
        offsets = np.fromfile(offsets_path, dtype=np.uint64)
        return cls(docs_path=Path(docs_path), offsets=offsets)

    def __len__(self) -> int:
        if self._records is not None:
            return len(self._records)
        assert self._offsets is not None
        return int(self._offsets.shape[0])

    def is_file_backed(self) -> bool:
        return self._records is None

    def get(self, row_id: int) -> dict:
        if self._records is not None:
            return self._records[row_id]
        assert self._docs_path is not None and self._offsets is not None
        offset = int(self._offsets[row_id])
        with self._docs_path.open("rb") as f:
            f.seek(offset)
            line = f.readline()
        row = json.loads(line)
        return {"text": row["text"], "source": row["source"]}


def write_docs_jsonl(path: str | Path, texts: list[str], sources: list[str]) -> np.ndarray:
    if len(texts) != len(sources):
        raise ValueError("texts and sources length mismatch")
    offsets: list[int] = []
    with Path(path).open("wb") as f:
        for text, source in zip(texts, sources):
            offsets.append(f.tell())
            line = json.dumps({"text": text, "source": source}, ensure_ascii=False) + "\n"
            f.write(line.encode("utf-8"))
    return np.asarray(offsets, dtype=np.uint64)


def build_offsets(docs_path: str | Path) -> np.ndarray:
    offsets: list[int] = []
    with Path(docs_path).open("rb") as f:
        while True:
            pos = f.tell()
            line = f.readline()
            if not line:
                break
            offsets.append(pos)
    return np.asarray(offsets, dtype=np.uint64)


def write_offsets(path: str | Path, offsets: np.ndarray) -> None:
    np.asarray(offsets, dtype=np.uint64).tofile(path)
