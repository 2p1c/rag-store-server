from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np

from rag_store.config import CANDIDATES, ENCODE_BATCH_SIZE, MAX_CHARS, MIN_SCORE, MODEL_ID
from rag_store.document_store import (
    OFFSETS_FILE,
    DocumentStore,
    write_docs_jsonl,
    write_offsets,
)
from rag_store.translate import translate_query
from rag_store.vector_index import VectorIndex

EncodeFn = Callable[[list[str]], np.ndarray]

INDEX_FILE = "index.faiss"
DOCS_FILE = "docs.jsonl"
INFO_FILE = "info.json"


def pack_results(
    hits: list[tuple[str, str, float]],
    min_score: float = MIN_SCORE,
    max_chars: int = MAX_CHARS,
) -> list[dict]:
    results: list[dict] = []
    used = 0
    for text, source, score in hits:
        if not text.strip() or score < min_score:
            continue
        n = len(text)
        if results and used + n > max_chars:
            break
        results.append({"text": text, "source": source, "score": float(score)})
        used += n
    return results


def format_bytes(n: int) -> str:
    size = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1024
    return f"{n} B"


def make_encoder(model_id: str = MODEL_ID) -> EncodeFn:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_id)

    def encode(texts: list[str]) -> np.ndarray:
        return model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            batch_size=ENCODE_BATCH_SIZE,
            show_progress_bar=len(texts) > 32,
        )

    return encode


class VectorStore:
    def __init__(
        self,
        index: VectorIndex,
        doc_store: DocumentStore,
        encode_fn: EncodeFn,
        index_dir: Path | None = None,
        model_id: str = MODEL_ID,
        translate_fn: Callable[[str], str] | None = None,
    ) -> None:
        if index.ntotal != len(doc_store):
            raise ValueError("index size does not match documents")
        self.index = index
        self.doc_store = doc_store
        self.encode_fn = encode_fn
        self.index_dir = index_dir
        self.model_id = model_id
        self.translate_fn = translate_fn or translate_query

    @classmethod
    def build(
        cls,
        texts: list[str],
        sources: list[str],
        encode_fn: EncodeFn | None = None,
        model_id: str = MODEL_ID,
        translate_fn: Callable[[str], str] | None = None,
    ) -> VectorStore:
        if encode_fn is None:
            encode_fn = make_encoder(model_id)
        vecs = np.ascontiguousarray(encode_fn(texts), dtype=np.float32)
        index = VectorIndex.flat_ip(vecs.shape[1])
        index.add(vecs)
        return cls(
            index,
            DocumentStore.from_lists(texts, sources),
            encode_fn,
            model_id=model_id,
            translate_fn=translate_fn,
        )

    def save(self, index_dir: str | Path) -> Path:
        if self.doc_store.is_file_backed():
            raise ValueError("save() requires an in-memory document store")
        path = Path(index_dir)
        path.mkdir(parents=True, exist_ok=True)
        self.index.write(path / INDEX_FILE)
        docs = [self.doc_store.get(i) for i in range(len(self.doc_store))]
        offsets = write_docs_jsonl(
            path / DOCS_FILE,
            [d["text"] for d in docs],
            [d["source"] for d in docs],
        )
        write_offsets(path / OFFSETS_FILE, offsets)
        info = {
            "model_id": self.model_id,
            "ntotal": int(self.index.ntotal),
            "dim": int(self.index.d),
            "format": "v2",
            "docs_file": DOCS_FILE,
            "offsets_file": OFFSETS_FILE,
        }
        (path / INFO_FILE).write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
        self.index_dir = path
        return path

    @classmethod
    def load(
        cls,
        index_dir: str | Path,
        encode_fn: EncodeFn | None = None,
        model_id: str = MODEL_ID,
    ) -> VectorStore:
        path = Path(index_dir)
        info_path = path / INFO_FILE
        if not info_path.exists():
            raise FileNotFoundError(f"no index at {path}; run python -m rag_store.ingest first")
        info = json.loads(info_path.read_text(encoding="utf-8"))
        stored_model = info.get("model_id")
        if stored_model != model_id:
            raise ValueError(f"index built with {stored_model}, current model is {model_id}")
        fmt = info.get("format", "v1")
        if fmt == "v2":
            docs_file = info.get("docs_file", DOCS_FILE)
            offsets_file = info.get("offsets_file", OFFSETS_FILE)
            offsets_path = path / offsets_file
            if not offsets_path.exists():
                raise FileNotFoundError(
                    f"missing {offsets_path}; run python -m rag_store.migrate_offsets {path}"
                )
            doc_store = DocumentStore.from_files(path / docs_file, offsets_path)
        else:
            texts: list[str] = []
            sources: list[str] = []
            with (path / DOCS_FILE).open(encoding="utf-8") as f:
                for line in f:
                    row = json.loads(line)
                    texts.append(row["text"])
                    sources.append(row["source"])
            doc_store = DocumentStore.from_lists(texts, sources)
        index = VectorIndex.read(path / INDEX_FILE)
        if encode_fn is None:
            encode_fn = make_encoder(model_id)
        return cls(index, doc_store, encode_fn, index_dir=path, model_id=model_id)

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.ascontiguousarray(self.encode_fn(texts), dtype=np.float32)

    def search(self, query: str, k: int = CANDIDATES) -> list[dict]:
        hits = self.top_k(self.translate_fn(query), k)
        return pack_results(hits)

    def top_k(self, query: str, k: int = CANDIDATES) -> list[tuple[str, str, float]]:
        vec = self.encode([query])
        hits: list[tuple[str, str, float]] = []
        for idx, score in self.index.search(vec, k):
            doc = self.doc_store.get(idx)
            hits.append((doc["text"], doc["source"], score))
        return hits

    def size_info(self) -> dict:
        disk_bytes = 0
        if self.index_dir is not None and self.index_dir.exists():
            disk_bytes = sum(p.stat().st_size for p in self.index_dir.iterdir() if p.is_file())
        return {
            "ntotal": int(self.index.ntotal),
            "dim": int(self.index.d),
            "disk_bytes": disk_bytes,
        }

    def format_size(self) -> str:
        info = self.size_info()
        lines = [
            f"vectors: {info['ntotal']}",
            f"dim: {info['dim']}",
        ]
        if info["disk_bytes"]:
            lines.append(f"disk: {format_bytes(info['disk_bytes'])}")
        return "\n".join(lines)
