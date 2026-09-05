from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import faiss
import numpy as np

from rag_store.config import CANDIDATES, ENCODE_BATCH_SIZE, MAX_CHARS, MIN_SCORE, MODEL_ID
from rag_store.translate import translate_query

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
        index: faiss.Index,
        texts: list[str],
        sources: list[str],
        encode_fn: EncodeFn,
        index_dir: Path | None = None,
        model_id: str = MODEL_ID,
        translate_fn: Callable[[str], str] | None = None,
    ) -> None:
        if len(texts) != len(sources):
            raise ValueError("texts and sources length mismatch")
        if index.ntotal != len(texts):
            raise ValueError("index size does not match documents")
        self.index = index
        self.texts = texts
        self.sources = sources
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
        index = faiss.IndexFlatIP(vecs.shape[1])
        index.add(vecs)
        return cls(
            index,
            texts,
            sources,
            encode_fn,
            model_id=model_id,
            translate_fn=translate_fn,
        )

    def save(self, index_dir: str | Path) -> Path:
        path = Path(index_dir)
        path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path / INDEX_FILE))
        with (path / DOCS_FILE).open("w", encoding="utf-8") as f:
            for text, source in zip(self.texts, self.sources):
                f.write(json.dumps({"text": text, "source": source}, ensure_ascii=False) + "\n")
        info = {
            "model_id": self.model_id,
            "ntotal": int(self.index.ntotal),
            "dim": int(self.index.d),
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
        texts: list[str] = []
        sources: list[str] = []
        with (path / DOCS_FILE).open(encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                texts.append(row["text"])
                sources.append(row["source"])
        index = faiss.read_index(str(path / INDEX_FILE))
        if encode_fn is None:
            encode_fn = make_encoder(model_id)
        return cls(index, texts, sources, encode_fn, index_dir=path, model_id=model_id)

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.ascontiguousarray(self.encode_fn(texts), dtype=np.float32)

    def search(self, query: str, k: int = CANDIDATES) -> list[dict]:
        hits = self.top_k(self.translate_fn(query), k)
        return pack_results(hits)

    def top_k(self, query: str, k: int = CANDIDATES) -> list[tuple[str, str, float]]:
        vec = self.encode([query])
        k = min(k, self.index.ntotal)
        scores, ids = self.index.search(vec, k)
        hits: list[tuple[str, str, float]] = []
        for score, idx in zip(scores[0], ids[0]):
            if idx < 0:
                continue
            hits.append((self.texts[idx], self.sources[idx], float(score)))
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
