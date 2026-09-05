from datasets import load_dataset

from rag_store.config import (
    CANDIDATES,
    CORPUS_SIZE,
    DATASET_CONFIG,
    DATASET_ID,
    DATASET_SPLIT,
    INDEX_DIR,
    MODEL_ID,
)
from rag_store.store import VectorStore


DEMO_QUERY = "what is machine learning"


def load_passages(limit: int = CORPUS_SIZE) -> tuple[list[str], list[str]]:
    ds = load_dataset(DATASET_ID, DATASET_CONFIG, split=DATASET_SPLIT, streaming=True)
    texts: list[str] = []
    sources: list[str] = []
    for row in ds:
        text = (row.get("text") or "").strip()
        if not text:
            continue
        texts.append(text)
        sources.append(f"msmarco-passage:{row['pid']}")
        if len(texts) >= limit:
            break
    return texts, sources


def main() -> None:
    print(f"model: {MODEL_ID}")
    print(f"dataset: {DATASET_ID} ({DATASET_CONFIG}, first {CORPUS_SIZE} of {DATASET_SPLIT})")
    texts, sources = load_passages()
    print(f"loaded {len(texts)} passages")
    store = VectorStore.build(texts, sources)
    store.save(INDEX_DIR)
    print("vector store saved to", INDEX_DIR)
    print(store.format_size())
    print()
    print(f"top {CANDIDATES} for: {DEMO_QUERY!r}")
    for i, (text, source, score) in enumerate(store.top_k(DEMO_QUERY, CANDIDATES), 1):
        preview = text if len(text) <= 200 else text[:200] + "..."
        print(f"[{i}] score={score:.4f} source={source}")
        print(preview)
        print("=" * 50)


if __name__ == "__main__":
    main()
