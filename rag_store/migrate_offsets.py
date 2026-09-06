from __future__ import annotations

import json
from pathlib import Path

from rag_store.document_store import OFFSETS_FILE, build_offsets, write_offsets
from rag_store.store import DOCS_FILE, INDEX_FILE, INFO_FILE


def migrate(index_dir: str | Path) -> Path:
    path = Path(index_dir)
    info_path = path / INFO_FILE
    if not info_path.exists():
        raise FileNotFoundError(f"no index at {path}")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    docs_path = path / info.get("docs_file", DOCS_FILE)
    if not docs_path.exists():
        raise FileNotFoundError(f"missing {docs_path}")
    if not (path / INDEX_FILE).exists():
        raise FileNotFoundError(f"missing {path / INDEX_FILE}")

    offsets = build_offsets(docs_path)
    expected = info.get("ntotal")
    if expected is not None and len(offsets) != int(expected):
        raise ValueError(f"offset count {len(offsets)} != ntotal {expected}")

    offsets_name = info.get("offsets_file", OFFSETS_FILE)
    offsets_path = path / offsets_name
    write_offsets(offsets_path, offsets)

    info["format"] = "v2"
    info["docs_file"] = docs_path.name
    info["offsets_file"] = offsets_name
    info_path.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    return offsets_path


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Write docs.offsets and mark an index dir as v2")
    parser.add_argument("index_dir")
    args = parser.parse_args()
    out = migrate(args.index_dir)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
