"""Deprecated compatibility shim – use `vector_store.chroma` instead."""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Iterable, List, Tuple

from vector_store.chroma import ChromaVectorStore

warnings.warn(
    "Importing 'vector_store' is deprecated; use 'vector_store.chroma' instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Singleton backing store
_store = ChromaVectorStore()

# re-export collections for callers that access them directly
code_collection = _store._code  # type: ignore[attr-defined]
text_collection = _store._text  # type: ignore[attr-defined]

# ---------------------------------------------------------------------------
# Legacy function wrappers ---------------------------------------------------
# ---------------------------------------------------------------------------

def add_chunks(
    chunks: Iterable[Tuple[Path, dict]],
    embeddings: List[List[float]],
    batch_size: int = 100,
) -> None:
    _store.add(chunks, embeddings, batch_size=batch_size)


def query(query_text: str, embed_fn, k: int = 10, where: dict | None = None):
    query_emb = embed_fn([query_text])[0]
    return _store.query(query_emb, k=k, where=where)


