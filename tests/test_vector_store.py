import tempfile
from pathlib import Path

import pytest

try:
    import chromadb  # noqa: F401

    from code_search_mcp.vector_store.base import (  # Updated for new structure
        VectorStore,
    )
    from code_search_mcp.vector_store.chroma import (  # Example based on actual imports
        ChromaVectorStore,
    )
except ImportError:  # pragma: no cover
    ChromaVectorStore = None  # type: ignore


@pytest.mark.skipif(ChromaVectorStore is None, reason="chromadb not available")
def test_chroma_round_trip():
    with tempfile.TemporaryDirectory() as tmp:
        store = ChromaVectorStore(db_path=tmp)
        chunk = {"text": "hello world", "metadata": {}}
        store.add([(Path("foo.py"), chunk)], embeddings=[[0.1, 0.2, 0.3]])
        assert store.count() == 1
        docs, metas = store.query([0.1, 0.2, 0.3], k=1)
        assert docs and metas
        assert docs[0] == "hello world"
