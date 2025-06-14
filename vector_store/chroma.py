"""ChromaDB implementation of VectorStore interface."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Iterable, List, Tuple

try:
    import chromadb
    from chromadb.config import Settings
except ImportError:  # Lightweight stub for test environments without chromadb

    class _DummyCollection:
        def __init__(self):
            self._docs = []
            self._embs = []
            self._metas = []

        def add(self, embeddings, documents, metadatas, ids):
            self._embs.extend(embeddings)
            self._docs.extend(documents)
            self._metas.extend(metadatas)

        def query(self, query_embeddings, n_results=10, where=None):
            return {
                "documents": [self._docs[:n_results]],
                "metadatas": [self._metas[:n_results]],
            }

        def count(self):
            return len(self._docs)

    class _DummyClient:
        def __init__(self, *_, **__):
            self._collections = {}

        def get_or_create_collection(self, name, metadata=None):
            return self._collections.setdefault(name, _DummyCollection())

    class chromadb:  # type: ignore
        PersistentClient = _DummyClient

    class Settings:  # type: ignore
        def __init__(self, **kwargs):
            pass


from .base import VectorStore


class ChromaVectorStore(VectorStore):
    def __init__(self, db_path: Path | str | None = None) -> None:
        path = Path(db_path) if db_path else Path(".chroma_db")
        path.mkdir(exist_ok=True)
        client = chromadb.PersistentClient(
            path=str(path), settings=Settings(allow_reset=True)
        )
        self._code = client.get_or_create_collection(
            name="code_collection", metadata={"hnsw:space": "cosine"}
        )
        self._text = client.get_or_create_collection(
            name="text_collection", metadata={"hnsw:space": "cosine"}
        )

    def _split(
        self,
        chunks: Iterable[Tuple[str, dict]],
        embeddings: List[List[float]],
    ) -> tuple[list, list, list, list, list, list, list, list]:
        code_embs: list = []
        code_docs: list = []
        code_meta: list = []
        code_ids: list = []
        text_embs: list = []
        text_docs: list = []
        text_meta: list = []
        text_ids: list = []
        for (path, chunk), emb in zip(chunks, embeddings, strict=False):
            md = chunk["metadata"]
            md["path"] = str(path)
            md["mcp_id"] = str(uuid.uuid4())
            doc = chunk["text"]
            model = md.get("model", "codebert-base")
            if model == "codebert-base":
                code_embs.append(emb)
                code_docs.append(doc)
                code_meta.append(md)
                code_ids.append(md["mcp_id"])
            else:
                text_embs.append(emb)
                text_docs.append(doc)
                text_meta.append(md)
                text_ids.append(md["mcp_id"])
        return (
            code_embs,
            code_docs,
            code_meta,
            code_ids,
            text_embs,
            text_docs,
            text_meta,
            text_ids,
        )

    def add(
        self,
        chunks: Iterable[Tuple[str, dict]],
        embeddings: List[List[float]],
        batch_size: int = 100,
    ) -> None:
        (
            c_embs,
            c_docs,
            c_meta,
            c_ids,
            t_embs,
            t_docs,
            t_meta,
            t_ids,
        ) = self._split(chunks, embeddings)
        for i in range(0, len(c_embs), batch_size):
            self._code.add(
                embeddings=c_embs[i : i + batch_size],
                documents=c_docs[i : i + batch_size],
                metadatas=c_meta[i : i + batch_size],
                ids=c_ids[i : i + batch_size],
            )
        for i in range(0, len(t_embs), batch_size):
            self._text.add(
                embeddings=t_embs[i : i + batch_size],
                documents=t_docs[i : i + batch_size],
                metadatas=t_meta[i : i + batch_size],
                ids=t_ids[i : i + batch_size],
            )

    def query(self, embedding: List[float], k: int = 10, where: dict | None = None):
        code_res = (
            self._code.query(query_embeddings=[embedding], n_results=k, where=where)
            if self._code.count() > 0
            else {"documents": [[]], "metadatas": [[]]}
        )
        text_res = (
            self._text.query(query_embeddings=[embedding], n_results=k, where=where)
            if self._text.count() > 0
            else {"documents": [[]], "metadatas": [[]]}
        )
        docs = code_res["documents"][0] + text_res["documents"][0]
        metas = code_res["metadatas"][0] + text_res["metadatas"][0]
        return docs, metas

    def count(self) -> int:
        return self._code.count() + self._text.count()
