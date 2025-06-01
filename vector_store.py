import uuid
from pathlib import Path

import chromadb
from chromadb.config import Settings

DB_PATH = Path("./.chroma_db")
DB_PATH.mkdir(exist_ok=True)

client = chromadb.PersistentClient(
    path=str(DB_PATH), settings=Settings(allow_reset=True)
)

# Initialize two collections for different embedding models
code_collection = client.get_or_create_collection(
    name="code_collection", metadata={"hnsw:space": "cosine"}
)
text_collection = client.get_or_create_collection(
    name="text_collection", metadata={"hnsw:space": "cosine"}
)


def add_chunks(chunks, embeddings, batch_size=100):
    code_embs = []
    code_docs = []
    code_metas = []
    code_ids = []

    text_embs = []
    text_docs = []
    text_metas = []
    text_ids = []

    for (path, chunk), emb in zip(chunks, embeddings, strict=False):
        metadata = chunk["metadata"]
        metadata["path"] = str(path)
        metadata["mcp_id"] = str(uuid.uuid4())
        metadata["source"] = "codebase"

        doc = chunk["text"]
        model = metadata.get("model", "codebert-base")  # Default to codebert

        if model == "codebert-base":
            code_embs.append(emb)
            code_docs.append(doc)
            code_metas.append(metadata)
            code_ids.append(metadata["mcp_id"])
        else:
            text_embs.append(emb)
            text_docs.append(doc)
            text_metas.append(metadata)
            text_ids.append(metadata["mcp_id"])

    # Add to code collection
    if code_embs:
        for i in range(0, len(code_embs), batch_size):
            code_collection.add(
                embeddings=code_embs[i : i + batch_size],
                documents=code_docs[i : i + batch_size],
                metadatas=code_metas[i : i + batch_size],
                ids=code_ids[i : i + batch_size],
            )

    # Add to text collection
    if text_embs:
        for i in range(0, len(code_embs), batch_size):
            text_collection.add(
                embeddings=text_embs[i : i + batch_size],
                documents=text_docs[i : i + batch_size],
                metadatas=text_metas[i : i + batch_size],
                ids=text_ids[i : i + batch_size],
            )


def query(query_text, embed_fn, k=10, where=None):
    query_emb = embed_fn([query_text])[0]

    # Query both collections
    code_results = (
        code_collection.query(query_embeddings=[query_emb], n_results=k, where=where)
        if code_collection.count() > 0
        else {"documents": [[]], "metadatas": [[]]}
    )

    text_results = (
        text_collection.query(query_embeddings=[query_emb], n_results=k, where=where)
        if text_collection.count() > 0
        else {"documents": [[]], "metadatas": [[]]}
    )

    # Combine results
    documents = code_results["documents"][0] + text_results["documents"][0]
    metadatas = code_results["metadatas"][0] + text_results["metadatas"][0]

    # Sort by distance (if available) and limit to k
    if documents:
        combined = sorted(
            [
                (doc, meta)
                for doc, meta in zip(documents, metadatas, strict=False)
                if doc
            ],
            key=lambda x: x[1].get("distance", float("inf")),
        )[:k]
        documents, metadatas = zip(*combined, strict=False) if combined else ([], [])

    return list(documents), list(metadatas)
