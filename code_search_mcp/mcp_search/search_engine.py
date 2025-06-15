import logging
import sqlite3
from pathlib import Path

from tqdm import tqdm

from code_search_mcp.chunker import scan_project
from code_search_mcp.embedder import embed
from code_search_mcp.token_counter import count_tokens
from code_search_mcp.vector_store.chroma import ChromaVectorStore

_STORE = ChromaVectorStore()

logger = logging.getLogger(__name__)

DB_PATH = Path(".embed_cache/file_timestamps.db")


def init_timestamp_db():
    DB_PATH.parent.mkdir(exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS file_timestamps (
                path TEXT PRIMARY KEY,
                mtime REAL,
                indexed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)


def get_file_mtime(file_path):
    return Path(file_path).stat().st_mtime


def get_cached_mtime(file_path):
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute(
            "SELECT mtime FROM file_timestamps WHERE path = ?", (str(file_path),)
        )
        row = cur.fetchone()
        return row[0] if row else None


def update_cached_mtime(file_path, mtime):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO file_timestamps (path, mtime) VALUES (?, ?)",
            (str(file_path), mtime),
        )


def index_project(project_path, progress_callback=None):
    init_timestamp_db()
    all_chunks = []

    # Scan files with progress
    file_iterator = tqdm(scan_project(project_path), desc="Scanning files", leave=False)

    for file, chunks in file_iterator:
        for chunk in chunks:
            chunk["metadata"]["path"] = str(file)
            all_chunks.append((file, chunk))
        update_cached_mtime(file, get_file_mtime(file))
        if progress_callback:
            progress_callback()

    # Embed chunks with progress
    if all_chunks:
        chunk_iterator = tqdm(all_chunks, desc="Embedding chunks", leave=False)
        embeddings = embed(chunk_iterator)
        _STORE.add(all_chunks, embeddings)
        logger.info(f"Indexed {len(all_chunks)} code chunks.")
    else:
        logger.info("No chunks to index.")


def index_project_incremental(project_path, progress_callback=None):
    init_timestamp_db()
    all_chunks = []

    # Scan files with progress
    file_iterator = tqdm(scan_project(project_path), desc="Scanning files", leave=False)

    for file, chunks in file_iterator:
        current_mtime = get_file_mtime(file)
        cached_mtime = get_cached_mtime(file)
        if cached_mtime is None or current_mtime > cached_mtime:
            for chunk in chunks:
                chunk["metadata"]["path"] = str(file)
                all_chunks.append((file, chunk))
            update_cached_mtime(file, current_mtime)
        if progress_callback:
            progress_callback()

    # Embed chunks with progress
    if all_chunks:
        chunk_iterator = tqdm(all_chunks, desc="Embedding chunks", leave=False)
        embeddings = embed(chunk_iterator)
        _STORE.add(all_chunks, embeddings)
        logger.info(f"Incrementally indexed {len(all_chunks)} code chunks.")
    else:
        logger.info("No changes detected.")


def context_aggregator(chunks, metadatas, max_tokens=8000):
    summary = []
    total_tokens = 0

    for chunk, meta in zip(chunks, metadatas, strict=False):
        chunk_tokens = count_tokens(chunk)
        if total_tokens + chunk_tokens > max_tokens:
            break
        path = meta.get("path", "unknown")
        code_type = meta.get("type", "unknown")
        start_line = meta.get("start_line", 1)
        end_line = meta.get("end_line", 1)
        name = meta.get("name", "")
        content = (
            f"File: {path}, Type: {code_type}, Name: {name}, "
            f"Lines: {start_line}-{end_line}\n{chunk}\n"
        )
        summary.append(content)
        total_tokens += chunk_tokens

    return "\n".join(summary), total_tokens


def stream_code_chunks(query_text, k=5, max_tokens=8000, metadata_filter=None):
    """Yield (chunk, metadata) pairs incrementally until token budget is exhausted."""
    query_text = " ".join(query_text.lower().split())
    where_clause = metadata_filter if metadata_filter else None
    query_emb = embed([(None, {"text": query_text, "metadata": {"path": "<query>"}})])[
        0
    ]
    documents, metadatas = _STORE.query(query_emb, k=k, where=where_clause)

    total_tokens = 0
    for chunk, meta in zip(documents, metadatas, strict=False):
        tokens = count_tokens(chunk)
        if total_tokens + tokens > max_tokens:
            break
        total_tokens += tokens
        yield chunk, meta, total_tokens


def search_code_hybrid(query_text, k=5, max_tokens=8000, metadata_filter=None):
    query_text = " ".join(query_text.lower().split())
    where_clause = metadata_filter if metadata_filter else None
    # Embed query once and query store
    query_emb = embed([(None, {"text": query_text, "metadata": {"path": "<query>"}})])[
        0
    ]

    documents, metadatas = _STORE.query(query_emb, k=k, where=where_clause)

    if not documents:
        return {"content": "", "metadata": [], "tokens": 0}

    # Keep original order (Chroma already sorts by distance)
    # NOTE: ranking may need to be performed for other vector stores
    context, tokens = context_aggregator(documents, metadatas, max_tokens)

    return {
        "content": context,
        "metadata": [
            {
                "path": m.get("path", "unknown"),
                "type": m.get("type", "unknown"),
                "name": m.get("name", ""),
                "start_line": m.get("start_line", 1),
                "end_line": m.get("end_line", 1),
            }
            for m in metadatas
        ],
        "tokens": tokens,
    }
