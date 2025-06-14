# Code Search MCP – Project Specification

_Last updated: 2025-06-13_

## 1. Purpose & Vision
The **Code Search MCP** project provides semantic, vector-based code search that can be embedded inside every OpenHands runtime container.  By replacing naïve line-by-line file scans with dense-vector retrieval, we greatly improve accuracy, latency and cost when supplying code context to LLM-powered assistants.

The component must remain **self-contained**, require **no external services**, and expose a narrow interface so it can be wired into existing OpenHands workflows now and evolved later without breaking callers.

---

## 2. High-level Goals
|  # | Goal | Status |
|---|------|---------|
| G1 | Index an arbitrary project directory into vector embeddings | **Implemented** (CLI `index`, `reindex`) |
| G2 | Serve top-k code snippets & aggregated context for a natural-language query | **Implemented** (FastAPI endpoints) |
| G3 | Deliver as a lightweight library (`Indexer`, `Searcher`) importable by runtime code | **Planned** |
| G4 | Run entirely on CPU with small (< 50 MB) models, but allow plug-ins for GPU or remote models | **Planned** |
| G5 | Support multiple languages via tree-sitter chunking | **Planned** |
| G6 | Provide incremental watch-mode re-indexing | **Planned** |
| G7 | Maintain a pluggable vector-store layer (FAISS, Chroma, remote) | **Planned** |
| G8 | Track model / index version for compatibility & auto-migration | **Planned** |

---

## 3. Functional Requirements
### 3.1 Indexing
* **F-IDX-1**  Scan project path, respecting `.gitignore` + default ignore patterns.
* **F-IDX-2**  Split source files into *chunks*:
  * Python → classes & functions via `libcst` (implemented)
  * Other languages → tree-sitter parsers where available (planned)
  * Fallback → whole-file chunk
* **F-IDX-3**  Generate embeddings per chunk.
  * Code → `CodeBERT` (default)  
  * Natural-language docs → `MiniLM` (default)
  * Models may be quantised INT8 for size
* **F-IDX-4**  Store vectors + metadata in a pluggable vector store.
* **F-IDX-5**  Cache embeddings on disk (DuckDB/SQLite) keyed by md5(text+model).
* **F-IDX-6**  Support incremental update if file mtime changes.
* **F-IDX-7**  (Planned) Watch-mode: filesystem events trigger re-embed.

### 3.2 Search & Context Retrieval
* **F-SRCH-1**  Accept natural-language query string.
* **F-SRCH-2**  Embed query with same model(s).
* **F-SRCH-3**  Retrieve top-`k` candidate chunks from vector store (default `k=10`).
* **F-SRCH-4**  Re-rank candidates with a small cross-encoder for higher precision.
* **F-SRCH-5**  Aggregate snippets until a configurable token budget is met (default 8 000 tokens).
* **F-SRCH-6**  Return both raw search results and aggregated context.
* **F-SRCH-7**  Optional metadata filters (`type`, `path`, etc.).
* **F-SRCH-8**  (Planned) Batch query endpoint for multiple queries in one call.
* **F-SRCH-9**  (Planned) Relevance-feedback endpoint to collect positive/negative signals.

### 3.3 API Layers
* **Library API** (primary)
  ```python
  from mcp_search import Indexer, Searcher
  indexer = Indexer(project_path)
  indexer.index_full()
  searcher = Searcher(project_path)
  ctx = searcher.context("paginate Django queryset", k=5)
  ```
* **FastAPI Adapter** (optional; current implementation)
  * `POST /mcp/v1/context`
  * `POST /mcp/v1/context/search`
  * `POST /mcp/v1/file`
  * `POST /mcp/v1/reindex`
* **gRPC / Unix-socket JSON** (future): thin transport for intra-container calls.

---

## 4. Non-functional Requirements
* **NFR-1  Footprint** – Entire component ≤ 300 MB disk, ≤ 500 MB RAM at runtime.
* **NFR-2  Latency** – Single-query context retrieval ≤ 250 ms on 4-core CPU for 30 k-chunk index.
* **NFR-3  Portability** – No external databases or message queues required.
* **NFR-4  Deterministic builds** – All models & deps pinned via `requirements.txt` (+ lock file).
* **NFR-5  Observability** – Structured INFO logs for major operations; DEBUG opt-in.

---

## 5. Data Model
```jsonc
{
  "id": "uuid",            // unique per chunk
  "text": "def foo(): …",  // chunk text
  "embedding": [ … ],       // vector stored in backend
  "metadata": {
    "path": "src/foo.py",
    "type": "function|class|file",
    "name": "foo",
    "start_line": 10,
    "end_line": 34,
    "model": "codebert-base",
    "index_version": "2025-06-02" // bump when embedder changes
  }
}
```

---

## 6. Road-map (Condensed)
| Phase | Items |
|-------|-------|
| **0.9** | Library-first refactor; export `Indexer`, `Searcher`; wrap existing FastAPI on top |
| **1.0** | Tree-sitter chunking for JS/TS/Java/Go; INT8 quantised models; pluggable vector store interface |
| **1.1** | Watch-mode incremental indexing; batch query API; semantic deduplication |
| **1.2** | Lightweight relevance-feedback capture & periodic fine-tuning pipeline |

---

## 7. Out-of-scope Items
* Web UI / dashboard
* PR annotation bot
* Kubernetes probes, CI pipelines, heavy security hardening – left to host environment
* Distributed multi-node vector store (may revisit if container scaling proves insufficient)

---

## 8. Glossary
| Term | Meaning |
|------|---------|
| **Chunk** | Smallest indexed unit—function, class or whole file text |
| **Embedding** | Numerical vector representing a chunk or query |
| **Vector Store** | Backend that supports nearest-neighbour search on embeddings |
| **Cross-encoder** | Model that re-scores (query, chunk) pairs for precision |
| **Watch-mode** | Background filesystem watcher that triggers incremental index |

---

## 9. Contributors Guide (brief)
1. Create feature branch `feat/<area>`
2. Run `poetry install && pytest` before PR
3. Adhere to `ruff` style (`ruff --fix .`)
4. Document public functions & update this spec if behaviour changes.

---

## 10. License
This project inherits the MIT license used across OpenHands open-source components unless otherwise noted.
