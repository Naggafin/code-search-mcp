"""Library-first interface for Code Search MCP.

Provides thin `Indexer` and `Searcher` classes that wrap the existing
`search_engine` module so OpenHands runtime containers can import and use
vector search without running the HTTP service.  This keeps the public
surface area small while allowing the underlying implementation to evolve.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Optional

from . import search_engine
from .utils import sse_event

# Re-export version for clients that want to pin behaviour
__version__ = "0.9.0"

__all__ = [
    "Indexer",
    "Searcher",
    "sse_event",
    "__version__",
]


class Indexer:
    """Wraps full and incremental indexing for a project path."""

    def __init__(self, project_path: str | Path):
        self.project_path = Path(project_path).resolve()

    # ----------------------------------------------------- public API
    def index_full(
        self, progress_callback: Optional[Callable[[], None]] = None
    ) -> None:
        """Index *all* files under ``project_path``."""
        try:
            search_engine.index_project(
                self.project_path, progress_callback=progress_callback
            )
        except ImportError as exc:
            raise RuntimeError(
                "search_engine dependencies missing; install 'code-search-mcp[models]' or skip tests."
            ) from exc

    def index_incremental(
        self, progress_callback: Optional[Callable[[], None]] = None
    ) -> None:
        """Index only files that changed since the last run."""
        search_engine.index_project_incremental(
            self.project_path, progress_callback=progress_callback
        )


class Searcher:
    """Provides search and context aggregation against an existing index."""

    def __init__(self, project_path: str | Path | None = None):
        # project_path kept for future multi-index support
        self.project_path = Path(project_path).resolve() if project_path else None

    # ----------------------------------------------------- public API
    def search(
        self,
        query: str,
        *,
        k: int = 5,
        max_tokens: int = 8000,
        metadata_filter: Optional[dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Return raw ranked search results (no concatenation)."""
        return search_engine.search_code_hybrid(
            query_text=query,
            k=k,
            max_tokens=max_tokens,
            metadata_filter=metadata_filter,
        )

    def context(
        self,
        query: str,
        *,
        k: int = 5,
        max_tokens: int = 8000,
        metadata_filter: Optional[dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Return concatenated context suitable for LLM input."""
        return self.search(
            query,
            k=k,
            max_tokens=max_tokens,
            metadata_filter=metadata_filter,
        )

    # ----------------------------------------------------------------- Streaming
    def stream_context(
        self,
        query: str,
        *,
        k: int = 5,
        max_tokens: int = 8000,
        metadata_filter: Optional[dict[str, Any]] = None,
    ):
        """Yield Server-Sent Events for a context request.

        Currently emits a single ``result`` event followed by ``end``. Can be
        extended to chunk-level streaming without breaking contract.
        """

        total_tokens = 0
        for chunk, meta, _total_tokens in search_engine.stream_code_chunks(
            query,
            k=k,
            max_tokens=max_tokens,
            metadata_filter=metadata_filter,
        ):
            total_tokens += _total_tokens
            yield sse_event("chunk", {"content": chunk, "metadata": meta})

        # summary event when done
        yield sse_event("end", {"tokens": total_tokens})
