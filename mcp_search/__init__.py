"""Library-first interface for Code Search MCP.

Provides thin `Indexer` and `Searcher` classes that wrap the existing
`search_engine` module so OpenHands runtime containers can import and use
vector search without running the HTTP service.  This keeps the public
surface area small while allowing the underlying implementation to evolve.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, Callable, Dict, Optional

# Re-export version for clients that want to pin behaviour
__version__ = "0.9.0"

# Lazy import to avoid heavy deps (transformers) until actually needed
_se: Any | None = None


def _lazy_search_engine():
    global _se
    if _se is None:
        _se = importlib.import_module("search_engine")
    return _se


__all__ = [
    "Indexer",
    "Searcher",
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
            _lazy_search_engine().index_project(
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
        _lazy_search_engine().index_project_incremental(
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
        try:
            return _lazy_search_engine().search_code_hybrid(
                query_text=query,
                k=k,
                max_tokens=max_tokens,
                metadata_filter=metadata_filter,
            )
        except ImportError as exc:
            raise RuntimeError(
                "search_engine dependencies missing; install 'code-search-mcp[models]' or skip tests."
            ) from exc

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
