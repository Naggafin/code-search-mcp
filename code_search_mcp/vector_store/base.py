"""Abstract base definitions for vector store backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, List, Tuple

from code_search_mcp.embedder import (  # Updated for new structure, adjust based on actual imports
    embed,
)


class VectorStore(ABC):
    """Base interface for pluggable vector databases."""

    @abstractmethod
    def add(
        self, chunks: Iterable[Tuple[str, dict]], embeddings: List[List[float]]
    ) -> None:  # noqa: D401 E501
        """Add *chunks* with corresponding *embeddings* to the store."""

    @abstractmethod
    def query(
        self,
        embedding: List[float],
        k: int = 10,
        where: dict | None = None,
    ) -> tuple[List[str], List[dict]]:
        """Return top-*k* documents and metadata most similar to *embedding*."""

    @abstractmethod
    def count(self) -> int:  # pragma: no cover
        """Number of stored vectors."""
