"""VectorStore interface — persistence and similarity search over vectors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from app.domain.models import RetrievedChunk


class VectorStore(ABC):
    """
    A per-collection vector database.

    Implementations must raise app.errors.QdrantUnavailableError (or an
    equivalent application error) when the backing store is unreachable, rather
    than degrading silently.
    """

    @abstractmethod
    def create_collection(self, name: str, vector_size: int) -> None:
        """Create a fresh collection, dropping any existing one with this name."""

    @abstractmethod
    def collection_exists(self, name: str) -> bool:
        """Return True if a collection with this name exists."""

    @abstractmethod
    def delete_collection(self, name: str) -> None:
        """Drop a collection if present (no-op if absent)."""

    @abstractmethod
    def list_collections(self) -> List[str]:
        """Return the names of all collections in the store."""

    @abstractmethod
    def upsert(
        self,
        name: str,
        ids: List[int],
        vectors: List[List[float]],
        payloads: List[dict],
    ) -> None:
        """Insert/replace points (parallel id/vector/payload lists) in a collection."""

    @abstractmethod
    def search(self, name: str, query_vector: List[float], limit: int) -> List[RetrievedChunk]:
        """Return up to `limit` nearest chunks; each `score` is the cosine similarity."""

    @abstractmethod
    def ping(self) -> bool:
        """Lightweight connectivity probe; True if the store is reachable."""
