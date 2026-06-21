"""KeywordIndex interface — lexical (BM25-style) retrieval."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from app.domain.models import Chunk, RetrievedChunk


class KeywordIndex(ABC):
    """
    A per-session keyword index complementing semantic search.

    Held in process memory; a session with no built index simply yields no
    keyword hits (semantic-only fallback), it does not error.
    """

    @abstractmethod
    def build(self, session_id: str, chunks: List[Chunk]) -> None:
        """Build and cache the index for a session's chunks."""

    @abstractmethod
    def search(self, session_id: str, query: str, limit: int) -> List[RetrievedChunk]:
        """Return up to `limit` chunks; each `score` is the raw keyword score."""

    @abstractmethod
    def drop(self, session_id: str) -> None:
        """Discard a session's index (no-op if absent)."""
