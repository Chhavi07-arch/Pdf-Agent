"""Retriever interface — the full query-to-chunks pipeline."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

from app.domain.models import RetrievedChunk


class Retriever(ABC):
    """Retrieves the most relevant chunks for a query within a session."""

    @abstractmethod
    def retrieve(
        self,
        query: str,
        session_id: str,
        top_k: int = 10,
        original_query: Optional[str] = None,
    ) -> List[RetrievedChunk]:
        """
        Return up to `top_k` relevant chunks (best first), or [] when nothing
        clears the relevance threshold so the caller can refuse.
        """
