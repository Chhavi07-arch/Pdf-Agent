"""Reranker interface — precision re-scoring of a candidate set."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from app.domain.models import RetrievedChunk


class Reranker(ABC):
    """Re-scores retrieved candidates against the query and keeps the best."""

    @abstractmethod
    def rerank(
        self,
        query: str,
        candidates: List[RetrievedChunk],
        top_k: int,
    ) -> List[RetrievedChunk]:
        """
        Return the top `top_k` candidates ordered by rerank relevance.

        Must degrade gracefully (return the input order, trimmed) on failure so
        retrieval never breaks.
        """
