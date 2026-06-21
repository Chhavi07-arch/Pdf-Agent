"""Chunker interface — pages to overlapping, page-tagged chunks."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from app.domain.models import Chunk, Page


class Chunker(ABC):
    """Splits page text into chunks suitable for embedding."""

    @abstractmethod
    def chunk(self, pages: List[Page]) -> List[Chunk]:
        """Return a flat list of chunks; each chunk stays within a single page."""
