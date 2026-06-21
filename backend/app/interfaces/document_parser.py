"""DocumentParser interface — bytes to per-page text."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from app.domain.models import Page


class DocumentParser(ABC):
    """Extracts text from a document, one entry per page with text."""

    @abstractmethod
    def parse(self, data: bytes) -> List[Page]:
        """
        Return the pages that contain extractable text.

        Raises ValueError for unusable documents (encrypted, image-only, empty).
        """
