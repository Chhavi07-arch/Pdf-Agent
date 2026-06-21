"""Embedder interface — turns text into dense vectors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    import numpy as np


class Embedder(ABC):
    """Encodes text into fixed-size embedding vectors."""

    @abstractmethod
    def encode(self, texts: List[str]) -> "np.ndarray":
        """Return an (N, dimension) float32 array of embeddings for `texts`."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Output vector dimensionality."""
