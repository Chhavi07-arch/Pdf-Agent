"""LLMClient interface — chat completion (blocking + streaming)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator, List, Optional


class LLMClient(ABC):
    """A chat-completion provider (OpenAI-compatible message format)."""

    @abstractmethod
    def complete(
        self,
        messages: List[dict],
        max_tokens: int,
        temperature: Optional[float] = None,
    ) -> str:
        """Return the assistant message content for a blocking completion."""

    @abstractmethod
    def stream(self, messages: List[dict], max_tokens: int) -> Iterator[str]:
        """Yield content token deltas as they arrive from a streaming completion."""
