"""Session and conversation-history repository interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

from app.domain.models import DocSummary, Session


class SessionRepository(ABC):
    """Stores per-session metadata (filename, chunk count, summary, status)."""

    @abstractmethod
    def add(self, session: Session) -> None: ...

    @abstractmethod
    def get(self, session_id: str) -> Optional[Session]: ...

    @abstractmethod
    def exists(self, session_id: str) -> bool: ...

    @abstractmethod
    def delete(self, session_id: str) -> None: ...

    @abstractmethod
    def count(self) -> int: ...

    @abstractmethod
    def set_summary(self, session_id: str, summary: Optional[DocSummary]) -> None:
        """Attach the generated document summary to an existing session."""


class HistoryRepository(ABC):
    """Stores ordered conversation turns ({"role","content"}) per session."""

    @abstractmethod
    def init(self, session_id: str) -> None:
        """Start an empty history for a new session."""

    @abstractmethod
    def get(self, session_id: str) -> List[dict]:
        """Return the session's turns (empty list if none)."""

    @abstractmethod
    def append(self, session_id: str, role: str, content: str) -> None:
        """Append one turn to the session's history."""

    @abstractmethod
    def drop(self, session_id: str) -> None:
        """Discard a session's history (no-op if absent)."""
