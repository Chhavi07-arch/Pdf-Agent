"""
memory_session_repository.py — In-memory Session and History repositories.

Process-local dicts (intentional for the assessment scope; a production system
would use Redis or a database). Sessions and conversation history are lost on
restart — the frontend re-uploads to recover.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from app.domain.models import DocSummary, Session
from app.interfaces.session_repository import HistoryRepository, SessionRepository


class MemorySessionRepository(SessionRepository):
    """Stores Session metadata in a process-local dict."""

    def __init__(self) -> None:
        self._sessions: Dict[str, Session] = {}

    def add(self, session: Session) -> None:
        self._sessions[session.session_id] = session

    def get(self, session_id: str) -> Optional[Session]:
        return self._sessions.get(session_id)

    def exists(self, session_id: str) -> bool:
        return session_id in self._sessions

    def delete(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    def count(self) -> int:
        return len(self._sessions)

    def set_summary(self, session_id: str, summary: Optional[DocSummary]) -> None:
        session = self._sessions.get(session_id)
        if session is not None:
            session.doc_summary = summary


class MemoryHistoryRepository(HistoryRepository):
    """Stores ordered conversation turns per session in a process-local dict."""

    def __init__(self) -> None:
        self._histories: Dict[str, List[dict]] = {}

    def init(self, session_id: str) -> None:
        self._histories[session_id] = []

    def get(self, session_id: str) -> List[dict]:
        return self._histories.get(session_id, [])

    def append(self, session_id: str, role: str, content: str) -> None:
        self._histories.setdefault(session_id, []).append({"role": role, "content": content})

    def drop(self, session_id: str) -> None:
        self._histories.pop(session_id, None)
