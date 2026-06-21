"""
agent.py — Thin facade over the chat services.

Retrieval orchestration, prompt construction, the Mistral client, query
rewriting, and citation logic now live in app/services and app/infrastructure.
This module wires them into a ChatService and exposes the stable functions
used by main.py and evaluation.py:

    get_answer / stream_answer / is_refusal  (+ _HIGH_CONFIDENCE_SCORE)
"""

from __future__ import annotations

from typing import Iterator, List, Optional

from app import container
from app.config import HIGH_CONFIDENCE_SCORE
from app.services.citation_service import is_refusal  # re-exported for main.py / evaluation.py

# Re-exported for evaluation.py (kept in sync with config so the two never drift).
_HIGH_CONFIDENCE_SCORE = HIGH_CONFIDENCE_SCORE

__all__ = ["get_answer", "stream_answer", "is_refusal", "_HIGH_CONFIDENCE_SCORE"]


def get_answer(
    query: str,
    session_id: str,
    conversation_history: Optional[List[dict]] = None,
) -> dict:
    """Full pipeline → {"answer", "cited_pages", "chunks_used"}."""
    return container.chat_service().get_answer(query, session_id, conversation_history)


def stream_answer(
    query: str,
    session_id: str,
    conversation_history: Optional[List[dict]] = None,
) -> Iterator[str]:
    """Streaming counterpart to get_answer — yields NDJSON event strings."""
    return container.chat_service().stream_answer(query, session_id, conversation_history)
