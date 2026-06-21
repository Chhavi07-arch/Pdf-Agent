"""
agent.py — Thin facade over the chat services.

Retrieval orchestration, prompt construction, the Mistral client, query
rewriting, and citation logic now live in app/services and app/infrastructure.
This module wires them into a ChatService and exposes the stable functions
used by main.py and evaluation.py:

    get_answer / stream_answer / is_refusal  (+ _HIGH_CONFIDENCE_SCORE)
"""

from __future__ import annotations

import os
from typing import Iterator, List, Optional

from app.config import HIGH_CONFIDENCE_SCORE
from app.infrastructure.mistral_client import MistralClient
from app.services.chat_service import ChatService
from app.services.citation_service import is_refusal  # re-exported for main.py / evaluation.py
from app.services.query_builder import QueryBuilder

# Re-exported for evaluation.py (kept in sync with config so the two never drift).
_HIGH_CONFIDENCE_SCORE = HIGH_CONFIDENCE_SCORE

__all__ = ["get_answer", "stream_answer", "is_refusal", "_HIGH_CONFIDENCE_SCORE"]

_chat_service: Optional[ChatService] = None


def _get_chat_service() -> ChatService:
    """
    Build the ChatService on first use.

    Lazy because the Mistral API key and the Qdrant retriever both depend on env
    vars loaded after import time. Reuses the shared hybrid retriever singleton
    from embeddings so retrieval state (BM25 indexes, diagnostics) is consistent
    across upload and chat.
    """
    global _chat_service
    if _chat_service is None:
        from embeddings import _get_retriever  # local import avoids import cycle

        llm = MistralClient(os.getenv("MISTRAL_API_KEY", ""))
        query_builder = QueryBuilder(llm)
        _chat_service = ChatService(_get_retriever(), llm, query_builder)
    return _chat_service


def get_answer(
    query: str,
    session_id: str,
    conversation_history: Optional[List[dict]] = None,
) -> dict:
    """Full pipeline → {"answer", "cited_pages", "chunks_used"}."""
    return _get_chat_service().get_answer(query, session_id, conversation_history)


def stream_answer(
    query: str,
    session_id: str,
    conversation_history: Optional[List[dict]] = None,
) -> Iterator[str]:
    """Streaming counterpart to get_answer — yields NDJSON event strings."""
    return _get_chat_service().stream_answer(query, session_id, conversation_history)
