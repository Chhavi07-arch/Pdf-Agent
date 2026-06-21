"""
summarizer.py — Thin facade over SummaryService.

Keeps the function-based API used by main.py (generate_doc_summary,
is_document_level_query, format_summary_answer) while the logic lives in
app/services/summary_service.py.

Failure contract: generate_doc_summary returns None on any failure — the upload
still succeeds and document-level queries fall back to vector retrieval.
"""

from __future__ import annotations

from typing import List, Optional

from app.domain.models import DocSummary
from app.infrastructure.mistral_client import MistralClient
from app.services.summary_service import SummaryService


def generate_doc_summary(chunks: List[dict], filename: str, api_key: str) -> Optional[dict]:
    """Generate a structured document summary (dict) from representative chunks."""
    summary = SummaryService(MistralClient(api_key)).generate(chunks, filename)
    return summary.to_dict() if summary else None


def is_document_level_query(query: str) -> bool:
    """True if the query targets the document as a whole rather than a detail."""
    return SummaryService.is_document_level_query(query)


def format_summary_answer(query: str, summary: dict, filename: str) -> str:
    """Compose a document-level answer from a stored summary dict."""
    return SummaryService.format_answer(query, DocSummary.from_dict(summary), filename)
