"""
summary_service.py — Upload-time document summary + document-level query routing.

At upload, a lightweight structured summary is generated from a few
representative chunks (via the injected LLMClient) and stored in session
metadata. Document-level questions ("What is this about?", "Summarize this")
are answered from that summary, bypassing vector retrieval.

Never raises out of generate(): any failure returns None so the upload still
succeeds and document-level queries fall back to retrieval.
"""

from __future__ import annotations

import json
import re
from typing import List, Optional

from app.domain.models import DocSummary
from app.interfaces.llm_client import LLMClient

# Max characters of PDF text to send for summary generation (~1 000 tokens).
_MAX_INPUT_CHARS = 4_000
_NUM_REP_CHUNKS = 6

# ── document-level query detection ──────────────────────────────────────────
_GLOBAL_PATTERNS = (
    "what is this", "what's this", "what is the document", "what's the document",
    "what does this document", "what does this pdf", "what does this file",
    "what does it contain", "what does this contain", "what's in this",
    "summarize this", "summarize the document", "summarize the pdf",
    "summarise this", "summarise the document", "give me a summary",
    "give me an overview", "give an overview", "what topics are", "what topics does",
    "what kind of document", "what kind of pdf", "what type of document",
    "what type of pdf", "what is it about", "tell me about this document",
    "tell me about this pdf", "describe this document", "describe the document",
    "what can you tell me about this", "overview of this", "summary of this",
)
_GLOBAL_INTENT_WORDS = frozenset({
    "summarize", "summarise", "summary", "overview", "about", "contain", "contains",
    "topic", "topics", "cover", "covers", "covered", "describe", "description",
})
_DOC_REF_WORDS = frozenset({"document", "doc", "pdf", "file", "this", "it"})


def _looks_like_heading(text: str) -> bool:
    first_line = text.strip().split("\n")[0].strip()
    return (
        len(first_line) > 0
        and len(first_line) < 80
        and first_line[-1] not in (".", ",", ";", ":")
    )


def select_representative_chunks(chunks: List[dict], max_chunks: int = _NUM_REP_CHUNKS) -> List[dict]:
    """First 2 chunks (intro), then heading-bearing, then evenly-spaced fallback."""
    if not chunks:
        return []
    if len(chunks) <= max_chunks:
        return list(chunks)

    selected: List[dict] = []
    seen: set = set()
    for c in chunks[:2]:
        selected.append(c)
        seen.add(c["chunk_index"])
    for c in chunks[2:]:
        if len(selected) >= max_chunks:
            break
        if c["chunk_index"] not in seen and _looks_like_heading(c["text"]):
            selected.append(c)
            seen.add(c["chunk_index"])
    if len(selected) < max_chunks:
        slots_needed = max_chunks - len(selected)
        step = max(1, len(chunks) // (slots_needed + 1))
        for i in range(step, len(chunks), step):
            if len(selected) >= max_chunks:
                break
            c = chunks[i]
            if c["chunk_index"] not in seen:
                selected.append(c)
                seen.add(c["chunk_index"])
    return selected


class SummaryService:
    """Generates and serves document-level summaries."""

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    # ── generation ──────────────────────────────────────────────────────────
    def generate(self, chunks: List[dict], filename: str) -> Optional[DocSummary]:
        if not chunks or not self._llm.available:
            return None

        rep_chunks = select_representative_chunks(chunks)
        context_parts: List[str] = []
        total_chars = 0
        for c in rep_chunks:
            excerpt = f"[Page {c['page']}]\n{c['text']}"
            if total_chars + len(excerpt) > _MAX_INPUT_CHARS:
                remaining = _MAX_INPUT_CHARS - total_chars
                if remaining > 200:
                    context_parts.append(excerpt[:remaining])
                break
            context_parts.append(excerpt)
            total_chars += len(excerpt)

        if not context_parts:
            print("[summary] No context built — all chunks exceed char cap.")
            return None

        source_pages = sorted({c["page"] for c in rep_chunks[: len(context_parts)]})
        context = "\n\n".join(context_parts)
        prompt = self._build_prompt(filename, context)

        try:
            raw = self._llm.complete([{"role": "user", "content": prompt}], max_tokens=400, temperature=0.1)
        except Exception as exc:  # noqa: BLE001 — upload must still succeed
            print(f"[summary] Generation failed: {exc!r} — falling back to retrieval.")
            return None

        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            print(f"[summary] No JSON found in response: {raw[:200]!r}")
            return None
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            print(f"[summary] JSON parse error: {exc!r} — skipping summary.")
            return None

        required = {"title", "summary", "topics", "document_type", "entities"}
        if required - data.keys():
            print(f"[summary] Missing keys: {required - data.keys()}")
            return None

        summary = DocSummary(
            title=str(data.get("title") or filename),
            summary=str(data.get("summary") or ""),
            topics=list(data.get("topics") or []),
            document_type=str(data.get("document_type") or "other"),
            entities=list(data.get("entities") or []),
            source_pages=source_pages,
        )
        print(f"[summary] type={summary.document_type!r} | topics={summary.topics} | title={summary.title!r}")
        return summary

    @staticmethod
    def _build_prompt(filename: str, context: str) -> str:
        return (
            f'You are analyzing excerpts from a PDF document named "{filename}".\n'
            "Based ONLY on the excerpts below, output a structured JSON profile.\n"
            "Output valid JSON and nothing else — no prose before or after.\n\n"
            "Required structure (use these exact keys):\n"
            "{\n"
            '  "title": "Document title or best inference from content",\n'
            '  "summary": "2–4 sentence description of what this document is about",\n'
            '  "topics": ["topic1", "topic2", "topic3"],\n'
            '  "document_type": "exactly one of: resume/CV | research paper | '
            'technical documentation | report | article | book chapter | tutorial | '
            'specification | other",\n'
            '  "entities": ["key people / organisations / technologies / concepts"]\n'
            "}\n\n"
            "Rules:\n"
            "- Base everything strictly on the provided text — no external knowledge.\n"
            "- summary: 2–4 sentences only.\n- topics: 3–7 items.\n"
            "- entities: up to 8 items.\n- document_type: single best match from the list above.\n\n"
            "[DOCUMENT EXCERPTS]\n"
            f"{context}\n"
            "[END EXCERPTS]\n\nJSON:"
        )

    # ── routing + formatting ──────────────────────────────────────────────────
    @staticmethod
    def is_document_level_query(query: str) -> bool:
        lower = query.lower().strip()
        if any(lower.startswith(p) or lower == p for p in _GLOBAL_PATTERNS):
            print(f"[summary] DOCUMENT-LEVEL (pattern): {query!r}")
            return True
        words = set(re.findall(r"[a-z]+", lower))
        if (words & _GLOBAL_INTENT_WORDS) and (words & _DOC_REF_WORDS):
            print(f"[summary] DOCUMENT-LEVEL (keyword pair): {query!r}")
            return True
        return False

    @staticmethod
    def format_answer(query: str, summary: DocSummary, filename: str) -> str:
        lower = query.lower()
        title = summary.title or filename
        doc_summary = summary.summary
        topics = summary.topics
        doc_type = summary.document_type
        entities = summary.entities
        source_pages = summary.source_pages

        asks_type = any(w in lower for w in ("kind", "type"))
        asks_topics = any(w in lower for w in ("topic", "topics", "cover", "covers", "covered"))

        lines: List[str] = []
        if asks_type:
            lines.append(f"**Document type:** {doc_type.title()}")
            lines.append(f"**Title:** {title}")
            if doc_summary:
                lines.append(f"\n{doc_summary}")
            if topics:
                lines.append(f"\n**Key topics:** {', '.join(topics)}")
        elif asks_topics:
            lines.append("**Topics covered in this document:**")
            for t in topics:
                lines.append(f"- {t}")
            if doc_summary:
                lines.append(f"\n{doc_summary}")
        else:
            lines.append(f"**{title}**")
            lines.append(f"\n{doc_summary}")
            if topics:
                lines.append(f"\n**Key topics:** {', '.join(topics)}")
            if doc_type:
                lines.append(f"**Document type:** {doc_type.title()}")
            if entities:
                entities_display = ", ".join(entities[:6])
                if len(entities) > 6:
                    entities_display += f" *(+{len(entities) - 6} more)*"
                lines.append(f"**Key entities:** {entities_display}")

        if source_pages:
            pages_str = ", ".join(str(p) for p in source_pages)
            lines.append(
                f"\n*(This overview was generated from pages {pages_str}. "
                f"Ask specific questions to retrieve detailed, page-cited answers.)*"
            )
        return "\n".join(lines)
