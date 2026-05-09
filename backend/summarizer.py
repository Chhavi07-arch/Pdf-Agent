"""
summarizer.py — Upload-time document summary generation and global query routing.

During PDF upload, a lightweight document summary is generated from a small
selection of representative chunks (~5-6 chunks, ≤ 4 000 chars, ~1 000 tokens).
The summary is stored in the session metadata dict and used to answer
document-level questions ("What is this about?", "What topics are covered?")
without triggering vector retrieval.

Architecture contract:
  • generate_doc_summary()  → called once during /upload, result stored in sessions[]
  • is_document_level_query() → called at the start of every /chat turn
  • format_summary_answer()  → replaces get_answer() for document-level turns

Failure contract:
  • Any failure in generate_doc_summary() returns None — the upload still succeeds,
    and document-level queries fall back gracefully to vector retrieval.
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import List, Optional

import requests  # type: ignore

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"
_SUMMARY_MODEL = "mistral-small-latest"

# Maximum characters of PDF text to send for summary generation.
# At ~4 chars/token this is ≈ 1 000 tokens — cheap and Render-safe.
_MAX_INPUT_CHARS = 4_000

# How many representative chunks to select (before the char cap applies).
_NUM_REP_CHUNKS = 6

# ---------------------------------------------------------------------------
# Representative chunk selection
# ---------------------------------------------------------------------------


def _looks_like_heading(text: str) -> bool:
    """
    Return True if the chunk appears to begin with a section heading.

    Headings are typically short single lines (< 80 chars) that do not end
    with sentence punctuation.  Matching them gives the summary model a
    structural overview of the document's sections.
    """
    first_line = text.strip().split("\n")[0].strip()
    return (
        len(first_line) > 0
        and len(first_line) < 80
        and not first_line[-1] in (".", ",", ";", ":")
    )


def select_representative_chunks(
    chunks: List[dict],
    max_chunks: int = _NUM_REP_CHUNKS,
) -> List[dict]:
    """
    Select a small set of representative chunks that give the summary model
    a balanced view of the document.

    Strategy (in priority order):
      1. First 2 chunks — usually title, abstract, or introduction.
      2. Heading-bearing chunks — reveal document structure and section topics.
      3. Evenly-spaced fallback — ensures coverage of middle/end content.

    The character cap (_MAX_INPUT_CHARS) is applied separately in
    generate_doc_summary() after selection.

    Args:
        chunks:     All chunks produced during the upload pipeline.
        max_chunks: Target number to select (default 6).

    Returns:
        Ordered list of up to max_chunks chunk dicts.
    """
    if not chunks:
        return []
    if len(chunks) <= max_chunks:
        return list(chunks)

    selected: List[dict] = []
    seen: set = set()

    # Priority 1 — first two chunks
    for c in chunks[:2]:
        selected.append(c)
        seen.add(c["chunk_index"])

    # Priority 2 — heading chunks (skip first two already selected)
    for c in chunks[2:]:
        if len(selected) >= max_chunks:
            break
        if c["chunk_index"] not in seen and _looks_like_heading(c["text"]):
            selected.append(c)
            seen.add(c["chunk_index"])

    # Priority 3 — evenly-spaced fallback
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


# ---------------------------------------------------------------------------
# Summary generation
# ---------------------------------------------------------------------------


def generate_doc_summary(
    chunks: List[dict],
    filename: str,
    api_key: str,
) -> Optional[dict]:
    """
    Generate a lightweight structured document summary from representative chunks.

    Calls Mistral with ≤ 4 000 chars of context and asks for a JSON document
    profile: title, 2–4 sentence summary, topics, document type, and key entities.
    At ~4 chars/token this is ≈ 1 000 input tokens — cheap and fast.

    Designed to be called once during upload.  Never raises — any failure
    returns None so the upload continues uninterrupted.

    Args:
        chunks:   All chunks from the upload (representative selection is done here).
        filename: Original filename shown to the model as a structural hint.
        api_key:  Mistral API key.

    Returns:
        Dict with keys: title, summary, topics, document_type, entities,
        source_pages (pages the selected chunks came from), or None on failure.
    """
    if not chunks or not api_key:
        return None

    rep_chunks = select_representative_chunks(chunks)

    # Build context string, respecting _MAX_INPUT_CHARS
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
        print("[summarizer] No context built — all chunks exceed char cap.")
        return None

    source_pages = sorted({c["page"] for c in rep_chunks[: len(context_parts)]})
    context = "\n\n".join(context_parts)

    prompt = (
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
        "- summary: 2–4 sentences only.\n"
        "- topics: 3–7 items.\n"
        "- entities: up to 8 items.\n"
        "- document_type: single best match from the list above.\n\n"
        "[DOCUMENT EXCERPTS]\n"
        f"{context}\n"
        "[END EXCERPTS]\n\n"
        "JSON:"
    )

    t0 = time.monotonic()
    try:
        resp = requests.post(
            MISTRAL_API_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": _SUMMARY_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 400,
                "temperature": 0.1,
            },
            timeout=30,
        )
        resp.raise_for_status()
        elapsed = time.monotonic() - t0

        raw_content = resp.json()["choices"][0]["message"]["content"].strip()
        usage = resp.json().get("usage", {})
        print(
            f"[summarizer] Generated in {elapsed:.1f}s | "
            f"tokens={usage.get('prompt_tokens','?')}in/"
            f"{usage.get('completion_tokens','?')}out | "
            f"input_chars={total_chars}"
        )

        # Extract the JSON object — model may wrap it in ```json ... ``` fences
        json_match = re.search(r"\{.*\}", raw_content, re.DOTALL)
        if not json_match:
            print(f"[summarizer] No JSON found in response: {raw_content[:200]!r}")
            return None

        summary = json.loads(json_match.group(0))

        # Validate all required keys are present
        required = {"title", "summary", "topics", "document_type", "entities"}
        missing = required - summary.keys()
        if missing:
            print(f"[summarizer] Missing keys in response: {missing}")
            return None

        # Normalise types (guard against model returning a string where a list is expected)
        summary["topics"] = list(summary.get("topics") or [])
        summary["entities"] = list(summary.get("entities") or [])
        summary["title"] = str(summary.get("title") or filename)
        summary["summary"] = str(summary.get("summary") or "")
        summary["document_type"] = str(summary.get("document_type") or "other")
        summary["source_pages"] = source_pages  # pages the summary was derived from

        print(
            f"[summarizer] type={summary['document_type']!r} | "
            f"topics={summary['topics']} | "
            f"title={summary['title']!r}"
        )
        return summary

    except json.JSONDecodeError as exc:
        print(f"[summarizer] JSON parse error: {exc!r} — skipping summary.")
        return None
    except Exception as exc:
        print(
            f"[summarizer] Summary generation failed: {exc!r} — "
            "document-level queries will fall back to vector retrieval."
        )
        return None


# ---------------------------------------------------------------------------
# Document-level query detection
# ---------------------------------------------------------------------------

# Pass 1 — explicit phrase patterns (checked as startswith or exact match).
# These are unambiguous document-level question forms.
_GLOBAL_PATTERNS: tuple = (
    "what is this",
    "what's this",
    "what is the document",
    "what's the document",
    "what does this document",
    "what does this pdf",
    "what does this file",
    "what does it contain",
    "what does this contain",
    "what's in this",
    "summarize this",
    "summarize the document",
    "summarize the pdf",
    "summarise this",
    "summarise the document",
    "give me a summary",
    "give me an overview",
    "give an overview",
    "what topics are",
    "what topics does",
    "what kind of document",
    "what kind of pdf",
    "what type of document",
    "what type of pdf",
    "what is it about",
    "tell me about this document",
    "tell me about this pdf",
    "describe this document",
    "describe the document",
    "what can you tell me about this",
    "overview of this",
    "summary of this",
)

# Pass 2 — keyword-pair check (applied when Pass 1 misses).
# Requires both a "global intent" word AND a "document reference" word
# so that topic-specific queries like "What topics exist in NLP?" don't match.
_GLOBAL_INTENT_WORDS: frozenset = frozenset({
    "summarize", "summarise", "summary", "overview",
    "about", "contain", "contains",
    "topic", "topics", "cover", "covers", "covered",
    "describe", "description",
})
_DOC_REF_WORDS: frozenset = frozenset({
    "document", "doc", "pdf", "file", "this", "it",
})


def is_document_level_query(query: str) -> bool:
    """
    Return True if the query asks about the document as a whole rather than
    a specific piece of content within it.

    Two-pass detection:
      Pass 1 — pattern match: checks for explicit question forms that
               unambiguously target the whole document.
      Pass 2 — keyword pair: query must contain both a "global intent" word
               (summarize, topics, about, …) AND a "document reference" word
               (document, pdf, this, it, …). This two-word requirement prevents
               false positives from topic-specific queries that happen to contain
               "topics" or "about".

    Examples that return True:
      "What is this document about?"
      "Summarize this document"
      "What topics are covered?"
      "What kind of PDF is this?"
      "Tell me about this document"

    Examples that return False:
      "What is perplexity?"
      "Explain the Adapter pattern"
      "Summarize the alternatives"     ← no doc reference
      "What topics exist in NLP?"      ← no doc reference
    """
    lower = query.lower().strip()

    # Pass 1: explicit patterns
    if any(lower.startswith(p) or lower == p for p in _GLOBAL_PATTERNS):
        print(f"[summarizer] Query classified as DOCUMENT-LEVEL (pattern match): {query!r}")
        return True

    # Pass 2: keyword pair
    words = set(re.findall(r"[a-z]+", lower))
    if (words & _GLOBAL_INTENT_WORDS) and (words & _DOC_REF_WORDS):
        print(f"[summarizer] Query classified as DOCUMENT-LEVEL (keyword pair): {query!r}")
        return True

    return False


# ---------------------------------------------------------------------------
# Summary-based answer formatting
# ---------------------------------------------------------------------------


def format_summary_answer(query: str, summary: dict, filename: str) -> str:
    """
    Compose a natural-language answer to a document-level query from the
    stored summary metadata.

    Adapts its structure based on what the query is focused on:
    - "type / kind"  → document type + summary
    - "topics"       → bulleted topic list + summary
    - default        → full summary block (title + summary + topics + type + entities)

    All answers include a source note indicating which pages the summary
    was derived from, since there are no inline chunk citations for this path.

    Args:
        query:    The original user query (used to shape the response).
        summary:  The stored summary dict from generate_doc_summary().
        filename: Original uploaded filename (fallback if title is missing).

    Returns:
        Markdown-formatted answer string.
    """
    lower = query.lower()

    title = summary.get("title") or filename
    doc_summary = summary.get("summary", "")
    topics = summary.get("topics") or []
    doc_type = summary.get("document_type", "other")
    entities = summary.get("entities") or []
    source_pages = summary.get("source_pages") or []

    # Determine response shape
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
        lines.append(f"**Topics covered in this document:**")
        for t in topics:
            lines.append(f"- {t}")
        if doc_summary:
            lines.append(f"\n{doc_summary}")

    else:
        # Default: full summary block
        lines.append(f"**{title}**")
        lines.append(f"\n{doc_summary}")
        if topics:
            lines.append(f"\n**Key topics:** {', '.join(topics)}")
        if doc_type:
            lines.append(f"**Document type:** {doc_type.title()}")
        if entities:
            # Cap display at 6 to avoid overwhelming the response
            entities_display = ", ".join(entities[:6])
            if len(entities) > 6:
                entities_display += f" *(+{len(entities) - 6} more)*"
            lines.append(f"**Key entities:** {entities_display}")

    # Source note — honest about the absence of inline page citations
    if source_pages:
        pages_str = ", ".join(str(p) for p in source_pages)
        lines.append(
            f"\n*(This overview was generated from pages {pages_str}. "
            f"Ask specific questions to retrieve detailed, page-cited answers.)*"
        )

    return "\n".join(lines)
