"""
recursive_chunker.py — Chunker using LangChain's RecursiveCharacterTextSplitter.

Each page is split INDEPENDENTLY (a chunk never crosses a page boundary) so the
page number on every chunk is accurate — the foundation of trustworthy [Page N]
citations. Chunks are tagged with the document section they fall under (metadata
only; chunk text is never modified, so embeddings/ranking are unaffected).
"""

from __future__ import annotations

import re
from typing import List, Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.domain.models import Chunk, Page
from app.interfaces.chunker import Chunker

# Separator hierarchy: coarser splits first (paragraph → line → sentence → word
# → char), so sentences stay whole unless necessary.
_SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", " ", ""]

_HEADING_NUMBERED_RE = re.compile(r"^\d+(\.\d+)*\.?\s+\S")
_HEADING_KEYWORD_RE = re.compile(r"^(chapter|section|part|appendix)\b", re.IGNORECASE)
_UNKNOWN_SECTION = "Unknown"


def _normalize_heading(text: str) -> str:
    """All-caps headings → Title Case ('ADAPTER PATTERN' → 'Adapter Pattern')."""
    return text.title() if text == text.upper() else text


def _extract_section(line: str) -> Optional[str]:
    """Return a cleaned section name if `line` looks like a heading, else None."""
    s = line.strip()
    if not s or len(s) >= 60:
        return None

    if s.endswith(":"):
        name = s[:-1].strip()
        return name or None

    alpha = [c for c in s if c.isalpha()]
    if len(alpha) >= 2 and s == s.upper() and not s.endswith((".", ",", ";")):
        return _normalize_heading(s)

    if _HEADING_NUMBERED_RE.match(s) or _HEADING_KEYWORD_RE.match(s):
        return s

    words = s.split()
    if (
        1 <= len(words) <= 6
        and s[0].isupper()
        and not s.endswith((".", "!", "?", ",", ";"))
        and not s.islower()
    ):
        cap_words = sum(1 for w in words if w[:1].isupper())
        if len(words) == 1 or cap_words >= max(1, len(words) // 2):
            return s

    return None


class RecursiveChunker(Chunker):
    """Per-page recursive character chunker with section tagging."""

    def __init__(self, chunk_size: int = 900, chunk_overlap: int = 200) -> None:
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=_SEPARATORS,
            length_function=len,
            is_separator_regex=False,
            strip_whitespace=True,
        )

    def chunk(self, pages: List[Page]) -> List[Chunk]:
        chunks: List[Chunk] = []
        chunk_index = 0
        current_section = _UNKNOWN_SECTION  # persists across pages until a new heading

        for page_data in pages:
            page_num = page_data.page
            page_text = page_data.text

            # 1. Detect headings and record their character offset in the page.
            headings: List[tuple] = []
            offset = 0
            for line in page_text.split("\n"):
                name = _extract_section(line)
                if name:
                    headings.append((offset, name))
                offset += len(line) + 1  # +1 for the stripped '\n'

            # 2. Split this page's text in isolation.
            page_chunks = self._splitter.split_text(page_text)

            # 3. Assign each chunk the most recent heading at/before its start.
            cursor = 0
            for chunk_text in page_chunks:
                if not chunk_text:
                    continue

                probe = chunk_text[:40] if len(chunk_text) >= 40 else chunk_text
                pos = page_text.find(probe, cursor)
                if pos == -1:
                    pos = page_text.find(chunk_text[:20], cursor)
                start = pos if pos != -1 else cursor
                if pos != -1:
                    cursor = pos + 1

                for h_off, h_name in headings:
                    if h_off <= start:
                        current_section = h_name
                    else:
                        break

                chunks.append(
                    Chunk(
                        text=chunk_text,
                        page=page_num,
                        chunk_index=chunk_index,
                        section=current_section,
                    )
                )
                chunk_index += 1

        print(f"[chunker] Created {len(chunks)} chunk(s) from {len(pages)} page(s).")
        return chunks
