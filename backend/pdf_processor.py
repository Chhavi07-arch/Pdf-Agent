"""
pdf_processor.py — PDF parsing and text chunking pipeline.

Converts raw PDF bytes into a flat list of overlapping text chunks,
each tagged with the source page number. No disk I/O — operates
entirely in memory using PyMuPDF (fitz).

Chunking strategy:
  • RecursiveCharacterTextSplitter walks a separator hierarchy:
      paragraph breaks (\n\n) → line breaks (\n) → sentence ends (. ) →
      word boundaries ( ) → character fallback
    so chunks end at natural linguistic boundaries rather than mid-sentence.
  • Each page is split INDEPENDENTLY — a chunk never crosses a page boundary.
    This guarantees that the page metadata on every chunk is always accurate,
    which is the foundation of trustworthy [Page N] citations.
  • chunk_size=900 chars (~130 words) captures a full paragraph as one unit,
    giving the embedding model enough context to represent a complete idea.
  • chunk_overlap=200 chars ensures sentences near chunk boundaries appear in
    both adjacent chunks, so the retriever never misses a fact just because it
    straddled a boundary.
"""

import re
from typing import List, Optional

import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Separator hierarchy: try coarser splits first, fall back to finer ones.
# This keeps sentences whole unless absolutely necessary.
_SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", " ", ""]

# ---------------------------------------------------------------------------
# Section-header detection (Step 3 — contextual section metadata)
# ---------------------------------------------------------------------------
#
# Goal: tag each chunk with the document section it belongs to, as METADATA
# only — chunk text is never modified, so embeddings and retrieval ranking are
# unaffected. The section field is used for observability and can surface
# structure to the UI / citations later.

# Numbered heading: "1 Intro", "1. Adapter", "2.3 Forces" — requires text after
# the number so bare page numbers ("8") are not mistaken for headings.
_HEADING_NUMBERED_RE = re.compile(r"^\d+(\.\d+)*\.?\s+\S")
# Keyword heading: "Chapter 1", "Section 2", "Part II", "Appendix A".
_HEADING_KEYWORD_RE = re.compile(r"^(chapter|section|part|appendix)\b", re.IGNORECASE)

# Default section before any heading is seen.
_UNKNOWN_SECTION = "Unknown"


def _normalize_heading(text: str) -> str:
    """All-caps headings → Title Case for readability ('ADAPTER PATTERN' → 'Adapter Pattern')."""
    return text.title() if text == text.upper() else text


def _extract_section(line: str) -> Optional[str]:
    """
    Return a cleaned section name if `line` looks like a heading, else None.

    A heading must be a SHORT line (< 60 chars) AND match one of:
      1. Ends with ':'            → "Responsibilities:", "Drawbacks:"
      2. ALL CAPS                 → "ADAPTER PATTERN", "DESIGN PRINCIPLES"
      3. Numbered / keyword       → "1. Adapter", "2.3 Forces", "Chapter 1"
      4. Guarded title-case line  → "Forces, Constraints, Goals", "Pitfalls"

    The short-line requirement plus these signals keeps detection precise — a
    generic short line alone is NOT treated as a heading (that would reset the
    section on nearly every wrapped line).
    """
    s = line.strip()
    if not s or len(s) >= 60:
        return None

    # 1. Ends with a colon
    if s.endswith(":"):
        name = s[:-1].strip()
        return name or None

    # 2. ALL CAPS (with at least two letters, not a sentence)
    alpha = [c for c in s if c.isalpha()]
    if len(alpha) >= 2 and s == s.upper() and not s.endswith((".", ",", ";")):
        return _normalize_heading(s)

    # 3. Numbered / chapter-section style
    if _HEADING_NUMBERED_RE.match(s) or _HEADING_KEYWORD_RE.match(s):
        return s

    # 4. Guarded title-case heading: short, no terminal sentence punctuation,
    #    ≤ 6 words, and either a single word or mostly-capitalized words.
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


def parse_pdf(pdf_bytes: bytes) -> List[dict]:
    """
    Extract text from every page of a PDF.

    Opens the document from bytes (no temp file), iterates pages,
    and returns only pages that have extractable text content.

    Args:
        pdf_bytes: Raw bytes of the uploaded PDF file.

    Returns:
        List of page dicts: [{"page": int (1-indexed), "text": str}, ...]

    Raises:
        ValueError: If the PDF is password-protected, image-only, or empty.
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    if doc.is_encrypted:
        doc.close()
        raise ValueError("PDF is password protected")

    pages: List[dict] = []
    has_images_only = False

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text").strip()

        if text:
            pages.append({"page": page_num + 1, "text": text})
        else:
            # Track whether blank pages have raster images (scanned doc indicator)
            if page.get_images(full=False):
                has_images_only = True

    doc.close()

    if not pages:
        if has_images_only:
            raise ValueError(
                "PDF appears to be scanned/image-only. "
                "Text extraction requires a text-layer PDF."
            )
        raise ValueError("PDF contains no extractable text")

    print(f"[pdf_processor] Extracted text from {len(pages)} page(s).")
    return pages


def chunk_pages(
    pages: List[dict],
    chunk_size: int = 900,
    chunk_overlap: int = 200,
) -> List[dict]:
    """
    Split page texts into overlapping chunks using RecursiveCharacterTextSplitter.

    Each page is split INDEPENDENTLY so a chunk never spans two pages,
    preserving accurate page-number attribution in citations.

    RecursiveCharacterTextSplitter walks a separator hierarchy (paragraph →
    line → sentence → word → character) and only falls back to a coarser
    split if the text at a given boundary would still exceed chunk_size.
    In practice this means almost all chunks end at sentence or paragraph
    boundaries rather than mid-word like the old character-window approach.

    Args:
        pages:        Output of parse_pdf() — list of {"page", "text"} dicts.
        chunk_size:   Target character count per chunk (default 900).
                      ~130 words, enough to capture a full paragraph as one
                      semantic unit without diluting the embedding.
        chunk_overlap: Characters shared between consecutive chunks on the same
                      page (default 200, ~22% of chunk_size). Ensures sentences
                      near chunk boundaries are embedded in both neighbours.

    Returns:
        Flat list of chunk dicts:
            {"text": str, "page": int, "chunk_index": int, "section": str}
        chunk_index is a global counter across all pages, 0-based.
        section is the detected heading the chunk falls under (metadata only —
        chunk text is never modified), or "Unknown" before any heading is seen.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=_SEPARATORS,
        length_function=len,       # character count — deterministic, no tokeniser
        is_separator_regex=False,
        strip_whitespace=True,
    )

    chunks: List[dict] = []
    chunk_index = 0
    current_section = _UNKNOWN_SECTION  # persists across pages until a new heading

    for page_data in pages:
        page_num = page_data["page"]
        page_text = page_data["text"]

        # ── 1. Detect headings and record their character offset in the page ──
        headings: List[tuple] = []  # [(offset, section_name), ...] in reading order
        offset = 0
        for line in page_text.split("\n"):
            name = _extract_section(line)
            if name:
                headings.append((offset, name))
                print(f"[pdf_processor] Detected section: {name}")
            offset += len(line) + 1  # +1 for the stripped '\n'

        # ── 2. Split this page's text in isolation (chunking unchanged) ──────
        page_chunks = splitter.split_text(page_text)

        # ── 3. Assign each chunk the most recent heading at/before its start ──
        cursor = 0
        for chunk_text in page_chunks:
            if not chunk_text:  # strip_whitespace=True handles ends, but guard anyway
                continue

            # Locate the chunk's start offset within the page text so we can map
            # it to the section it falls under. Search by a short prefix to stay
            # robust to the splitter's whitespace trimming.
            probe = chunk_text[:40] if len(chunk_text) >= 40 else chunk_text
            pos = page_text.find(probe, cursor)
            if pos == -1:
                pos = page_text.find(chunk_text[:20], cursor)
            start = pos if pos != -1 else cursor
            if pos != -1:
                cursor = pos + 1

            # Advance current_section through any headings at/before this start.
            for h_off, h_name in headings:
                if h_off <= start:
                    current_section = h_name
                else:
                    break

            chunks.append(
                {
                    "text":        chunk_text,
                    "page":        page_num,
                    "chunk_index": chunk_index,
                    "section":     current_section,
                }
            )
            print(f"[pdf_processor] Chunk assigned to section: {current_section}")
            chunk_index += 1

    print(f"[pdf_processor] Created {len(chunks)} chunk(s) from {len(pages)} page(s).")
    return chunks


def parse_and_chunk(
    pdf_bytes: bytes,
    chunk_size: int = 900,
    chunk_overlap: int = 200,
) -> List[dict]:
    """
    Full pipeline: bytes → parsed pages → chunks.

    Convenience wrapper that calls parse_pdf then chunk_pages.

    Args:
        pdf_bytes:    Raw bytes of the uploaded PDF.
        chunk_size:   Passed through to chunk_pages (default 900 chars).
        chunk_overlap: Passed through to chunk_pages (default 200 chars).

    Returns:
        Flat list of chunk dicts ready for embedding and Qdrant storage.

    Raises:
        ValueError: Propagated from parse_pdf for bad PDFs.
    """
    pages = parse_pdf(pdf_bytes)
    return chunk_pages(pages, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
