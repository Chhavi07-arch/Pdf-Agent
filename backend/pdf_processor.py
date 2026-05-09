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

from typing import List

import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Separator hierarchy: try coarser splits first, fall back to finer ones.
# This keeps sentences whole unless absolutely necessary.
_SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", " ", ""]


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
            {"text": str, "page": int, "chunk_index": int}
        chunk_index is a global counter across all pages, 0-based.
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

    for page_data in pages:
        page_num = page_data["page"]
        # Split this page's text in isolation — cross-page chunks are never produced.
        page_chunks = splitter.split_text(page_data["text"])

        for chunk_text in page_chunks:
            if chunk_text:  # strip_whitespace=True handles leading/trailing, but guard anyway
                chunks.append(
                    {
                        "text":        chunk_text,
                        "page":        page_num,
                        "chunk_index": chunk_index,
                    }
                )
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
