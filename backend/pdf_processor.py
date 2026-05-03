"""
pdf_processor.py — PDF parsing and text chunking pipeline.

Converts raw PDF bytes into a flat list of overlapping text chunks,
each tagged with the source page number. No disk I/O — operates
entirely in memory using PyMuPDF (fitz).
"""

import fitz  # PyMuPDF
from typing import List


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
    chunk_size: int = 400,
    overlap: int = 80,
) -> List[dict]:
    """
    Split page texts into overlapping fixed-size character windows.

    Each page is chunked independently so a chunk never spans two pages,
    preserving clean page-number attribution in citations.

    Args:
        pages:      Output of parse_pdf() — list of {"page", "text"} dicts.
        chunk_size: Target character count for each chunk (default 500).
        overlap:    Characters shared between consecutive chunks on the same
                    page, giving the retriever context around chunk boundaries
                    (default 50).

    Returns:
        Flat list of chunk dicts:
            {"text": str, "page": int, "chunk_index": int}
        chunk_index is a global counter across all pages, 0-based.
    """
    chunks: List[dict] = []
    chunk_index = 0

    for page_data in pages:
        text = page_data["text"]
        page_num = page_data["page"]
        start = 0

        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append(
                    {
                        "text": chunk_text,
                        "page": page_num,
                        "chunk_index": chunk_index,
                    }
                )
                chunk_index += 1

            # Stop if we've consumed the full page text
            if end >= len(text):
                break

            # Slide forward, keeping `overlap` chars of context
            start = end - overlap

    print(f"[pdf_processor] Created {len(chunks)} chunk(s) from {len(pages)} page(s).")
    return chunks


def parse_and_chunk(
    pdf_bytes: bytes,
    chunk_size: int = 400,
    overlap: int = 80,
) -> List[dict]:
    """
    Full pipeline: bytes → parsed pages → chunks.

    Convenience wrapper that calls parse_pdf then chunk_pages.

    Args:
        pdf_bytes:  Raw bytes of the uploaded PDF.
        chunk_size: Passed through to chunk_pages (default 500 chars).
        overlap:    Passed through to chunk_pages (default 50 chars).

    Returns:
        Flat list of chunk dicts ready for embedding and storage.

    Raises:
        ValueError: Propagated from parse_pdf for bad PDFs.
    """
    pages = parse_pdf(pdf_bytes)
    return chunk_pages(pages, chunk_size=chunk_size, overlap=overlap)
