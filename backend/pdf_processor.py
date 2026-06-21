"""
pdf_processor.py — Thin facade over the parsing + chunking infrastructure.

PDF text extraction lives in PyMuPDFParser and chunking in RecursiveChunker.
This module wires them into the single entry point used by main.py and
evaluation.py: bytes → parsed pages → page-tagged chunks (as dicts).
"""

from __future__ import annotations

from typing import List

from app import container
from app.infrastructure.recursive_chunker import RecursiveChunker


def parse_and_chunk(
    pdf_bytes: bytes,
    chunk_size: int = 900,
    chunk_overlap: int = 200,
) -> List[dict]:
    """
    Full pipeline: PDF bytes → parsed pages → overlapping, page-tagged chunks.

    Returns a flat list of {"text", "page", "chunk_index", "section"} dicts ready
    for embedding and Qdrant storage.

    Raises:
        ValueError: for unusable PDFs (encrypted, image-only, empty).
    """
    pages = container.parser().parse(pdf_bytes)
    chunker = RecursiveChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return [c.to_dict() for c in chunker.chunk(pages)]
