"""
pymupdf_parser.py — DocumentParser backed by PyMuPDF (fitz).

Extracts text per page from PDF bytes (no temp file). Returns only pages with
extractable text; raises ValueError for documents the pipeline cannot use
(encrypted, image-only/scanned, empty).
"""

from __future__ import annotations

from typing import List

import fitz  # PyMuPDF

from app.domain.models import Page
from app.interfaces.document_parser import DocumentParser


class PyMuPDFParser(DocumentParser):
    """Text-layer PDF parser."""

    def parse(self, data: bytes) -> List[Page]:
        doc = fitz.open(stream=data, filetype="pdf")

        if doc.is_encrypted:
            doc.close()
            raise ValueError("PDF is password protected")

        pages: List[Page] = []
        has_images_only = False

        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text").strip()
            if text:
                pages.append(Page(page=page_num + 1, text=text))
            elif page.get_images(full=False):
                has_images_only = True

        doc.close()

        if not pages:
            if has_images_only:
                raise ValueError(
                    "PDF appears to be scanned/image-only. "
                    "Text extraction requires a text-layer PDF."
                )
            raise ValueError("PDF contains no extractable text")

        print(f"[parser] Extracted text from {len(pages)} page(s).")
        return pages
