"""
bm25_keyword_index.py — KeywordIndex backed by rank-bm25 (BM25Okapi).

One in-memory BM25 index per session, built alongside the vector store. A
session with no built index simply returns no keyword hits (semantic-only
fallback) — e.g. a Qdrant session that survived a process restart.
"""

from __future__ import annotations

import re
from typing import Dict, List

import numpy as np
from rank_bm25 import BM25Okapi

from app.domain.models import Chunk, RetrievedChunk
from app.interfaces.keyword_index import KeywordIndex


class BM25KeywordIndex(KeywordIndex):
    """Per-session BM25 keyword index held in process memory."""

    def __init__(self) -> None:
        #  session_id -> {"bm25": BM25Okapi, "chunks": List[Chunk]}
        self._indexes: Dict[str, dict] = {}

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Lowercase alphanumeric word tokenizer used for indexing and querying."""
        return re.findall(r"[a-z0-9]+", text.lower())

    def build(self, session_id: str, chunks: List[Chunk]) -> None:
        corpus_tokens = [self._tokenize(c.text) for c in chunks]
        self._indexes[session_id] = {
            "bm25": BM25Okapi(corpus_tokens),
            "chunks": list(chunks),
        }
        print(f"[bm25] Built index for session '{session_id}' ({len(chunks)} docs).")

    def search(self, session_id: str, query: str, limit: int) -> List[RetrievedChunk]:
        entry = self._indexes.get(session_id)
        if not entry:
            print(f"[bm25] No index for session '{session_id}' — semantic-only fallback.")
            return []

        tokens = self._tokenize(query)
        if not tokens:
            return []

        bm25: BM25Okapi = entry["bm25"]
        chunks: List[Chunk] = entry["chunks"]
        scores = bm25.get_scores(tokens)  # np.ndarray aligned with chunks

        order = np.argsort(scores)[::-1][:limit]
        results: List[RetrievedChunk] = []
        for i in order:
            if scores[i] <= 0:
                continue  # no term overlap — not a real keyword match
            c = chunks[int(i)]
            results.append(
                RetrievedChunk(
                    text=c.text,
                    page=c.page,
                    chunk_index=c.chunk_index,
                    section=c.section,
                    score=float(scores[i]),  # raw BM25 score
                )
            )
        return results

    def drop(self, session_id: str) -> None:
        if self._indexes.pop(session_id, None) is not None:
            print(f"[bm25] Dropped index for session '{session_id}'.")
