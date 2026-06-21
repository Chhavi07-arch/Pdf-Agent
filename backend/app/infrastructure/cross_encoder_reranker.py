"""
cross_encoder_reranker.py — Reranker backed by a sentence-transformers CrossEncoder.

A cross-encoder reads (query, chunk) jointly, judging relevance far more
precisely than the retrieval bi-encoder — affordable only on the small candidate
set hybrid retrieval narrows us to. Loaded lazily on first use, never at startup.
Degrades gracefully to the fused order on any failure so retrieval never breaks.
"""

from __future__ import annotations

import os
import time
from typing import List, Optional

# transformers auto-imports its TensorFlow backend when present, which crashes on
# Keras 3. This app is PyTorch-only — disable the TF/Flax paths BEFORE importing
# sentence_transformers. Set here too (not only in the embedder module) so the
# guard holds regardless of which module imports sentence_transformers first.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")

from sentence_transformers import CrossEncoder

from app.config import CROSS_ENCODER_MODEL
from app.domain.models import RetrievedChunk
from app.interfaces.reranker import Reranker


class CrossEncoderReranker(Reranker):
    """Precision reranker; the cross-encoder model loads on first rerank call."""

    def __init__(self, model_name: str = CROSS_ENCODER_MODEL) -> None:
        self._model_name = model_name
        self._model: Optional[CrossEncoder] = None

    def _get(self) -> CrossEncoder:
        if self._model is None:
            print("[reranker] Loading cross encoder...")
            self._model = CrossEncoder(self._model_name)
            print("[reranker] Cross encoder loaded")
        return self._model

    def rerank(
        self,
        query: str,
        candidates: List[RetrievedChunk],
        top_k: int,
    ) -> List[RetrievedChunk]:
        if not candidates:
            return []

        print(f"[reranker] Reranking {len(candidates)} candidates")
        t0 = time.monotonic()
        try:
            cross_encoder = self._get()
            pairs = [(query, c.text) for c in candidates]
            scores = cross_encoder.predict(pairs)

            for c, s in zip(candidates, scores):
                c.rerank_score = round(float(s), 4)
            ranked = sorted(candidates, key=lambda c: c.rerank_score, reverse=True)[:top_k]

            elapsed = time.monotonic() - t0
            print(f"[reranker] Cross encoder top score={ranked[0].rerank_score if ranked else 'n/a'}")
            print(f"[reranker] Returning top {len(ranked)} reranked chunks ({elapsed:.2f}s)")
            return ranked

        except Exception as exc:  # noqa: BLE001 — must never break retrieval
            print(f"[reranker] Rerank failed: {exc!r} — falling back to fused order.")
            for c in candidates:
                c.rerank_score = c.fused_score
            return candidates[:top_k]
