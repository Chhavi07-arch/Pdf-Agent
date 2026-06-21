"""
sentence_transformer_embedder.py — Embedder backed by sentence-transformers.

Loads the model lazily on first encode (never at import/startup) to keep peak
memory low on small hosts. Encodes in small batches with GC between them, the
same strategy used by the original embeddings module.
"""

from __future__ import annotations

import gc
import os
from typing import List

# transformers (pulled in by sentence-transformers) auto-imports its TensorFlow
# backend when present, which crashes on Keras 3. This app is PyTorch-only, so
# disable the TF/Flax paths BEFORE importing sentence_transformers.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")

import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import EMBEDDING_MODEL_NAME, VECTOR_SIZE
from app.interfaces.embedder import Embedder


class SentenceTransformerEmbedder(Embedder):
    """Embedder using a sentence-transformers bi-encoder (default all-MiniLM-L6-v2)."""

    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME, batch_size: int = 8) -> None:
        self._model_name = model_name
        self._batch_size = batch_size
        self._model: SentenceTransformer | None = None

    def _get(self) -> SentenceTransformer:
        if self._model is None:
            print(f"[embedder] Loading sentence-transformers model ({self._model_name})…")
            self._model = SentenceTransformer(self._model_name)
            print("[embedder] Model loaded.")
        return self._model

    def encode(self, texts: List[str]) -> "np.ndarray":
        """Encode texts in small batches (with GC between). Returns (N, dim) float32."""
        model = self._get()
        batches = []
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            embs = model.encode(batch, show_progress_bar=False, convert_to_numpy=True)
            batches.append(embs.astype(np.float32))
            gc.collect()
        return np.vstack(batches)

    @property
    def dimension(self) -> int:
        return VECTOR_SIZE

    @property
    def is_loaded(self) -> bool:
        return self._model is not None
