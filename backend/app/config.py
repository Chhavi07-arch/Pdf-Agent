"""
config.py — Centralized application configuration (Single Source of Truth).

All environment-variable reads live here behind a single immutable Settings
object, so the rest of the codebase depends on typed configuration rather than
scattered os.getenv() calls (Dependency Inversion: high-level services depend on
this abstraction, not on the environment directly).

Build once at startup with `Settings.from_env()` and pass it into the
composition root. `MIN_RETRIEVAL_SCORE` is intentionally re-read live by the
retrieval layer (see `Settings.live_min_retrieval_score`) to preserve the
existing "tune without restart" behavior.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Static constants (not environment-driven)
# ---------------------------------------------------------------------------

MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"
MISTRAL_MODEL = "mistral-small-latest"

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
VECTOR_SIZE = 384  # all-MiniLM-L6-v2 output dimensionality

CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Hybrid retrieval tuning
CANDIDATE_POOL = 20
SEMANTIC_WEIGHT = 0.6
BM25_WEIGHT = 0.4
RERANK_TOP_K = 7

# Confidence bands (top cosine score)
HIGH_CONFIDENCE_SCORE = 0.45
MEDIUM_CONFIDENCE_SCORE = 0.30

DEFAULT_MIN_RETRIEVAL_SCORE = 0.20

_DEFAULT_ALLOWED_ORIGINS = "http://localhost:5173,http://localhost:3000"


@dataclass(frozen=True)
class Settings:
    """Immutable snapshot of environment-driven configuration."""

    mistral_api_key: str
    qdrant_url: str
    qdrant_api_key: str
    max_pdf_size_mb: int
    enable_reranking: bool
    allowed_origins: list[str] = field(default_factory=list)

    # ---- derived helpers ---------------------------------------------------

    @property
    def max_pdf_size_bytes(self) -> int:
        return self.max_pdf_size_mb * 1024 * 1024

    @property
    def has_qdrant_url(self) -> bool:
        return bool(self.qdrant_url.strip())

    @staticmethod
    def live_min_retrieval_score() -> float:
        """
        Read MIN_RETRIEVAL_SCORE fresh from the environment on every call.

        The retrieval layer uses this (not a cached value) so the score filter
        can be tuned at runtime without restarting the process — matching the
        original embeddings.py behavior.
        """
        return float(os.getenv("MIN_RETRIEVAL_SCORE", str(DEFAULT_MIN_RETRIEVAL_SCORE)))

    @classmethod
    def from_env(cls) -> "Settings":
        """Build a Settings snapshot from the current environment."""
        return cls(
            mistral_api_key=os.getenv("MISTRAL_API_KEY", "").strip(),
            qdrant_url=os.getenv("QDRANT_URL", "").strip(),
            qdrant_api_key=os.getenv("QDRANT_API_KEY", "").strip(),
            max_pdf_size_mb=int(os.getenv("MAX_PDF_SIZE_MB", "20")),
            enable_reranking=os.getenv("ENABLE_RERANKING", "true").lower() == "true",
            allowed_origins=os.getenv("ALLOWED_ORIGINS", _DEFAULT_ALLOWED_ORIGINS).split(","),
        )
