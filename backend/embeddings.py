"""
embeddings.py — Thin façade over the retrieval stack.

Embedding, Qdrant I/O, BM25, reranking, and fusion now live in the
infrastructure and services layers. This module wires those singletons together
and exposes the small, stable API used by agent.py and main.py:

    embed_and_store / retrieve_relevant_chunks / retrieve_multi_query /
    delete_session / warmup / get_status / check_qdrant_connectivity /
    get_last_retrieval_debug

A running Qdrant server is mandatory — QDRANT_URL must be set. There is no
in-memory fallback: if Qdrant is unconfigured or unreachable, the vector store
raises ConfigError / QdrantUnavailableError, surfaced by the API as
"Qdrant server not working".

Collection strategy: one Qdrant collection per session_id (UUID) — natural
isolation, atomic deletion, no per-query metadata filter.
"""

from __future__ import annotations

import gc
import os
from typing import List, Optional

from app.config import VECTOR_SIZE
from app.domain.models import Chunk
from app.infrastructure.bm25_keyword_index import BM25KeywordIndex
from app.infrastructure.cross_encoder_reranker import CrossEncoderReranker
from app.infrastructure.qdrant_vector_store import QdrantVectorStore
from app.infrastructure.sentence_transformer_embedder import SentenceTransformerEmbedder
from app.services.hybrid_retriever import HybridRetriever

# TODO:
# Qdrant currently serves as the vector store. If future requirements demand a
# relational solution, migrate via pgvector by providing a new VectorStore
# implementation — no change needed here or in the retrieval service.

# Cross-encoder reranking is the single largest memory cost (a second transformer
# model). On low-memory hosts (e.g. Render free tier, 512 MB) set
# ENABLE_RERANKING=false: the CrossEncoder is never loaded and retrieval returns
# the hybrid (BM25 + semantic) results directly. Default true.
ENABLE_RERANKING = os.getenv("ENABLE_RERANKING", "true").lower() == "true"

# ---------------------------------------------------------------------------
# Singletons (lazy where construction depends on env / heavy models)
# ---------------------------------------------------------------------------

_embedder = SentenceTransformerEmbedder()      # transformer loads on first encode
_keyword_index = BM25KeywordIndex()            # in-process per-session BM25
_reranker = CrossEncoderReranker()             # cross-encoder loads on first rerank
_store: Optional[QdrantVectorStore] = None     # constructed from env on first use
_retriever: Optional[HybridRetriever] = None


def _get_store() -> QdrantVectorStore:
    """
    Return the shared Qdrant vector store, constructed on first call from the
    current QDRANT_URL / QDRANT_API_KEY env vars.

    A running Qdrant server is mandatory — no in-memory fallback. The store
    raises ConfigError (URL unset) or QdrantUnavailableError (unreachable) on
    first use; callers propagate those so the API answers "Qdrant server not
    working".
    """
    global _store
    if _store is None:
        _store = QdrantVectorStore(
            url=os.getenv("QDRANT_URL", "").strip(),
            api_key=os.getenv("QDRANT_API_KEY", "").strip(),
        )
    return _store


def _get_retriever() -> HybridRetriever:
    """Return the shared hybrid retriever, composing the infra singletons."""
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever(
            embedder=_embedder,
            vector_store=_get_store(),
            keyword_index=_keyword_index,
            reranker=_reranker,
            enable_reranking=ENABLE_RERANKING,
        )
    return _retriever


# ---------------------------------------------------------------------------
# Public API (dict-based for backward compatibility with agent.py / main.py)
# ---------------------------------------------------------------------------


def embed_and_store(chunks: List[dict], session_id: str) -> None:
    """
    Embed all chunks and store them in a fresh per-session Qdrant collection,
    then build the session's BM25 keyword index.

    Args:
        chunks:     list of {"text", "page", "chunk_index", "section"} dicts.
        session_id: UUID string identifying this upload session.

    Raises:
        ValueError:             chunk list is empty.
        ConfigError:            QDRANT_URL is not set.
        QdrantUnavailableError: Qdrant is unreachable.
    """
    if not chunks:
        raise ValueError("chunk list is empty — nothing to embed")

    store = _get_store()

    # Drop stale collection if it exists, then create fresh.
    store.create_collection(session_id, VECTOR_SIZE)

    texts = [c["text"] for c in chunks]
    print(f"[embeddings] Embedding {len(texts)} chunk(s) for session '{session_id}'…")

    embeddings = _embedder.encode(texts)

    ids = list(range(len(chunks)))
    vectors = [embeddings[i].tolist() for i in range(len(chunks))]
    payloads = [
        {
            "text":        chunks[i]["text"],
            "page":        chunks[i]["page"],
            "chunk_index": chunks[i]["chunk_index"],
            "section":     chunks[i].get("section", "Unknown"),
        }
        for i in range(len(chunks))
    ]

    # Single upsert — all PDF sessions have O(100) chunks, well within one batch.
    store.upsert(session_id, ids, vectors, payloads)

    del embeddings
    gc.collect()

    print(f"[embeddings] Stored {len(chunks)} chunk(s) in Qdrant collection '{session_id}'.")

    # Build the per-session BM25 keyword index so retrieval can fuse semantic
    # similarity with exact-term matching.
    _keyword_index.build(session_id, [Chunk.from_dict(c) for c in chunks])


def retrieve_relevant_chunks(
    query: str,
    session_id: str,
    top_k: int = 10,
    original_query: Optional[str] = None,
) -> List[dict]:
    """
    Retrieve the most relevant chunks (hybrid + rerank) as dicts.

    Returns [] when nothing clears the relevance threshold (caller refuses).
    Each dict carries text/page/chunk_index/section/score plus the
    semantic/bm25/fused/rerank scores.
    """
    results = _get_retriever().retrieve(query, session_id, top_k=top_k, original_query=original_query)
    return [r.to_dict() for r in results]


def retrieve_multi_query(
    queries: List[str],
    session_id: str,
    top_k: int = 10,
    original_query: Optional[str] = None,
) -> List[dict]:
    """
    Multi-query retrieval: run several formulations, merge by chunk_index, rerank
    once. Returns dicts (same schema as retrieve_relevant_chunks).
    """
    results = _get_retriever().retrieve_multi(queries, session_id, top_k=top_k, original_query=original_query)
    return [r.to_dict() for r in results]


def get_last_retrieval_debug(session_id: str) -> Optional[dict]:
    """Return the most recent retrieval diagnostics for a session, or None."""
    return _retriever.get_last_debug(session_id) if _retriever is not None else None


def delete_session(session_id: str) -> None:
    """
    Drop the Qdrant collection and in-memory indexes for this session.

    Safe to call even if the session does not exist.

    Raises:
        QdrantUnavailableError: Qdrant is unreachable (caller may treat as
                                best-effort cleanup).
    """
    store = _get_store()

    # Drop in-memory state regardless of Qdrant state.
    _keyword_index.drop(session_id)
    if _retriever is not None:
        _retriever.drop_debug(session_id)

    store.delete_collection(session_id)
    print(f"[embeddings] Deleted Qdrant collection '{session_id}'.")


# ---------------------------------------------------------------------------
# Startup / health utilities
# ---------------------------------------------------------------------------


def warmup() -> None:
    """
    Construct the Qdrant client and probe connectivity at startup — no transformer
    models (those lazy-load on first use to keep startup memory low).

    Raises ConfigError / QdrantUnavailableError on failure; the caller
    (main.lifespan) logs without crashing so requests fail fast with
    "Qdrant server not working".
    """
    print("[embeddings] Warmup: initializing Qdrant client only (models load lazily).")
    _get_store().warmup()
    print("[embeddings] Warmup complete — Qdrant reachable; no transformer models loaded at startup.")


def get_status() -> dict:
    """Initialization state of singletons. Used by /health/debug — never raises."""
    return {
        "model_loaded":       _embedder.is_loaded,
        "client_initialized": _store is not None and _store.is_initialized,
    }


def check_qdrant_connectivity() -> bool:
    """Lightweight Qdrant probe. True if reachable, False otherwise."""
    if _store is None:
        return False
    return _store.ping()
