"""
embeddings.py — Thin facade over the retrieval stack (container-backed).

All collaborators are owned by app.container (the composition root); this module
just exposes the stable, dict-based API used by main.py and evaluation.py:

    embed_and_store / retrieve_relevant_chunks / retrieve_multi_query /
    delete_session / warmup / get_status / check_qdrant_connectivity /
    get_last_retrieval_debug

A running Qdrant server is mandatory — QDRANT_URL must be set. There is no
in-memory fallback: if Qdrant is unconfigured or unreachable, the vector store
raises ConfigError / QdrantUnavailableError, surfaced by the API as
"Qdrant server not working".
"""

from __future__ import annotations

import gc
import uuid
from typing import List, Optional, Set

from app import container
from app.config import VECTOR_SIZE
from app.domain.models import Chunk


def embed_and_store(chunks: List[dict], session_id: str) -> None:
    """
    Embed all chunks into a fresh per-session Qdrant collection, then build the
    session's BM25 keyword index.

    Raises:
        ValueError:             chunk list is empty.
        ConfigError:            QDRANT_URL is not set.
        QdrantUnavailableError: Qdrant is unreachable.
    """
    if not chunks:
        raise ValueError("chunk list is empty — nothing to embed")

    store = container.store()
    store.create_collection(session_id, VECTOR_SIZE)

    texts = [c["text"] for c in chunks]
    print(f"[embeddings] Embedding {len(texts)} chunk(s) for session '{session_id}'…")
    embeddings = container.embedder().encode(texts)

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
    store.upsert(session_id, ids, vectors, payloads)

    del embeddings
    gc.collect()
    print(f"[embeddings] Stored {len(chunks)} chunk(s) in Qdrant collection '{session_id}'.")

    container.keyword_index().build(session_id, [Chunk.from_dict(c) for c in chunks])


def retrieve_relevant_chunks(
    query: str,
    session_id: str,
    top_k: int = 10,
    original_query: Optional[str] = None,
) -> List[dict]:
    """Hybrid + rerank retrieval as dicts; [] when nothing clears the threshold."""
    results = container.retriever().retrieve(query, session_id, top_k=top_k, original_query=original_query)
    return [r.to_dict() for r in results]


def retrieve_multi_query(
    queries: List[str],
    session_id: str,
    top_k: int = 10,
    original_query: Optional[str] = None,
) -> List[dict]:
    """Multi-query retrieval (merge + single rerank) as dicts."""
    results = container.retriever().retrieve_multi(queries, session_id, top_k=top_k, original_query=original_query)
    return [r.to_dict() for r in results]


def get_last_retrieval_debug(session_id: str) -> Optional[dict]:
    """Most recent retrieval diagnostics for a session, or None."""
    return container.retriever().get_last_debug(session_id)


def delete_session(session_id: str) -> None:
    """
    Drop the Qdrant collection and in-memory indexes/diagnostics for a session.

    Raises:
        QdrantUnavailableError: Qdrant is unreachable (caller may treat as
                                best-effort cleanup).
    """
    container.keyword_index().drop(session_id)
    container.retriever().drop_debug(session_id)
    container.store().delete_collection(session_id)
    print(f"[embeddings] Deleted Qdrant collection '{session_id}'.")


def purge_orphan_collections(keep_ids: Set[str]) -> int:
    """
    Delete session collections (UUID-named) that no live session references.

    Because session metadata lives in process memory, after a restart there are
    no live sessions and every existing collection is orphaned (unreachable via
    the API). Sweeping them prevents unbounded Qdrant growth. Only UUID-named
    collections are touched, so any non-session collection on the cluster is left
    alone. Returns the number of collections deleted.

    Raises:
        ConfigError / QdrantUnavailableError: if Qdrant is unconfigured/unreachable.
    """
    store = container.store()
    deleted = 0
    for name in store.list_collections():
        if name in keep_ids:
            continue
        try:
            uuid.UUID(str(name))  # only sweep session-style (UUID) collections
        except ValueError:
            continue
        store.delete_collection(name)
        deleted += 1
    if deleted:
        print(f"[embeddings] Purged {deleted} orphan collection(s) with no live session.")
    return deleted


def warmup() -> None:
    """
    Construct the Qdrant client and probe connectivity at startup — no transformer
    models (those lazy-load on first use to keep startup memory low).

    Raises ConfigError / QdrantUnavailableError on failure; the caller logs
    without crashing so requests fail fast with "Qdrant server not working".
    """
    print("[embeddings] Warmup: initializing Qdrant client only (models load lazily).")
    container.store().warmup()
    print("[embeddings] Warmup complete — Qdrant reachable; no transformer models loaded at startup.")


def get_status() -> dict:
    """Initialization state of singletons. Used by /health/debug — never raises."""
    return {
        "model_loaded":       container.embedder().is_loaded,
        "client_initialized": container.store().is_initialized,
    }


def check_qdrant_connectivity() -> bool:
    """Lightweight Qdrant probe. True if reachable, False otherwise."""
    return container.store().ping()
