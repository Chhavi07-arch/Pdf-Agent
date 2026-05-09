"""
embeddings.py — Embedding generation and Qdrant vector store.

Replaces the previous NumPy in-memory store with Qdrant.

Connection mode (chosen at startup via env vars, lazy-initialized on first use):
  • QDRANT_URL + QDRANT_API_KEY set → Qdrant Cloud
      Vectors are stored remotely; the Render process holds only the
      embedding model (~90 MB), eliminating all per-session RAM overhead.
  • Neither set → in-memory Qdrant (:memory:)
      Same ephemeral behaviour as the old NumPy store, but with full
      Qdrant semantics (HNSW indexing, typed payloads, cosine search).
      Useful for local development without a cloud account.

Collection strategy: one Qdrant collection per session_id.
  • session_id (UUID) becomes the collection name directly.
  • Natural session isolation — queries need no metadata filter.
  • Deletion is a single atomic drop-collection call.
  • No shared state between sessions at the Qdrant level.
"""

from __future__ import annotations

import gc
import os
from typing import List, Optional

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VECTOR_SIZE = 384  # all-MiniLM-L6-v2 output dimensionality

# Default minimum cosine similarity score for a chunk to be included in results.
# Chunks scoring below this threshold are filtered before the LLM sees them.
# Rationale:
#   • Clearly relevant content scores 0.40–0.90
#   • Weakly related content scores 0.20–0.40 (may still provide useful context)
#   • Essentially unrelated content scores < 0.20 (vocabulary overlap only)
# Override via MIN_RETRIEVAL_SCORE env var for tuning without code changes.
_MIN_SCORE_DEFAULT = 0.20

# ---------------------------------------------------------------------------
# Module-level singletons (lazy-initialized on first use)
# ---------------------------------------------------------------------------

_model: Optional[SentenceTransformer] = None
_client: Optional[QdrantClient] = None


def _get_model() -> SentenceTransformer:
    """Return the shared sentence-transformer model, loading it on first call."""
    global _model
    if _model is None:
        print("[embeddings] Loading sentence-transformers model (all-MiniLM-L6-v2)…")
        _model = SentenceTransformer("all-MiniLM-L6-v2")
        print("[embeddings] Model loaded.")
    return _model


def _get_client() -> QdrantClient:
    """
    Return the shared Qdrant client, initializing it on first call.

    Checks QDRANT_URL and QDRANT_API_KEY environment variables:
      • Both present → connects to Qdrant Cloud (persistent, remote storage).
      • Not set      → creates an in-memory Qdrant instance (ephemeral, local).

    The singleton is initialized once per process; restart to pick up new env vars.
    """
    global _client
    if _client is None:
        url = os.getenv("QDRANT_URL", "").strip()
        api_key = os.getenv("QDRANT_API_KEY", "").strip()

        if url:
            print(f"[embeddings] Connecting to Qdrant Cloud at {url!r}…")
            _client = QdrantClient(
                url=url,
                api_key=api_key or None,
                timeout=30,
            )
            print("[embeddings] Qdrant Cloud client ready.")
        else:
            print("[embeddings] QDRANT_URL not set — using in-memory Qdrant.")
            _client = QdrantClient(":memory:")
            print("[embeddings] In-memory Qdrant client ready.")
    return _client


def _encode_in_batches(
    model: SentenceTransformer,
    texts: List[str],
    batch_size: int = 8,
) -> np.ndarray:
    """Encode texts in small batches with GC between each. Returns (N, D) float32 array."""
    batches = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        embs = model.encode(batch, show_progress_bar=False, convert_to_numpy=True)
        batches.append(embs.astype(np.float32))
        gc.collect()
    return np.vstack(batches)


# ---------------------------------------------------------------------------
# Public API — same signatures as the old NumPy version for drop-in compatibility
# ---------------------------------------------------------------------------


def embed_and_store(chunks: List[dict], session_id: str) -> None:
    """
    Embed all chunks and store them in a per-session Qdrant collection.

    Creates a fresh collection named after the session_id using cosine
    distance. Each Qdrant point carries the chunk text, page number, and
    global chunk index as payload.

    If a collection with this session_id already exists (e.g. from a
    previous upload attempt), it is dropped first so the new data is clean.

    Args:
        chunks:     Output of pdf_processor.parse_and_chunk() —
                    list of {"text", "page", "chunk_index"} dicts.
        session_id: UUID string identifying this upload session.
    """
    if not chunks:
        raise ValueError("chunk list is empty — nothing to embed")

    client = _get_client()
    model = _get_model()

    # Drop stale collection if it exists, then create fresh.
    if client.collection_exists(collection_name=session_id):
        client.delete_collection(collection_name=session_id)

    client.create_collection(
        collection_name=session_id,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )

    texts = [c["text"] for c in chunks]
    print(f"[embeddings] Embedding {len(texts)} chunk(s) for session '{session_id}'…")

    embeddings = _encode_in_batches(model, texts)

    points = [
        PointStruct(
            id=i,
            vector=embeddings[i].tolist(),
            payload={
                "text":        chunks[i]["text"],
                "page":        chunks[i]["page"],
                "chunk_index": chunks[i]["chunk_index"],
            },
        )
        for i in range(len(chunks))
    ]

    # Single upsert — all PDF sessions have O(100) chunks, well within one batch.
    client.upsert(collection_name=session_id, points=points)

    del embeddings
    gc.collect()

    print(f"[embeddings] Stored {len(chunks)} chunk(s) in Qdrant collection '{session_id}'.")


def retrieve_relevant_chunks(
    query: str,
    session_id: str,
    top_k: int = 10,
) -> List[dict]:
    """
    Retrieve the top-k most relevant chunks via Qdrant cosine similarity search,
    then filter out chunks below the minimum similarity threshold.

    Qdrant's Distance.COSINE returns cosine similarity scores in [−1, 1].
    For sentence-transformer embeddings, practically all scores fall in [0, 0.9]:
      • > 0.50  clearly relevant
      • 0.20–0.50  related / weakly associated (included)
      • < 0.20  essentially unrelated (filtered out)

    The threshold is read from the MIN_RETRIEVAL_SCORE environment variable on
    every call (default 0.20), so it can be tuned without restarting the server.

    If score filtering removes ALL candidates, an empty list is returned.
    The caller (agent.py) already handles this by retrying with the raw query
    and issuing a refusal if that also returns nothing — no new failure paths.

    Args:
        query:      The user's natural-language question.
        session_id: UUID string identifying the active upload session.
        top_k:      Number of candidates to fetch from Qdrant before filtering.
                    Intentionally higher than the LLM context budget so the
                    filter has room to discard low-quality candidates.

    Returns:
        List of dicts sorted by relevance (most relevant first):
            [{"text": str, "page": int, "chunk_index": int, "score": float}, ...]
        Returns empty list if the session collection does not exist or all
        candidates fall below the minimum score threshold.
    """
    client = _get_client()

    if not client.collection_exists(collection_name=session_id):
        print(f"[embeddings] Session '{session_id}' not found — returning empty results.")
        return []

    # Read threshold on every call — allows runtime tuning via env var.
    min_score = float(os.getenv("MIN_RETRIEVAL_SCORE", str(_MIN_SCORE_DEFAULT)))

    model = _get_model()
    query_vec: list[float] = (
        model.encode([query], show_progress_bar=False, convert_to_numpy=True)[0]
        .astype(np.float32)
        .tolist()
    )

    hits = client.search(
        collection_name=session_id,
        query_vector=query_vec,
        limit=top_k,
        with_payload=True,
    )

    results = []
    filtered_count = 0
    for hit in hits:
        if hit.score >= min_score:
            results.append(
                {
                    "text":        hit.payload["text"],
                    "page":        hit.payload["page"],
                    "chunk_index": hit.payload["chunk_index"],
                    "score":       round(hit.score, 4),
                }
            )
        else:
            filtered_count += 1

    if filtered_count:
        print(
            f"[embeddings] Score filter (min={min_score}): "
            f"kept {len(results)}, dropped {filtered_count} chunk(s)."
        )

    print(
        f"[embeddings] Retrieved {len(results)} chunk(s); "
        f"top score={results[0]['score'] if results else 'n/a'}."
    )
    return results


def delete_session(session_id: str) -> None:
    """
    Drop the Qdrant collection for this session, freeing remote/local storage.

    Safe to call even if the session does not exist (no-op with a log message).

    Args:
        session_id: UUID string identifying the session to remove.
    """
    client = _get_client()

    if not client.collection_exists(collection_name=session_id):
        print(f"[embeddings] Session '{session_id}' not found — nothing to delete.")
        return

    client.delete_collection(collection_name=session_id)
    print(f"[embeddings] Deleted Qdrant collection '{session_id}'.")


# ---------------------------------------------------------------------------
# Startup utilities
# ---------------------------------------------------------------------------


def warmup() -> None:
    """
    Pre-initialize the embedding model and Qdrant client.

    Called during application startup so the first upload does not bear the
    cold-start penalty of loading the ~90 MB sentence-transformer model and
    establishing the Qdrant connection.  Both operations are idempotent —
    subsequent calls return the already-initialized singletons immediately.
    """
    print("[embeddings] Warmup: loading model and Qdrant client…")
    _get_model()
    _get_client()
    print("[embeddings] Warmup complete.")


def get_status() -> dict:
    """
    Return the initialization state of module-level singletons.
    Used by the /health/debug endpoint — never raises.
    """
    return {
        "model_loaded":      _model is not None,
        "client_initialized": _client is not None,
    }


def check_qdrant_connectivity() -> bool:
    """
    Probe Qdrant with a lightweight list-collections call.
    Returns True if the call succeeds, False on any error.
    """
    if _client is None:
        return False
    try:
        _client.get_collections()
        return True
    except Exception as exc:
        print(f"[embeddings] Qdrant connectivity check failed: {exc!r}")
        return False
