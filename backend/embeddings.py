"""
embeddings.py — Embedding generation and in-memory vector store.

Replaces ChromaDB with exact numpy cosine similarity search.
For the collection sizes involved (tens to low hundreds of chunks per PDF),
exact search is both faster and more accurate than HNSW approximation, and
uses a fraction of the memory (no C++ index, no ONNX runtime, no SQLite).

Memory per session: ~70 KB (46 chunks × 384 dims × float32).
The sentence-transformer model (~90 MB) is loaded once and shared across sessions.
"""

from __future__ import annotations

import gc
from typing import Any, Dict, List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Module-level singletons
# ---------------------------------------------------------------------------

_model: Optional[SentenceTransformer] = None

# session_id → {"embeddings": np.ndarray shape (N, D), "chunks": List[dict]}
_store: Dict[str, Dict[str, Any]] = {}


def _get_model() -> SentenceTransformer:
    """Return the shared sentence-transformer model, loading it on first call."""
    global _model
    if _model is None:
        print("[embeddings] Loading sentence-transformers model (all-MiniLM-L6-v2)…")
        _model = SentenceTransformer("all-MiniLM-L6-v2")
        print("[embeddings] Model loaded.")
    return _model


def _normalise(vecs: np.ndarray) -> np.ndarray:
    """L2-normalise rows so dot product equals cosine similarity."""
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return vecs / norms


def _encode_in_batches(model: SentenceTransformer, texts: List[str], batch_size: int = 8) -> np.ndarray:
    """Encode texts in small batches with gc between each. Returns (N, D) float32 array."""
    batches = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        embs = model.encode(batch, show_progress_bar=False, convert_to_numpy=True)
        batches.append(embs.astype(np.float32))
        gc.collect()
    return np.vstack(batches)


# ---------------------------------------------------------------------------
# Public API — identical signatures to the ChromaDB version
# ---------------------------------------------------------------------------


def embed_and_store(chunks: List[dict], session_id: str) -> None:
    """
    Embed all chunks and store them in the in-memory session store.

    Args:
        chunks:     Output of pdf_processor.parse_and_chunk() —
                    list of {"text", "page", "chunk_index"} dicts.
        session_id: UUID string identifying this upload session.
    """
    if not chunks:
        raise ValueError("chunk list is empty — nothing to embed")

    model = _get_model()
    texts = [c["text"] for c in chunks]

    print(f"[embeddings] Embedding {len(texts)} chunk(s) for session '{session_id}'…")
    raw = _encode_in_batches(model, texts)
    normalised = _normalise(raw)
    del raw
    gc.collect()

    _store[session_id] = {"embeddings": normalised, "chunks": list(chunks)}
    print(f"[embeddings] Stored {len(chunks)} chunk(s) for session '{session_id}'.")


def retrieve_relevant_chunks(
    query: str,
    session_id: str,
    top_k: int = 10,
) -> List[dict]:
    """
    Retrieve the top-k most relevant chunks for a query using exact cosine similarity.

    Args:
        query:      The user's natural-language question.
        session_id: UUID string identifying the active upload session.
        top_k:      Number of chunks to return.

    Returns:
        List of dicts sorted by relevance (most relevant first):
            [{"text": str, "page": int, "chunk_index": int, "score": float}, ...]
        Returns empty list if session does not exist.
    """
    if session_id not in _store:
        print(f"[embeddings] Session '{session_id}' not found — returning empty results.")
        return []

    model = _get_model()
    entry = _store[session_id]
    stored_embs: np.ndarray = entry["embeddings"]  # (N, D), already normalised
    chunks: List[dict] = entry["chunks"]

    query_vec = model.encode([query], show_progress_bar=False, convert_to_numpy=True).astype(np.float32)
    query_vec = _normalise(query_vec)  # (1, D)

    scores = (stored_embs @ query_vec.T).squeeze()  # (N,)
    if scores.ndim == 0:
        scores = scores.reshape(1)

    k = min(top_k, len(chunks))
    # argpartition is O(N) — faster than full sort for large N
    top_indices = np.argpartition(scores, -k)[-k:]
    top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

    results = []
    for idx in top_indices:
        c = chunks[int(idx)]
        results.append({
            "text":        c["text"],
            "page":        c["page"],
            "chunk_index": c["chunk_index"],
            "score":       round(float(scores[idx]), 4),
        })

    print(
        f"[embeddings] Retrieved {len(results)} chunk(s); "
        f"top score={results[0]['score'] if results else 'n/a'}."
    )
    return results


def delete_session(session_id: str) -> None:
    """
    Remove a session's vectors from memory.

    Args:
        session_id: UUID string identifying the session to remove.
    """
    if session_id not in _store:
        print(f"[embeddings] Session '{session_id}' not found — nothing to delete.")
        return
    del _store[session_id]
    gc.collect()
    print(f"[embeddings] Deleted session '{session_id}'.")
