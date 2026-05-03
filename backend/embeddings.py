"""
embeddings.py — Embedding generation and ChromaDB vector store operations.

Module-level singletons are used for both the sentence-transformer model and
the ChromaDB client so they are initialised once per process, not per request.

Each uploaded PDF is stored in its own ChromaDB collection identified by
session_id, keeping sessions fully isolated with zero cross-contamination.
"""

from __future__ import annotations  # makes all annotations lazy strings — never evaluated at runtime

import gc
import os
from typing import Any, List

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Module-level singletons — initialised lazily on first use
# ---------------------------------------------------------------------------

_model: SentenceTransformer | None = None
_chroma_client: Any = None  # chromadb.PersistentClient is a factory fn, not a class


def _get_model() -> SentenceTransformer:
    """Return the shared sentence-transformer model, loading it on first call."""
    global _model
    if _model is None:
        print("[embeddings] Loading sentence-transformers model (all-MiniLM-L6-v2)…")
        _model = SentenceTransformer("all-MiniLM-L6-v2")
        print("[embeddings] Model loaded.")
    return _model


def _get_chroma_client() -> Any:
    """Return the shared ChromaDB client, creating it on first call."""
    global _chroma_client
    if _chroma_client is None:
        persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
        print(f"[embeddings] Initialising ChromaDB client at '{persist_dir}'…")
        _chroma_client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        print("[embeddings] ChromaDB client ready.")
    return _chroma_client


def _encode_in_batches(model: SentenceTransformer, texts: List[str], batch_size: int = 8) -> List[List[float]]:
    """Encode texts in small batches with gc between each to keep peak memory low."""
    all_embeddings: List[List[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        embs = model.encode(batch, show_progress_bar=False, convert_to_numpy=True)
        all_embeddings.extend(embs.tolist())
        del embs
        gc.collect()
    return all_embeddings


def _collection_name(session_id: str) -> str:
    """
    Derive a valid ChromaDB collection name from a session_id.

    ChromaDB collection names must be 3-63 chars, start/end with a
    lowercase letter or digit, and contain only [a-z0-9_-].
    UUIDs with hyphens already satisfy this, but we strip hyphens and
    add a stable prefix to guarantee it starts with a letter.
    """
    return "s" + session_id.replace("-", "")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def embed_and_store(chunks: List[dict], session_id: str) -> None:
    """
    Embed all chunks and persist them in a per-session ChromaDB collection.

    Creates the collection if it doesn't exist (idempotent upsert semantics),
    so re-uploading the same session_id safely overwrites previous data.

    Args:
        chunks:     Output of pdf_processor.chunk_pages() —
                    list of {"text", "page", "chunk_index"} dicts.
        session_id: UUID string identifying this upload session.
    """
    if not chunks:
        raise ValueError("chunk list is empty — nothing to embed")

    model = _get_model()
    client = _get_chroma_client()
    col_name = _collection_name(session_id)

    # Cosine similarity matches what sentence-transformers is trained for
    collection = client.get_or_create_collection(
        name=col_name,
        metadata={"hnsw:space": "cosine"},
    )

    texts = [c["text"] for c in chunks]
    print(f"[embeddings] Embedding {len(texts)} chunk(s) for session '{session_id}'…")
    embeddings = _encode_in_batches(model, texts, batch_size=8)

    ids = [f"{session_id}_{c['chunk_index']}" for c in chunks]
    metadatas = [{"page": c["page"], "chunk_index": c["chunk_index"]} for c in chunks]

    # upsert so duplicate uploads don't cause primary-key conflicts
    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
    )
    del embeddings, texts
    gc.collect()

    print(f"[embeddings] Stored {len(chunks)} chunk(s) in collection '{col_name}'.")


def retrieve_relevant_chunks(
    query: str,
    session_id: str,
    top_k: int = 10,
) -> List[dict]:
    """
    Retrieve the top-k most relevant chunks for a user query.

    Embeds the query with the same model used at index time, then runs
    an approximate nearest-neighbour search in the session's collection.

    Args:
        query:      The user's natural-language question.
        session_id: UUID string identifying the active upload session.
        top_k:      Number of chunks to return (default 5).

    Returns:
        List of dicts sorted by relevance (most relevant first):
            [{"text": str, "page": int, "chunk_index": int, "score": float}, ...]
        score is cosine similarity in [0, 1] (higher = more relevant).
        Returns an empty list if the session collection does not exist.
    """
    model = _get_model()
    client = _get_chroma_client()
    col_name = _collection_name(session_id)

    # Gracefully handle missing session
    existing = [c.name for c in client.list_collections()]
    if col_name not in existing:
        print(f"[embeddings] Collection '{col_name}' not found — returning empty results.")
        return []

    collection = client.get_collection(name=col_name)
    n_docs = collection.count()
    if n_docs == 0:
        return []

    # Clamp top_k to the number of stored documents
    k = min(top_k, n_docs)

    print(f"[embeddings] Retrieving top-{k} chunk(s) for session '{session_id}'…")
    query_embedding = model.encode([query], show_progress_bar=False).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )

    # ChromaDB returns nested lists (one row per query); we sent one query
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    # distances are in [0, 2] for cosine space (distance = 1 - similarity)
    distances = results["distances"][0]

    chunks = []
    for doc, meta, dist in zip(docs, metas, distances):
        chunks.append(
            {
                "text": doc,
                "page": meta["page"],
                "chunk_index": meta["chunk_index"],
                "score": round(1.0 - dist, 4),  # convert distance → similarity
            }
        )

    print(
        f"[embeddings] Retrieved {len(chunks)} chunk(s); "
        f"top score={chunks[0]['score'] if chunks else 'n/a'}."
    )
    return chunks


def delete_session(session_id: str) -> None:
    """
    Delete the ChromaDB collection for a session, freeing its vector data.

    Safe to call even if the session never existed or was already deleted.

    Args:
        session_id: UUID string identifying the session to remove.
    """
    client = _get_chroma_client()
    col_name = _collection_name(session_id)

    existing = [c.name for c in client.list_collections()]
    if col_name not in existing:
        print(f"[embeddings] Collection '{col_name}' does not exist — nothing to delete.")
        return

    client.delete_collection(name=col_name)
    print(f"[embeddings] Deleted collection '{col_name}' for session '{session_id}'.")
