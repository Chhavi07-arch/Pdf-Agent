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
import re
import time
from typing import Dict, List, Optional

# transformers (pulled in by sentence-transformers) auto-imports its TensorFlow
# backend when TF is present in the environment. On Keras 3 that import crashes
# ("install tf-keras"). This app is PyTorch-only, so disable the TF/Flax paths
# BEFORE importing sentence_transformers — this both prevents the crash and
# avoids loading TensorFlow into memory. Must be set before the import below;
# it cannot live in .env because main.py loads .env after importing this module.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VECTOR_SIZE = 384  # all-MiniLM-L6-v2 output dimensionality

# TODO:
# Qdrant currently serves as the vector store.
#
# If future requirements demand a relational database solution, migration can be
# performed using pgvector by replacing vector insert/search operations while
# preserving retrieval interfaces.

# Default minimum cosine similarity score for a chunk to be included in results.
# Chunks scoring below this threshold are filtered before the LLM sees them.
# Rationale:
#   • Clearly relevant content scores 0.40–0.90
#   • Weakly related content scores 0.20–0.40 (may still provide useful context)
#   • Essentially unrelated content scores < 0.20 (vocabulary overlap only)
# Override via MIN_RETRIEVAL_SCORE env var for tuning without code changes.
_MIN_SCORE_DEFAULT = 0.20

# ---------------------------------------------------------------------------
# Hybrid retrieval (BM25 + semantic) configuration
# ---------------------------------------------------------------------------

# Number of candidates pulled from EACH retriever (semantic + BM25) before
# fusion, and the size of the fused candidate pool returned by _hybrid_search.
# Intentionally larger than the LLM context budget so fusion (and, in Step 2,
# the cross-encoder reranker) has room to reorder before the final trim.
_CANDIDATE_POOL = 20

# Score-fusion weights. Semantic similarity is weighted higher than keyword
# overlap because the embedding model captures paraphrase/synonymy that BM25
# misses; BM25 contributes precision on exact term matches the embedding blurs.
_SEMANTIC_WEIGHT = 0.6
_BM25_WEIGHT = 0.4

# Cross-encoder reranker (Step 2). The 20-candidate hybrid pool is re-scored by
# this model and trimmed to the best _RERANK_TOP_K, which both improves
# precision and shrinks the LLM prompt. Loaded lazily (NOT at startup).
_CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_RERANK_TOP_K = 7

# Cross-encoder reranking is the single largest memory cost (a second transformer
# model loaded into RAM). On low-memory hosts (e.g. Render free tier, 512 MB) set
# ENABLE_RERANKING=false: the CrossEncoder is never loaded and retrieval returns
# the hybrid (BM25 + semantic) results directly. Default true preserves full
# answer quality on hosts with enough RAM.
ENABLE_RERANKING = os.getenv("ENABLE_RERANKING", "true").lower() == "true"

# ---------------------------------------------------------------------------
# Retrieval debug / observability (Phase 2.5)
# ---------------------------------------------------------------------------

# When True, every retrieval prints a consolidated [retrieval] diagnostics block
# (candidate counts, top scores, per-chunk semantic/bm25/fused/rerank scores).
# Observability only — does not affect retrieval behavior.
DEBUG_RETRIEVAL = True

# Latest diagnostics per session, captured on every retrieve_relevant_chunks
# call regardless of DEBUG_RETRIEVAL, so GET /debug/retrieval/{id} can serve it.
#   session_id -> diagnostics dict
_last_retrieval_debug: Dict[str, dict] = {}

# Side-channel stats written by _hybrid_search on each call (candidate counts +
# hybrid top score) so retrieve_relevant_chunks can include them in diagnostics
# without changing _hybrid_search's return contract.
_last_hybrid_stats: Dict[str, object] = {}

# ---------------------------------------------------------------------------
# Module-level singletons (lazy-initialized on first use)
# ---------------------------------------------------------------------------

_model: Optional[SentenceTransformer] = None
_client: Optional[QdrantClient] = None
_cross_encoder: Optional[CrossEncoder] = None

# Per-session BM25 keyword index, built during embed_and_store().
#   session_id -> {"bm25": BM25Okapi, "chunks": List[dict]}
# Held in process memory only. With Qdrant Cloud, vectors persist across a
# restart but this dict does not — retrieval falls back to semantic-only for
# any pre-restart session (handled gracefully in _bm25_search).
_bm25_indexes: Dict[str, dict] = {}


def _tokenize(text: str) -> List[str]:
    """Lowercase alphanumeric word tokenizer used for BM25 indexing/querying."""
    return re.findall(r"[a-z0-9]+", text.lower())


def _minmax_normalize(score_map: Dict[int, float]) -> Dict[int, float]:
    """
    Min-max normalize a {chunk_index: score} map into the range [0, 1].

    Each retriever's raw scores live on different scales (cosine ∈ [−1,1],
    BM25 ∈ [0, ∞)), so both are independently normalized before fusion.
    If all scores are equal (or there is a single candidate), every entry maps
    to 1.0 — they are equally relevant within that retriever.
    """
    if not score_map:
        return {}
    values = list(score_map.values())
    lo, hi = min(values), max(values)
    if hi - lo < 1e-9:
        return {k: 1.0 for k in score_map}
    return {k: (v - lo) / (hi - lo) for k, v in score_map.items()}


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


def get_cross_encoder() -> CrossEncoder:
    """
    Return the shared cross-encoder reranker, loading it on first call.

    Loaded lazily on the first reranked query — never at application startup —
    so cold-start (and the upload critical path) does not pay for this model.
    """
    global _cross_encoder
    if _cross_encoder is None:
        print("[embeddings] Loading cross encoder...")
        _cross_encoder = CrossEncoder(_CROSS_ENCODER_MODEL)
        print("[embeddings] Cross encoder loaded")
    return _cross_encoder


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
                "section":     chunks[i].get("section", "Unknown"),
            },
        )
        for i in range(len(chunks))
    ]

    # Single upsert — all PDF sessions have O(100) chunks, well within one batch.
    client.upsert(collection_name=session_id, points=points)

    del embeddings
    gc.collect()

    print(f"[embeddings] Stored {len(chunks)} chunk(s) in Qdrant collection '{session_id}'.")

    # Build the per-session BM25 keyword index alongside the vector store, so
    # retrieval can fuse semantic similarity with exact-term matching.
    _build_bm25_index(chunks, session_id)


def _build_bm25_index(chunks: List[dict], session_id: str) -> None:
    """
    Build and cache an in-memory BM25Okapi index for a session.

    Stores a lightweight copy of each chunk's payload (text, page, chunk_index,
    section) parallel to the tokenized corpus so BM25 hits can be returned in
    the same shape as semantic hits.
    """
    corpus_tokens = [_tokenize(c["text"]) for c in chunks]
    bm25 = BM25Okapi(corpus_tokens)
    _bm25_indexes[session_id] = {
        "bm25": bm25,
        "chunks": [
            {
                "text":        c["text"],
                "page":        c["page"],
                "chunk_index": c["chunk_index"],
                "section":     c.get("section", "Unknown"),
            }
            for c in chunks
        ],
    }
    print(f"[embeddings] Built BM25 index for session '{session_id}' ({len(chunks)} docs).")


def _semantic_search(query: str, session_id: str, limit: int) -> List[dict]:
    """
    Semantic retrieval via Qdrant cosine search (unchanged from the original
    implementation, just factored out). Each result's "score" is the raw cosine
    similarity in [−1, 1] — this is the value agent.py's confidence bands rely on.
    """
    client = _get_client()
    if not client.collection_exists(collection_name=session_id):
        print(f"[embeddings] Session '{session_id}' not found — returning empty results.")
        return []

    model = _get_model()
    query_vec: list[float] = (
        model.encode([query], show_progress_bar=False, convert_to_numpy=True)[0]
        .astype(np.float32)
        .tolist()
    )

    hits = client.search(
        collection_name=session_id,
        query_vector=query_vec,
        limit=limit,
        with_payload=True,
    )

    return [
        {
            "text":        h.payload["text"],
            "page":        h.payload["page"],
            "chunk_index": h.payload["chunk_index"],
            "section":     h.payload.get("section", "Unknown"),
            "score":       float(h.score),
        }
        for h in hits
    ]


def _bm25_search(query: str, session_id: str, limit: int) -> List[dict]:
    """
    BM25 keyword retrieval over the per-session index.

    Returns up to `limit` chunks with positive BM25 score, each carrying its raw
    BM25 score under "score". Returns [] (semantic-only fallback) if the session
    has no in-memory BM25 index — e.g. a Qdrant Cloud session that survived a
    process restart.
    """
    entry = _bm25_indexes.get(session_id)
    if not entry:
        print(f"[embeddings] No BM25 index for session '{session_id}' — semantic-only fallback.")
        return []

    tokens = _tokenize(query)
    if not tokens:
        return []

    bm25: BM25Okapi = entry["bm25"]
    chunks: List[dict] = entry["chunks"]
    scores = bm25.get_scores(tokens)  # np.ndarray aligned with chunks

    order = np.argsort(scores)[::-1][:limit]
    results = []
    for i in order:
        if scores[i] <= 0:
            continue  # no term overlap — not a real keyword match
        c = chunks[int(i)]
        results.append({**c, "score": float(scores[i])})
    return results


def _hybrid_search(
    query: str,
    session_id: str,
    pool_size: int = _CANDIDATE_POOL,
) -> List[dict]:
    """
    Hybrid retrieval: fuse semantic (Qdrant) and BM25 keyword results.

    Pipeline:
      1. Semantic top-`pool_size`, gated by MIN_RETRIEVAL_SCORE (preserves the
         existing "nothing relevant → refuse" fast-path: if no semantic chunk
         clears the threshold, this returns [] exactly as before).
      2. BM25 top-`pool_size` over the same session.
      3. Min-max normalize each retriever's scores independently to [0, 1].
      4. Fuse: final = 0.6·semantic_norm + 0.4·bm25_norm. A chunk present in
         only one result set contributes 0 for the missing retriever.
      5. Sort by fused score, return up to `pool_size` candidates.

    Each returned chunk keeps "score" = its semantic cosine (0.0 for BM25-only
    chunks) so agent.py's confidence/refusal logic is unaffected, and adds
    "semantic_score", "bm25_score", "fused_score" (all normalized) for ranking
    and observability. Fused order determines ranking; cosine still determines
    confidence.
    """
    semantic = _semantic_search(query, session_id, pool_size)

    # Gate on the semantic min-score threshold — read per call for live tuning.
    min_score = float(os.getenv("MIN_RETRIEVAL_SCORE", str(_MIN_SCORE_DEFAULT)))
    semantic_kept = [c for c in semantic if c["score"] >= min_score]
    dropped = len(semantic) - len(semantic_kept)
    if dropped:
        print(
            f"[embeddings] Score filter (min={min_score}): "
            f"kept {len(semantic_kept)}, dropped {dropped} semantic chunk(s)."
        )
    print(f"[embeddings] Semantic candidates={len(semantic_kept)}")

    if not semantic_kept:
        # No relevant semantic content → preserve the existing fast-refusal path.
        print("[embeddings] No semantic candidates above threshold — hybrid empty (refusal path).")
        _last_hybrid_stats.clear()
        _last_hybrid_stats.update(
            {"semantic_candidates": 0, "bm25_candidates": 0, "hybrid_top_score": None}
        )
        return []

    bm25 = _bm25_search(query, session_id, pool_size)
    print(f"[embeddings] BM25 candidates={len(bm25)}")

    # Independent normalization of each retriever's raw scores.
    sem_norm = _minmax_normalize({c["chunk_index"]: c["score"] for c in semantic_kept})
    bm25_norm = _minmax_normalize({c["chunk_index"]: c["score"] for c in bm25})

    # Union the two result sets, keeping the semantic cosine as "score".
    by_index: Dict[int, dict] = {}
    for c in semantic_kept:
        by_index[c["chunk_index"]] = {**c}            # score = cosine
    for c in bm25:
        if c["chunk_index"] not in by_index:
            by_index[c["chunk_index"]] = {**c, "score": 0.0}  # BM25-only: no cosine

    fused: List[dict] = []
    for idx, chunk in by_index.items():
        s = sem_norm.get(idx, 0.0)
        b = bm25_norm.get(idx, 0.0)
        final = _SEMANTIC_WEIGHT * s + _BM25_WEIGHT * b
        fused.append({
            **chunk,
            "score":          round(chunk["score"], 4),   # cosine — preserved
            "semantic_score": round(s, 4),
            "bm25_score":     round(b, 4),
            "fused_score":    round(final, 4),
        })

    fused.sort(key=lambda x: x["fused_score"], reverse=True)
    fused = fused[:pool_size]

    print(f"[embeddings] Hybrid candidates={len(fused)}")
    if fused:
        print(f"[embeddings] Hybrid top score={fused[0]['fused_score']}")

    _last_hybrid_stats.clear()
    _last_hybrid_stats.update({
        "semantic_candidates": len(semantic_kept),
        "bm25_candidates":     len(bm25),
        "hybrid_top_score":    fused[0]["fused_score"] if fused else None,
    })
    return fused


def _rerank_candidates(
    query: str,
    candidates: List[dict],
    final_k: int = _RERANK_TOP_K,
) -> List[dict]:
    """
    Re-score hybrid candidates with the cross-encoder and keep the best `final_k`.

    A cross-encoder reads (query, chunk_text) jointly rather than comparing two
    independent embeddings, so it judges relevance far more precisely than the
    bi-encoder used for retrieval — at a cost only affordable on a small
    candidate set, which is exactly what hybrid retrieval has narrowed us to.

    Each surviving chunk gains a "rerank_score" field; all existing fields
    (score, semantic_score, bm25_score, fused_score) are preserved untouched.

    On any failure (model load or predict error) this degrades gracefully to the
    fused order so retrieval never breaks — the chunks then carry their
    fused_score copied into rerank_score for schema consistency.
    """
    if not candidates:
        return []

    print(f"[embeddings] Reranking {len(candidates)} candidates")
    t0 = time.monotonic()
    try:
        cross_encoder = get_cross_encoder()
        pairs = [(query, c["text"]) for c in candidates]
        scores = cross_encoder.predict(pairs)

        reranked = [
            {**c, "rerank_score": round(float(s), 4)}
            for c, s in zip(candidates, scores)
        ]
        reranked.sort(key=lambda x: x["rerank_score"], reverse=True)
        reranked = reranked[:final_k]

        elapsed = time.monotonic() - t0
        print(
            f"[embeddings] Cross encoder top score="
            f"{reranked[0]['rerank_score'] if reranked else 'n/a'}"
        )
        print(f"[embeddings] Returning top {len(reranked)} reranked chunks")
        print(f"[embeddings] Rerank completed in {elapsed:.2f}s")
        return reranked

    except Exception as exc:
        print(
            f"[embeddings] Rerank failed: {exc!r} — falling back to fused order "
            f"(no quality gain this query, but retrieval still works)."
        )
        return [
            {**c, "rerank_score": c.get("fused_score")}
            for c in candidates[:final_k]
        ]


def retrieve_relevant_chunks(
    query: str,
    session_id: str,
    top_k: int = 10,
    original_query: Optional[str] = None,
) -> List[dict]:
    """
    Retrieve the most relevant chunks: hybrid (BM25 + semantic) → cross-encoder rerank.

    Pipeline:
      1. _hybrid_search → up to _CANDIDATE_POOL (20) fused candidates.
      2. _rerank_candidates → cross-encoder re-scores and keeps the best
         _RERANK_TOP_K (7). This both lifts precision and shrinks the LLM prompt.
      3. Return up to `top_k` of the reranked set (default top_k=10 ≥ 7, so the
         full reranked set is returned).

    Behavior contract is unchanged for callers:
      • Returns [] when the session is missing or nothing clears the semantic
        relevance threshold — agent.py's fallback + refusal paths still apply
        (the refusal fast-path is gated inside _hybrid_search, before reranking).
      • Each chunk's "score" remains the semantic cosine similarity, so the
        confidence bands and citation logic in agent.py work as before.

    Args:
        query:      The user's natural-language question (already rewritten /
                    expanded by agent.py — this layer does not alter it).
        session_id: UUID string identifying the active upload session.
        top_k:      Maximum number of chunks to return (default 10).

    Returns:
        List of dicts ordered by rerank relevance (most relevant first):
            [{"text", "page", "chunk_index", "section", "score",
              "semantic_score", "bm25_score", "fused_score", "rerank_score"}, ...]
    """
    candidates = _hybrid_search(query, session_id, pool_size=_CANDIDATE_POOL)

    # Refusal fast-path preserved: empty hybrid → empty result, no rerank.
    if not candidates:
        print("[embeddings] Retrieved 0 chunk(s); top score=n/a.")
        return []

    if ENABLE_RERANKING:
        reranked = _rerank_candidates(query, candidates, final_k=_RERANK_TOP_K)
        results = reranked[:top_k]
    else:
        # Low-memory mode: skip the CrossEncoder, return top hybrid candidates
        # directly (already ordered by fused score) — same final chunk count.
        print("[embeddings] Reranking disabled")
        results = candidates[:_RERANK_TOP_K][:top_k]

    print(
        f"[embeddings] Retrieved {len(results)} chunk(s); "
        f"top score={results[0]['score'] if results else 'n/a'}."
    )

    # Debug-only observability: which sections the returned chunks came from.
    if results:
        top_sections: List[str] = []
        for r in results:
            sec = r.get("section", "Unknown")
            if sec not in top_sections:
                top_sections.append(sec)
        print("[embeddings] Top sections:")
        for sec in top_sections[:5]:
            print(f"  - {sec}")

    # ── Capture consolidated retrieval diagnostics (Phase 2.5) ───────────────
    diagnostics = {
        "query":                original_query if original_query is not None else query,
        "rewritten_query":      query,
        "semantic_candidates":  _last_hybrid_stats.get("semantic_candidates", 0),
        "bm25_candidates":      _last_hybrid_stats.get("bm25_candidates", 0),
        "hybrid_top_score":     _last_hybrid_stats.get("hybrid_top_score"),
        "rerank_top_score":     results[0].get("rerank_score") if results else None,
        "chunks": [
            {
                "rank":           i + 1,
                "page":           r["page"],
                "section":        r.get("section", "Unknown"),
                "semantic_score": r.get("semantic_score"),
                "bm25_score":     r.get("bm25_score"),
                "fused_score":    r.get("fused_score"),
                "rerank_score":   r.get("rerank_score"),
            }
            for i, r in enumerate(results)
        ],
    }
    _last_retrieval_debug[session_id] = diagnostics
    if DEBUG_RETRIEVAL:
        _print_retrieval_debug(diagnostics)

    return results


def _fmt(value) -> str:
    """Format a numeric diagnostic value for the debug block ('n/a' if None)."""
    return f"{value:.2f}" if isinstance(value, (int, float)) else "n/a"


def _print_retrieval_debug(diag: dict) -> None:
    """Print the consolidated [retrieval] diagnostics block (DEBUG_RETRIEVAL)."""
    print("[retrieval]")
    print(f"Query: {diag['query']}")
    if diag["rewritten_query"] != diag["query"]:
        print(f"Rewritten Query: {diag['rewritten_query']}")
    print(f"Semantic Candidates: {diag['semantic_candidates']}")
    print(f"BM25 Candidates: {diag['bm25_candidates']}")
    print(f"Hybrid Top Score: {_fmt(diag['hybrid_top_score'])}")
    print(f"Rerank Top Score: {_fmt(diag['rerank_top_score'])}")
    for c in diag["chunks"]:
        print(f"\n#{c['rank']}")
        print(f"Page: {c['page']}")
        print(f"Section: {c['section']}")
        print(f"Semantic: {_fmt(c['semantic_score'])}")
        print(f"BM25: {_fmt(c['bm25_score'])}")
        print(f"Fused: {_fmt(c['fused_score'])}")
        print(f"Rerank: {_fmt(c['rerank_score'])}")


def get_last_retrieval_debug(session_id: str) -> Optional[dict]:
    """Return the most recent retrieval diagnostics for a session, or None."""
    return _last_retrieval_debug.get(session_id)


def retrieve_multi_query(
    queries: List[str],
    session_id: str,
    top_k: int = 10,
    original_query: Optional[str] = None,
) -> List[dict]:
    """
    Multi-query retrieval (Phase 3 Step 1).

    Runs the existing hybrid search for several query formulations, merges and
    deduplicates the fused candidates (keeping the highest fused score per
    chunk), then applies the SAME cross-encoder reranker ONCE.

    This only ADDS a fan-out + merge layer; _hybrid_search and
    _rerank_candidates are reused unchanged. Reranking uses original_query (the
    user's actual question) so relevance is judged against true intent, not a
    paraphrase. Returns up to top_k reranked chunks, or [] if every query was
    gated out by the semantic threshold (preserving the refusal fast-path).

    Pipeline:
        queries → _hybrid_search (each) → merge/dedup by chunk_index
                → _rerank_candidates (once) → top_k

    Args:
        queries:        Query formulations to search (e.g. [expanded_original,
                        variant1, variant2]).
        session_id:     UUID identifying the active upload session.
        top_k:          Maximum number of reranked chunks to return.
        original_query: The user's question, used for reranking + diagnostics.

    Returns:
        List of reranked chunk dicts (same schema as retrieve_relevant_chunks).
    """
    merged: Dict[int, dict] = {}
    total_retrieved = 0
    for q in queries:
        candidates = _hybrid_search(q, session_id, pool_size=_CANDIDATE_POOL)
        total_retrieved += len(candidates)
        for c in candidates:
            idx = c["chunk_index"]
            # Deduplicate by chunk_index, keeping the highest fused score.
            if idx not in merged or c["fused_score"] > merged[idx]["fused_score"]:
                merged[idx] = c

    merged_list = list(merged.values())
    print(f"Retrieved:\n{total_retrieved} chunks")
    print(f"Merged:\n{len(merged_list)} chunks")

    rerank_query = original_query or (queries[0] if queries else "")

    if not merged_list:
        # Every formulation gated out → preserve the refusal fast-path.
        print("After rerank:\n0 chunks")
        _last_retrieval_debug[session_id] = {
            "query":               rerank_query,
            "rewritten_query":     " | ".join(queries),
            "multi_query":         True,
            "num_queries":         len(queries),
            "retrieved":           total_retrieved,
            "merged":              0,
            "semantic_candidates": None,
            "bm25_candidates":     None,
            "hybrid_top_score":    None,
            "rerank_top_score":    None,
            "chunks":              [],
        }
        return []

    if ENABLE_RERANKING:
        reranked = _rerank_candidates(rerank_query, merged_list, final_k=_RERANK_TOP_K)
        results = reranked[:top_k]
    else:
        # Low-memory mode: skip the CrossEncoder; the merged set is unordered, so
        # sort by fused score before trimming to the same final chunk count.
        print("[embeddings] Reranking disabled")
        merged_sorted = sorted(merged_list, key=lambda c: c["fused_score"], reverse=True)
        results = merged_sorted[:_RERANK_TOP_K][:top_k]
    print(f"After rerank:\n{len(results)} chunks")

    # Top sections (parity with single-query observability).
    if results:
        top_sections: List[str] = []
        for r in results:
            sec = r.get("section", "Unknown")
            if sec not in top_sections:
                top_sections.append(sec)
        print("[embeddings] Top sections:")
        for sec in top_sections[:5]:
            print(f"  - {sec}")

    # Capture diagnostics for GET /debug/retrieval and DEBUG_RETRIEVAL block.
    hybrid_top = max((c["fused_score"] for c in merged_list), default=None)
    diagnostics = {
        "query":               rerank_query,
        "rewritten_query":     " | ".join(queries),
        "multi_query":         True,
        "num_queries":         len(queries),
        "retrieved":           total_retrieved,
        "merged":              len(merged_list),
        "semantic_candidates": None,  # aggregated across formulations; see 'retrieved'
        "bm25_candidates":     None,
        "hybrid_top_score":    hybrid_top,
        "rerank_top_score":    results[0].get("rerank_score") if results else None,
        "chunks": [
            {
                "rank":           i + 1,
                "page":           r["page"],
                "section":        r.get("section", "Unknown"),
                "semantic_score": r.get("semantic_score"),
                "bm25_score":     r.get("bm25_score"),
                "fused_score":    r.get("fused_score"),
                "rerank_score":   r.get("rerank_score"),
            }
            for i, r in enumerate(results)
        ],
    }
    _last_retrieval_debug[session_id] = diagnostics
    if DEBUG_RETRIEVAL:
        _print_retrieval_debug(diagnostics)

    print(
        f"[embeddings] Multi-query retrieved {len(results)} chunk(s); "
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

    # Drop the in-memory BM25 index regardless of Qdrant state.
    if _bm25_indexes.pop(session_id, None) is not None:
        print(f"[embeddings] Dropped BM25 index for session '{session_id}'.")
    _last_retrieval_debug.pop(session_id, None)

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
    Initialize ONLY the Qdrant client at startup — no transformer models.

    Transformer models (the embedding model, and the cross-encoder when
    ENABLE_RERANKING is true) are lazy-loaded on first use instead. Loading them
    at startup pushes peak memory past the Render free-tier 512 MB limit and
    OOM-kills the process before it can bind a port. Keeping startup model-free
    lets the app boot on low-memory hosts; the first upload/query pays the
    one-time model-load cost instead.
    """
    print("[embeddings] Warmup: initializing Qdrant client only (models load lazily).")
    _get_client()
    print("[embeddings] Warmup complete — no transformer models loaded at startup.")


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
