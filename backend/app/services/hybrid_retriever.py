"""
hybrid_retriever.py — Retriever that fuses semantic + BM25 and reranks.

Composition over inheritance: this service is constructed with an Embedder, a
VectorStore, a KeywordIndex, and a Reranker (all interfaces), and orchestrates
them into the full query→chunks pipeline. It depends only on abstractions, so
any collaborator can be swapped without touching this code (Dependency Inversion).

Pipeline (per query):
  1. Semantic search (Qdrant), gated by the live MIN_RETRIEVAL_SCORE threshold —
     if nothing clears it, return [] so the caller can refuse (fast-path).
  2. BM25 keyword search over the same session.
  3. Min-max normalize each retriever's scores, fuse:
        fused = SEMANTIC_WEIGHT*sem_norm + BM25_WEIGHT*bm25_norm
  4. Cross-encoder rerank the fused pool (skipped when reranking is disabled).

Each returned chunk keeps `score` = its semantic cosine (0.0 for BM25-only
chunks) so the caller's confidence/refusal bands are unaffected.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from app.config import (
    BM25_WEIGHT,
    CANDIDATE_POOL,
    RERANK_TOP_K,
    SEMANTIC_WEIGHT,
    Settings,
)
from app.domain.models import RetrievedChunk
from app.interfaces.embedder import Embedder
from app.interfaces.keyword_index import KeywordIndex
from app.interfaces.reranker import Reranker
from app.interfaces.retriever import Retriever
from app.interfaces.vector_store import VectorStore


def _minmax_normalize(score_map: Dict[int, float]) -> Dict[int, float]:
    """
    Min-max normalize a {chunk_index: score} map into [0, 1].

    Cosine and BM25 live on different scales, so each retriever's scores are
    normalized independently before fusion. All-equal (or single) → all 1.0.
    """
    if not score_map:
        return {}
    values = list(score_map.values())
    lo, hi = min(values), max(values)
    if hi - lo < 1e-9:
        return {k: 1.0 for k in score_map}
    return {k: (v - lo) / (hi - lo) for k, v in score_map.items()}


class HybridRetriever(Retriever):
    """Semantic + BM25 fusion with cross-encoder reranking."""

    def __init__(
        self,
        embedder: Embedder,
        vector_store: VectorStore,
        keyword_index: KeywordIndex,
        reranker: Reranker,
        *,
        enable_reranking: bool,
        candidate_pool: int = CANDIDATE_POOL,
        rerank_top_k: int = RERANK_TOP_K,
        debug: bool = True,
    ) -> None:
        self._embedder = embedder
        self._store = vector_store
        self._keyword_index = keyword_index
        self._reranker = reranker
        self._enable_reranking = enable_reranking
        self._pool = candidate_pool
        self._rerank_top_k = rerank_top_k
        self._debug = debug
        # session_id -> latest diagnostics (for GET /debug/retrieval)
        self._last_debug: Dict[str, dict] = {}

    # ── public API ────────────────────────────────────────────────────────────
    def retrieve(
        self,
        query: str,
        session_id: str,
        top_k: int = 10,
        original_query: Optional[str] = None,
    ) -> List[RetrievedChunk]:
        candidates, hybrid_stats = self._hybrid_search(query, session_id)

        if not candidates:
            print("[retriever] Retrieved 0 chunk(s); top score=n/a.")
            return []

        if self._enable_reranking:
            results = self._reranker.rerank(query, candidates, self._rerank_top_k)[:top_k]
        else:
            print("[retriever] Reranking disabled")
            results = candidates[: self._rerank_top_k][:top_k]

        print(
            f"[retriever] Retrieved {len(results)} chunk(s); "
            f"top score={results[0].score if results else 'n/a'}."
        )
        self._log_sections(results)
        self._record_debug(session_id, original_query or query, query, hybrid_stats, results)
        return results

    def retrieve_multi(
        self,
        queries: List[str],
        session_id: str,
        top_k: int = 10,
        original_query: Optional[str] = None,
    ) -> List[RetrievedChunk]:
        """Run several query formulations, merge by chunk_index, rerank once."""
        merged: Dict[int, RetrievedChunk] = {}
        total_retrieved = 0
        for q in queries:
            candidates, _ = self._hybrid_search(q, session_id)
            total_retrieved += len(candidates)
            for c in candidates:
                if c.chunk_index not in merged or (c.fused_score or 0) > (merged[c.chunk_index].fused_score or 0):
                    merged[c.chunk_index] = c

        merged_list = list(merged.values())
        print(f"[retriever] Multi-query retrieved={total_retrieved} merged={len(merged_list)}")

        rerank_query = original_query or (queries[0] if queries else "")
        if not merged_list:
            return []

        if self._enable_reranking:
            results = self._reranker.rerank(rerank_query, merged_list, self._rerank_top_k)[:top_k]
        else:
            merged_list.sort(key=lambda c: (c.fused_score or 0), reverse=True)
            results = merged_list[: self._rerank_top_k][:top_k]

        self._log_sections(results)
        self._record_debug(session_id, rerank_query, " | ".join(queries), None, results, multi=True)
        return results

    def get_last_debug(self, session_id: str) -> Optional[dict]:
        return self._last_debug.get(session_id)

    def drop_debug(self, session_id: str) -> None:
        """Forget a session's cached diagnostics (called on session delete)."""
        self._last_debug.pop(session_id, None)

    # ── internals ─────────────────────────────────────────────────────────────
    def _hybrid_search(self, query: str, session_id: str) -> tuple[List[RetrievedChunk], dict]:
        query_vec = self._embedder.encode([query])[0].astype("float32").tolist()
        semantic = self._store.search(session_id, query_vec, self._pool)

        # Gate on the live min-score threshold (read per call for runtime tuning).
        min_score = Settings.live_min_retrieval_score()
        semantic_kept = [c for c in semantic if c.score >= min_score]
        dropped = len(semantic) - len(semantic_kept)
        if dropped:
            print(f"[retriever] Score filter (min={min_score}): kept {len(semantic_kept)}, dropped {dropped}.")
        print(f"[retriever] Semantic candidates={len(semantic_kept)}")

        if not semantic_kept:
            print("[retriever] No semantic candidates above threshold — hybrid empty (refusal path).")
            return [], {"semantic_candidates": 0, "bm25_candidates": 0, "hybrid_top_score": None}

        bm25 = self._keyword_index.search(session_id, query, self._pool)
        print(f"[retriever] BM25 candidates={len(bm25)}")

        sem_norm = _minmax_normalize({c.chunk_index: c.score for c in semantic_kept})
        bm25_norm = _minmax_normalize({c.chunk_index: c.score for c in bm25})

        by_index: Dict[int, RetrievedChunk] = {}
        for c in semantic_kept:
            by_index[c.chunk_index] = c                # score = cosine
        for c in bm25:
            if c.chunk_index not in by_index:
                c.score = 0.0                          # BM25-only: no cosine
                by_index[c.chunk_index] = c

        fused: List[RetrievedChunk] = []
        for idx, chunk in by_index.items():
            s = sem_norm.get(idx, 0.0)
            b = bm25_norm.get(idx, 0.0)
            chunk.score = round(chunk.score, 4)
            chunk.semantic_score = round(s, 4)
            chunk.bm25_score = round(b, 4)
            chunk.fused_score = round(SEMANTIC_WEIGHT * s + BM25_WEIGHT * b, 4)
            fused.append(chunk)

        fused.sort(key=lambda x: x.fused_score, reverse=True)
        fused = fused[: self._pool]

        print(f"[retriever] Hybrid candidates={len(fused)}")
        stats = {
            "semantic_candidates": len(semantic_kept),
            "bm25_candidates": len(bm25),
            "hybrid_top_score": fused[0].fused_score if fused else None,
        }
        return fused, stats

    @staticmethod
    def _log_sections(results: List[RetrievedChunk]) -> None:
        if not results:
            return
        seen: List[str] = []
        for r in results:
            if r.section not in seen:
                seen.append(r.section)
        print("[retriever] Top sections:")
        for sec in seen[:5]:
            print(f"  - {sec}")

    def _record_debug(
        self,
        session_id: str,
        query: str,
        rewritten_query: str,
        hybrid_stats: Optional[dict],
        results: List[RetrievedChunk],
        multi: bool = False,
    ) -> None:
        diagnostics = {
            "query": query,
            "rewritten_query": rewritten_query,
            "multi_query": multi,
            "semantic_candidates": (hybrid_stats or {}).get("semantic_candidates"),
            "bm25_candidates": (hybrid_stats or {}).get("bm25_candidates"),
            "hybrid_top_score": (hybrid_stats or {}).get("hybrid_top_score"),
            "rerank_top_score": results[0].rerank_score if results else None,
            "chunks": [
                {
                    "rank": i + 1,
                    "page": r.page,
                    "section": r.section,
                    "semantic_score": r.semantic_score,
                    "bm25_score": r.bm25_score,
                    "fused_score": r.fused_score,
                    "rerank_score": r.rerank_score,
                }
                for i, r in enumerate(results)
            ],
        }
        self._last_debug[session_id] = diagnostics
        if self._debug:
            self._print_debug(diagnostics)

    @staticmethod
    def _fmt(value) -> str:
        return f"{value:.2f}" if isinstance(value, (int, float)) else "n/a"

    def _print_debug(self, diag: dict) -> None:
        print("[retrieval]")
        print(f"Query: {diag['query']}")
        if diag["rewritten_query"] != diag["query"]:
            print(f"Rewritten Query: {diag['rewritten_query']}")
        print(f"Semantic Candidates: {diag['semantic_candidates']}")
        print(f"BM25 Candidates: {diag['bm25_candidates']}")
        print(f"Hybrid Top Score: {self._fmt(diag['hybrid_top_score'])}")
        print(f"Rerank Top Score: {self._fmt(diag['rerank_top_score'])}")
        for c in diag["chunks"]:
            print(f"\n#{c['rank']}")
            print(f"Page: {c['page']}")
            print(f"Section: {c['section']}")
            print(f"Semantic: {self._fmt(c['semantic_score'])}")
            print(f"BM25: {self._fmt(c['bm25_score'])}")
            print(f"Fused: {self._fmt(c['fused_score'])}")
            print(f"Rerank: {self._fmt(c['rerank_score'])}")
