"""
chat_service.py — Orchestrates a grounded answer for one user turn.

Composes the Retriever, QueryBuilder, LLMClient, prompt builder, and citation
service into the full pipeline: build retrieval query → retrieve (layered
fallback) → refuse if empty → call the LLM with the ORIGINAL question →
validate citations. Provides both a blocking `get_answer` and a streaming
`stream_answer`.
"""

from __future__ import annotations

import json
from typing import Iterator, List, Optional

from app.config import HIGH_CONFIDENCE_SCORE, MEDIUM_CONFIDENCE_SCORE
from app.interfaces.llm_client import LLMClient
from app.interfaces.retriever import Retriever
from app.services import prompt_builder
from app.services.citation_service import (
    REFUSAL_PREFIX,
    extract_cited_pages,
    is_refusal,
    validate_citations,
)
from app.services.query_builder import QueryBuilder

MAX_TOKENS = 1024

# Multi-query diversification: disabled by default (raised refusal rate + latency
# in evaluation). The implementation is kept intact and re-enables by flipping
# this flag — the retriever must expose retrieve_multi for it to take effect.
ENABLE_MULTI_QUERY = False


def _confidence_band(top_score: float) -> str:
    if top_score >= HIGH_CONFIDENCE_SCORE:
        return "HIGH"
    if top_score >= MEDIUM_CONFIDENCE_SCORE:
        return "MEDIUM"
    return "LOW"


class ChatService:
    """Answers a user turn, grounded strictly in retrieved PDF chunks."""

    def __init__(self, retriever: Retriever, llm: LLMClient, query_builder: QueryBuilder) -> None:
        self._retriever = retriever
        self._llm = llm
        self._qb = query_builder

    # ── retrieval (shared by both response paths) ──────────────────────────────
    def _retrieve_dicts(self, query: str, session_id: str, original_query: str) -> List[dict]:
        chunks = self._retriever.retrieve(query, session_id, top_k=10, original_query=original_query)
        return [c.to_dict() for c in chunks]

    def _run_retrieval(self, query: str, session_id: str, history: List[dict]) -> List[dict]:
        retrieval_query = self._qb.build_retrieval_query(query, history)
        expanded_query = self._qb.expand_query(retrieval_query)
        print(f"[chat] Final retrieval query (len={len(expanded_query)}): {expanded_query!r}")

        if ENABLE_MULTI_QUERY and hasattr(self._retriever, "retrieve_multi"):
            variants = self._qb.generate_variants(query)
            if variants:
                query_set = [expanded_query] + [self._qb.expand_query(v) for v in variants]
                chunks = self._retriever.retrieve_multi(query_set, session_id, top_k=10, original_query=query)
                if chunks:
                    return [c.to_dict() for c in chunks]
                print("[chat] Multi-query returned nothing — single-query fallback.")

        # Layered fallback: expanded → unexpanded → original+synonyms → raw original.
        chunks = self._retrieve_dicts(expanded_query, session_id, query)
        if not chunks:
            print("[chat] Fallback 1: retrieval query without expansion.")
            chunks = self._retrieve_dicts(retrieval_query, session_id, query)
        if not chunks and retrieval_query != query:
            print("[chat] Fallback 2: original user query.")
            chunks = self._retrieve_dicts(self._qb.expand_query(query), session_id, query)
            if not chunks:
                chunks = self._retrieve_dicts(query, session_id, query)
        return chunks

    @staticmethod
    def _refusal_for(query: str) -> str:
        topic = query if len(query) <= 80 else query[:77] + "…"
        return f"{REFUSAL_PREFIX} {topic}."

    @staticmethod
    def _log_confidence(chunks: List[dict]) -> tuple[float, set]:
        top_score = chunks[0].get("score", 0.0)
        pages = {c["page"] for c in chunks}
        print(
            f"[chat] Retrieval confidence: {_confidence_band(top_score)} "
            f"(top_score={top_score:.4f}, chunks={len(chunks)}, pages={sorted(pages)})"
        )
        return top_score, pages

    # ── blocking ───────────────────────────────────────────────────────────────
    def get_answer(self, query: str, session_id: str, history: Optional[List[dict]] = None) -> dict:
        history = history or []
        print(f"[chat] Query: {query!r} | session={session_id} | history_turns={len(history)}")

        chunks = self._run_retrieval(query, session_id, history)
        if not chunks:
            print("[chat] REFUSAL: zero_chunks.")
            return {"answer": self._refusal_for(query), "cited_pages": [], "chunks_used": 0}

        top_score, chunk_pages = self._log_confidence(chunks)
        force_answer = top_score >= HIGH_CONFIDENCE_SCORE
        if force_answer:
            print("[chat] Refusal override: confidence HIGH, forcing evidence-based answer.")

        user_content = prompt_builder.build_user_message(query, chunks, force_answer=force_answer)
        messages = prompt_builder.build_messages(list(history), user_content)
        answer = self._llm.complete(messages, MAX_TOKENS)

        # HIGH-confidence refusal guard: retry once with an explicit escalation.
        if force_answer and is_refusal(answer):
            print("[chat] Refusal override: HIGH confidence but LLM refused — retrying once.")
            retry_content = prompt_builder.build_user_message(query, chunks, force_answer=True, retry=True)
            retry_messages = prompt_builder.build_messages(list(history), retry_content)
            answer = self._llm.complete(retry_messages, MAX_TOKENS)

        cited_pages = validate_citations(extract_cited_pages(answer), chunk_pages)
        print(
            f"[chat] Response done | is_refusal={is_refusal(answer)} | "
            f"cited_pages={cited_pages} | confidence={_confidence_band(top_score)}"
        )
        return {"answer": answer, "cited_pages": cited_pages, "chunks_used": len(chunks)}

    # ── streaming ──────────────────────────────────────────────────────────────
    def stream_answer(
        self, query: str, session_id: str, history: Optional[List[dict]] = None
    ) -> Iterator[str]:
        history = history or []
        print(f"[chat] STREAM Query: {query!r} | session={session_id} | history_turns={len(history)}")

        chunks = self._run_retrieval(query, session_id, history)
        if not chunks:
            print("[chat] STREAM REFUSAL: zero_chunks.")
            refusal = self._refusal_for(query)
            yield _ndjson({"type": "token", "text": refusal})
            yield _ndjson({"type": "done", "answer": refusal, "cited_pages": [],
                           "is_refusal": True, "chunks_used": 0})
            return

        top_score, chunk_pages = self._log_confidence(chunks)
        force_answer = top_score >= HIGH_CONFIDENCE_SCORE
        if force_answer:
            print("[chat] STREAM Refusal override: confidence HIGH, forcing answer.")

        user_content = prompt_builder.build_user_message(query, chunks, force_answer=force_answer)
        messages = prompt_builder.build_messages(list(history), user_content)

        parts: List[str] = []
        for token in self._llm.stream(messages, MAX_TOKENS):
            parts.append(token)
            yield _ndjson({"type": "token", "text": token})

        answer = "".join(parts)
        cited_pages = validate_citations(extract_cited_pages(answer), chunk_pages)
        refusal = is_refusal(answer)
        print(f"[chat] STREAM done | is_refusal={refusal} | cited_pages={cited_pages} | chars={len(answer)}")
        yield _ndjson({"type": "done", "answer": answer, "cited_pages": cited_pages,
                       "is_refusal": refusal, "chunks_used": len(chunks)})


def _ndjson(obj: dict) -> str:
    return json.dumps(obj, ensure_ascii=False) + "\n"
