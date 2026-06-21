"""
container.py — Composition root.

The single place where the object graph is wired. Every collaborator is built
here as a lazily-initialized process singleton and exposed through an accessor,
so the rest of the app depends on this module for construction rather than each
module new-ing up its own dependencies.

Lazy because Qdrant/Mistral configuration comes from env vars loaded after
import. Accessors read the environment on first call, so importing this module
is side-effect free.

The thin facade modules (embeddings.py, agent.py, pdf_processor.py,
summarizer.py) delegate to these accessors, guaranteeing a single shared graph —
critically, the same BM25 index and retriever instances are used at ingest time
and at query time.
"""

from __future__ import annotations

import os
from typing import Optional

from app.config import Settings
from app.infrastructure.bm25_keyword_index import BM25KeywordIndex
from app.infrastructure.cross_encoder_reranker import CrossEncoderReranker
from app.infrastructure.memory_session_repository import (
    MemoryHistoryRepository,
    MemorySessionRepository,
)
from app.infrastructure.mistral_client import MistralClient
from app.infrastructure.pymupdf_parser import PyMuPDFParser
from app.infrastructure.qdrant_vector_store import QdrantVectorStore
from app.infrastructure.sentence_transformer_embedder import SentenceTransformerEmbedder
from app.services.chat_service import ChatService
from app.services.hybrid_retriever import HybridRetriever
from app.services.query_builder import QueryBuilder
from app.services.summary_service import SummaryService

# ---------------------------------------------------------------------------
# Singletons
# ---------------------------------------------------------------------------

_settings: Optional[Settings] = None
_embedder: Optional[SentenceTransformerEmbedder] = None
_store: Optional[QdrantVectorStore] = None
_keyword_index: Optional[BM25KeywordIndex] = None
_reranker: Optional[CrossEncoderReranker] = None
_retriever: Optional[HybridRetriever] = None
_llm: Optional[MistralClient] = None
_query_builder: Optional[QueryBuilder] = None
_chat_service: Optional[ChatService] = None
_summary_service: Optional[SummaryService] = None
_parser: Optional[PyMuPDFParser] = None
_session_repo: Optional[MemorySessionRepository] = None
_history_repo: Optional[MemoryHistoryRepository] = None


def settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings.from_env()
    return _settings


def embedder() -> SentenceTransformerEmbedder:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformerEmbedder()
    return _embedder


def store() -> QdrantVectorStore:
    global _store
    if _store is None:
        _store = QdrantVectorStore(
            url=os.getenv("QDRANT_URL", "").strip(),
            api_key=os.getenv("QDRANT_API_KEY", "").strip(),
        )
    return _store


def keyword_index() -> BM25KeywordIndex:
    global _keyword_index
    if _keyword_index is None:
        _keyword_index = BM25KeywordIndex()
    return _keyword_index


def reranker() -> CrossEncoderReranker:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoderReranker()
    return _reranker


def retriever() -> HybridRetriever:
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever(
            embedder=embedder(),
            vector_store=store(),
            keyword_index=keyword_index(),
            reranker=reranker(),
            enable_reranking=os.getenv("ENABLE_RERANKING", "true").lower() == "true",
        )
    return _retriever


def llm() -> MistralClient:
    global _llm
    if _llm is None:
        _llm = MistralClient(os.getenv("MISTRAL_API_KEY", ""))
    return _llm


def query_builder() -> QueryBuilder:
    global _query_builder
    if _query_builder is None:
        _query_builder = QueryBuilder(llm())
    return _query_builder


def chat_service() -> ChatService:
    global _chat_service
    if _chat_service is None:
        _chat_service = ChatService(retriever(), llm(), query_builder())
    return _chat_service


def summary_service() -> SummaryService:
    global _summary_service
    if _summary_service is None:
        _summary_service = SummaryService(llm())
    return _summary_service


def parser() -> PyMuPDFParser:
    global _parser
    if _parser is None:
        _parser = PyMuPDFParser()
    return _parser


def session_repo() -> MemorySessionRepository:
    global _session_repo
    if _session_repo is None:
        _session_repo = MemorySessionRepository()
    return _session_repo


def history_repo() -> MemoryHistoryRepository:
    global _history_repo
    if _history_repo is None:
        _history_repo = MemoryHistoryRepository()
    return _history_repo
