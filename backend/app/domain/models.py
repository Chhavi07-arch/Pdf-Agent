"""
models.py — Domain data structures.

Typed replacements for the ad-hoc dicts passed between the parsing, embedding,
retrieval, and generation layers. Each model offers `to_dict` / `from_dict`
helpers so they can be adopted incrementally without breaking call sites that
still expect the original dict shapes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Page:
    """A single extracted PDF page."""

    page: int  # 1-indexed
    text: str


@dataclass
class Chunk:
    """A chunk of page text ready for embedding."""

    text: str
    page: int
    chunk_index: int
    section: str = "Unknown"

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "page": self.page,
            "chunk_index": self.chunk_index,
            "section": self.section,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Chunk":
        return cls(
            text=d["text"],
            page=d["page"],
            chunk_index=d["chunk_index"],
            section=d.get("section", "Unknown"),
        )


@dataclass
class RetrievedChunk:
    """A chunk returned by retrieval, carrying relevance scores."""

    text: str
    page: int
    chunk_index: int
    section: str = "Unknown"
    score: float = 0.0            # semantic cosine similarity (drives confidence bands)
    semantic_score: Optional[float] = None  # normalized
    bm25_score: Optional[float] = None      # normalized
    fused_score: Optional[float] = None
    rerank_score: Optional[float] = None

    def to_dict(self) -> dict:
        d = {
            "text": self.text,
            "page": self.page,
            "chunk_index": self.chunk_index,
            "section": self.section,
            "score": self.score,
        }
        for k in ("semantic_score", "bm25_score", "fused_score", "rerank_score"):
            v = getattr(self, k)
            if v is not None:
                d[k] = v
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "RetrievedChunk":
        return cls(
            text=d["text"],
            page=d["page"],
            chunk_index=d["chunk_index"],
            section=d.get("section", "Unknown"),
            score=d.get("score", 0.0),
            semantic_score=d.get("semantic_score"),
            bm25_score=d.get("bm25_score"),
            fused_score=d.get("fused_score"),
            rerank_score=d.get("rerank_score"),
        )


@dataclass
class DocSummary:
    """Structured document profile produced at upload time."""

    title: str
    summary: str
    topics: List[str] = field(default_factory=list)
    document_type: str = "other"
    entities: List[str] = field(default_factory=list)
    source_pages: List[int] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "summary": self.summary,
            "topics": list(self.topics),
            "document_type": self.document_type,
            "entities": list(self.entities),
            "source_pages": list(self.source_pages),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "DocSummary":
        return cls(
            title=d.get("title", ""),
            summary=d.get("summary", ""),
            topics=list(d.get("topics") or []),
            document_type=d.get("document_type", "other"),
            entities=list(d.get("entities") or []),
            source_pages=list(d.get("source_pages") or []),
        )


@dataclass
class Session:
    """Metadata for one uploaded-PDF chat session."""

    session_id: str
    filename: str
    chunk_count: int
    created_at: str
    status: str = "ready"
    doc_summary: Optional[DocSummary] = None


@dataclass
class ChatResult:
    """The outcome of answering one user turn."""

    answer: str
    cited_pages: List[int] = field(default_factory=list)
    chunks_used: int = 0

    def to_dict(self) -> dict:
        return {
            "answer": self.answer,
            "cited_pages": list(self.cited_pages),
            "chunks_used": self.chunks_used,
        }
