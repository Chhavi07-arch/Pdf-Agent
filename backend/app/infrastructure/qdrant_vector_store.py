"""
qdrant_vector_store.py — VectorStore backed by Qdrant (no in-memory fallback).

A running Qdrant server is mandatory. If QDRANT_URL is unset the store raises
ConfigError; any connectivity/HTTP failure is translated to
QdrantUnavailableError so the API layer can answer "Qdrant server not working"
instead of leaking vendor exceptions or degrading to ephemeral memory.

Collection strategy: one Qdrant collection per session_id (UUID) — natural
isolation, atomic deletion, no per-query metadata filter.
"""

from __future__ import annotations

from typing import List, Optional

from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse
from qdrant_client.models import Distance, PointStruct, VectorParams

from app.domain.models import RetrievedChunk
from app.errors import ConfigError, QdrantUnavailableError
from app.interfaces.vector_store import VectorStore

# Qdrant client exceptions that indicate the server is unreachable or returned an
# unexpected response. Translated to QdrantUnavailableError.
_QDRANT_ERRORS = (UnexpectedResponse, ResponseHandlingException, ConnectionError, TimeoutError, OSError)


class QdrantVectorStore(VectorStore):
    """Qdrant-backed vector store; client is lazily constructed on first use."""

    def __init__(self, url: str, api_key: str = "", timeout: int = 30) -> None:
        self._url = (url or "").strip()
        self._api_key = (api_key or "").strip()
        self._timeout = timeout
        self._client: Optional[QdrantClient] = None

    # ── connection ──────────────────────────────────────────────────────────
    def _get(self) -> QdrantClient:
        if self._client is None:
            if not self._url:
                raise ConfigError(
                    "QDRANT_URL is not set. A running Qdrant server is required — "
                    "the in-memory fallback has been removed."
                )
            print(f"[qdrant] Connecting to Qdrant at {self._url!r}…")
            try:
                self._client = QdrantClient(url=self._url, api_key=self._api_key or None, timeout=self._timeout)
            except _QDRANT_ERRORS as exc:
                raise QdrantUnavailableError(f"Could not connect to Qdrant: {exc!r}") from exc
            print("[qdrant] Client ready.")
        return self._client

    @property
    def is_initialized(self) -> bool:
        return self._client is not None

    def warmup(self) -> None:
        """Construct the client and probe connectivity (raises on failure)."""
        client = self._get()
        try:
            client.get_collections()
        except _QDRANT_ERRORS as exc:
            raise QdrantUnavailableError(f"Qdrant not reachable at startup: {exc!r}") from exc

    # ── collection lifecycle ─────────────────────────────────────────────────
    def create_collection(self, name: str, vector_size: int) -> None:
        client = self._get()
        try:
            if client.collection_exists(collection_name=name):
                client.delete_collection(collection_name=name)
            client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )
        except _QDRANT_ERRORS as exc:
            raise QdrantUnavailableError(f"Qdrant collection setup failed: {exc!r}") from exc

    def collection_exists(self, name: str) -> bool:
        client = self._get()
        try:
            return client.collection_exists(collection_name=name)
        except _QDRANT_ERRORS as exc:
            raise QdrantUnavailableError(f"Qdrant collection lookup failed: {exc!r}") from exc

    def delete_collection(self, name: str) -> None:
        client = self._get()
        try:
            if client.collection_exists(collection_name=name):
                client.delete_collection(collection_name=name)
        except _QDRANT_ERRORS as exc:
            raise QdrantUnavailableError(f"Qdrant delete failed: {exc!r}") from exc

    # ── data ─────────────────────────────────────────────────────────────────
    def upsert(
        self,
        name: str,
        ids: List[int],
        vectors: List[List[float]],
        payloads: List[dict],
    ) -> None:
        client = self._get()
        points = [
            PointStruct(id=ids[i], vector=vectors[i], payload=payloads[i])
            for i in range(len(ids))
        ]
        try:
            client.upsert(collection_name=name, points=points)
        except _QDRANT_ERRORS as exc:
            raise QdrantUnavailableError(f"Qdrant upsert failed: {exc!r}") from exc

    def search(self, name: str, query_vector: List[float], limit: int) -> List[RetrievedChunk]:
        client = self._get()
        try:
            if not client.collection_exists(collection_name=name):
                print(f"[qdrant] Collection '{name}' not found — returning empty results.")
                return []
            hits = client.search(
                collection_name=name,
                query_vector=query_vector,
                limit=limit,
                with_payload=True,
            )
        except _QDRANT_ERRORS as exc:
            raise QdrantUnavailableError(f"Qdrant search failed: {exc!r}") from exc

        return [
            RetrievedChunk(
                text=h.payload["text"],
                page=h.payload["page"],
                chunk_index=h.payload["chunk_index"],
                section=h.payload.get("section", "Unknown"),
                score=float(h.score),
            )
            for h in hits
        ]

    def ping(self) -> bool:
        if self._client is None:
            return False
        try:
            self._client.get_collections()
            return True
        except Exception as exc:  # noqa: BLE001 — probe must never raise
            print(f"[qdrant] Connectivity check failed: {exc!r}")
            return False
