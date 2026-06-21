"""
interfaces — abstract contracts (ABCs) for the application's collaborators.

Following the Dependency Inversion and Interface Segregation principles, each
interface is small and focused, and higher-level services depend on these
abstractions rather than on concrete infrastructure (Qdrant, Mistral, PyMuPDF,
sentence-transformers).
"""

from app.interfaces.chunker import Chunker
from app.interfaces.document_parser import DocumentParser
from app.interfaces.embedder import Embedder
from app.interfaces.keyword_index import KeywordIndex
from app.interfaces.llm_client import LLMClient
from app.interfaces.reranker import Reranker
from app.interfaces.retriever import Retriever
from app.interfaces.session_repository import HistoryRepository, SessionRepository
from app.interfaces.vector_store import VectorStore

__all__ = [
    "Chunker",
    "DocumentParser",
    "Embedder",
    "KeywordIndex",
    "LLMClient",
    "Reranker",
    "Retriever",
    "HistoryRepository",
    "SessionRepository",
    "VectorStore",
]
