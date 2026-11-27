"""
BaseVectorStore — the common interface every vector store backend must implement.

Currently only ChromaVectorStore is provided, but the abstraction makes it
trivial to swap in Qdrant, Pinecone, Weaviate, etc. in a future phase.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Dict, Optional

from .loader import Document


class BaseVectorStore(ABC):
    """Abstract base for vector store backends."""

    @abstractmethod
    def add_documents(self, documents: List[Document], batch_size: int = 100) -> None:
        """Embed and store *documents*."""

    @abstractmethod
    def search(
        self,
        query: str,
        top_k: int = 5,
        filter_metadata: Optional[Dict] = None,
    ) -> List[Dict]:
        """
        Return up to *top_k* chunks most relevant to *query*.

        Each result dict must contain at minimum:
            id (str), content (str), metadata (dict), similarity (float)
        """

    @abstractmethod
    def clear(self) -> None:
        """Delete all stored documents."""

    @abstractmethod
    def count(self) -> int:
        """Return the number of stored documents."""

    @abstractmethod
    def get_stats(self) -> Dict:
        """Return a dict of store statistics (name, count, etc.)."""


class ChromaVectorStore(BaseVectorStore):
    """
    Thin adapter that makes the existing VectorStore conform to BaseVectorStore.

    This keeps all the real ChromaDB logic in vector_store.py and avoids
    duplicating code — we just delegate every call.
    """

    def __init__(self, vector_store) -> None:
        """
        Args:
            vector_store: An instance of rag_system.knowledge_base.vector_store.VectorStore
        """
        self._store = vector_store

    def add_documents(self, documents: List[Document], batch_size: int = 100) -> None:
        self._store.add_documents(documents, batch_size=batch_size)

    def search(
        self,
        query: str,
        top_k: int = 5,
        filter_metadata: Optional[Dict] = None,
    ) -> List[Dict]:
        return self._store.search(query, top_k=top_k, filter_metadata=filter_metadata)

    def clear(self) -> None:
        self._store.clear()

    def count(self) -> int:
        return self._store.collection.count()

    def get_stats(self) -> Dict:
        return self._store.get_stats()

    # Expose the underlying collection for tools that need raw ChromaDB access
    # (e.g. HybridRetriever._build_bm25_index)
    @property
    def collection(self):
        return self._store.collection


def create_vector_store(cfg: dict) -> "ChromaVectorStore":
    """
    Build a ChromaVectorStore from the application config dict.

    Currently only Chroma is supported; the factory pattern means adding
    Qdrant/Pinecone later only requires a new branch here.
    """
    from .vector_store import VectorStore

    kb = cfg.get("knowledge_base", {})
    embed = cfg.get("embedding", {})
    retrieval = cfg.get("retrieval", {})
    specialist = cfg.get("specialist", {})

    vs = VectorStore(
        collection_name=specialist.get("collection_name", "climate_docs"),
        persist_dir=kb.get("persist_dir", "./chroma_db"),
        model_name=embed.get("model_name", "all-MiniLM-L6-v2"),
        similarity_threshold=retrieval.get("similarity_threshold", 0.0),
    )
    return ChromaVectorStore(vs)
