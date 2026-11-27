"""
ChromaDB vector store wrapper.

Uses cosine similarity (hnsw:space=cosine) so that similarity scores are
directly interpretable: 1.0 = identical, 0.0 = orthogonal.
"""
import chromadb
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional
from loguru import logger
import os

from .loader import Document


class VectorStore:

    def __init__(
        self,
        collection_name: str = "documents",
        persist_dir: str = "./chroma_db",
        model_name: str = "all-MiniLM-L6-v2",
        similarity_threshold: float = 0.0,
    ):
        logger.info(f"Initializing VectorStore: {collection_name}")
        os.makedirs(persist_dir, exist_ok=True)

        # PersistentClient so indexed data survives process restarts
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.similarity_threshold = similarity_threshold

        # cosine distance → similarity is simply (1 - distance) for normalised vectors
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={
                "description": "Document embeddings for retrieval",
                "hnsw:space": "cosine",
            },
        )

        self.embedding_model = SentenceTransformer(model_name)
        logger.info(f"VectorStore ready with {self.collection.count()} documents")

    def add_documents(self, documents: List[Document], batch_size: int = 100):
        if not documents:
            return

        logger.info(f"Adding {len(documents)} documents to vector store")

        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]
            texts = [doc.content for doc in batch]
            ids = [str(doc.doc_id) for doc in batch]
            metadatas = [doc.metadata for doc in batch]

            embeddings = self.embedding_model.encode(
                texts,
                show_progress_bar=True,
                convert_to_numpy=True,
            ).tolist()

            self.collection.add(
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
                ids=ids,
            )

        logger.info(f"Total documents in store: {self.collection.count()}")

    def search(
        self,
        query: str,
        top_k: int = 5,
        filter_metadata: Optional[Dict] = None,
    ) -> List[Dict]:
        query_embedding = self.embedding_model.encode([query])[0].tolist()

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=filter_metadata,
        )

        formatted_results = []

        if results["ids"] and len(results["ids"][0]) > 0:
            for i in range(len(results["ids"][0])):
                distance = results["distances"][0][i]
                # With cosine space: distance ∈ [0, 2]; similarity = 1 - distance
                similarity = 1.0 - distance

                if similarity < self.similarity_threshold:
                    continue

                formatted_results.append(
                    {
                        "id": results["ids"][0][i],
                        "content": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i],
                        "similarity": round(similarity, 4),
                        "distance": round(distance, 4),
                    }
                )

        return formatted_results

    def clear(self):
        self.client.delete_collection(self.collection.name)
        self.collection = self.client.create_collection(
            self.collection.name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("Vector store cleared")

    def get_stats(self) -> Dict:
        return {
            "collection_name": self.collection.name,
            "document_count": self.collection.count(),
            "embedding_dimension": self.embedding_model.get_sentence_embedding_dimension(),
            "similarity_threshold": self.similarity_threshold,
        }
