"""
Hybrid retriever: BM25 (lexical) + dense vector search fused with
Reciprocal Rank Fusion (RRF), with an optional cross-encoder re-ranker.

All features are toggled via config.yaml:
  retrieval.hybrid: true/false
  retrieval.rerank: true/false
  retrieval.rrf_k:  RRF constant (default 60)
"""
from __future__ import annotations

from typing import List, Dict, Optional
from loguru import logger


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------

def reciprocal_rank_fusion(
    ranked_lists: List[List[str]],
    k: int = 60,
) -> List[tuple[str, float]]:
    """
    Fuse multiple ranked lists of document IDs using RRF.

    Args:
        ranked_lists: Each inner list is a ranking (best first) of doc IDs.
        k: RRF constant. Larger values reduce the impact of rank position.

    Returns:
        List of (doc_id, rrf_score) sorted by descending score.
    """
    scores: Dict[str, float] = {}
    for ranking in ranked_lists:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


# ---------------------------------------------------------------------------
# HybridRetriever
# ---------------------------------------------------------------------------

class HybridRetriever:
    """
    Retriever that wraps a VectorStore and optionally adds:
    - BM25 lexical search (requires rank_bm25)
    - RRF score fusion
    - Cross-encoder re-ranking (requires sentence-transformers CrossEncoder)
    """

    def __init__(
        self,
        vector_store,
        hybrid: bool = True,
        rerank: bool = False,
        rrf_k: int = 60,
        rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ):
        self.vector_store = vector_store
        self.hybrid = hybrid
        self.rerank = rerank
        self.rrf_k = rrf_k
        self._rerank_model_name = rerank_model
        self._cross_encoder = None
        self._bm25 = None
        self._bm25_corpus: List[str] = []
        self._bm25_ids: List[str] = []
        self._bm25_metadatas: List[Dict] = []

        if hybrid:
            self._build_bm25_index()

        if rerank:
            self._load_cross_encoder()

    # ------------------------------------------------------------------
    # Index building
    # ------------------------------------------------------------------

    def _build_bm25_index(self) -> None:
        """Build a BM25 index from all documents currently in the vector store."""
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            logger.warning(
                "rank_bm25 not installed — falling back to dense-only search. "
                "Install it with: pip install rank-bm25"
            )
            self.hybrid = False
            return

        collection = getattr(self.vector_store, "collection", None)
        if collection is None:
            logger.warning("Vector store has no collection attribute — BM25 disabled.")
            self._bm25 = None
            return

        # Fetch all documents from ChromaDB
        result = collection.get(include=["documents", "metadatas"])
        ids = result.get("ids", [])
        texts = result.get("documents", [])
        metadatas = result.get("metadatas") or [{} for _ in ids]

        if not ids:
            logger.warning("BM25 index is empty: no documents in vector store yet.")
            self._bm25 = None
            self._bm25_ids = []
            self._bm25_corpus = []
            self._bm25_metadatas = []
            return

        self._bm25_ids = ids
        self._bm25_corpus = texts
        self._bm25_metadatas = metadatas
        tokenised = [t.lower().split() for t in texts]
        self._bm25 = BM25Okapi(tokenised)
        logger.info(f"BM25 index built with {len(ids)} documents")

    def refresh_index(self) -> None:
        """Rebuild the BM25 index after new documents are added to the vector store."""
        if self.hybrid:
            self._build_bm25_index()

    def _load_cross_encoder(self) -> None:
        """Load the cross-encoder model for re-ranking."""
        try:
            from sentence_transformers import CrossEncoder
            self._cross_encoder = CrossEncoder(self._rerank_model_name)
            logger.info(f"Cross-encoder loaded: {self._rerank_model_name}")
        except Exception as exc:
            logger.warning(f"Cross-encoder unavailable ({exc}) — reranking disabled.")
            self.rerank = False

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = 5,
        filter_metadata: Optional[Dict] = None,
    ) -> List[Dict]:
        """
        Retrieve the top-k most relevant chunks for *query*.

        Pipeline:
          1. Dense vector search  (always on)
          2. BM25 lexical search  (if hybrid=True)
          3. RRF fusion           (if hybrid=True, else just dense ranking)
          4. Cross-encoder rerank (if rerank=True)
        """
        # --- 1. Dense retrieval -------------------------------------------
        # Fetch more candidates than top_k so fusion has room to rerank
        fetch_k = top_k * 3 if self.hybrid else top_k
        dense_results = self.vector_store.search(query, top_k=fetch_k, filter_metadata=filter_metadata)
        dense_ids = [r["id"] for r in dense_results]
        id_to_result: Dict[str, Dict] = {r["id"]: r for r in dense_results}

        if not self.hybrid or self._bm25 is None:
            results = dense_results[:top_k]
        else:
            # --- 2. BM25 retrieval ----------------------------------------
            tokens = query.lower().split()
            bm25_scores = self._bm25.get_scores(tokens)
            bm25_ranked = [
                self._bm25_ids[i]
                for i in sorted(range(len(bm25_scores)), key=lambda x: bm25_scores[x], reverse=True)
            ][:fetch_k]

            # populate id_to_result with any BM25-only hits
            for doc_id in bm25_ranked:
                if doc_id not in id_to_result:
                    idx = self._bm25_ids.index(doc_id)
                    meta = self._bm25_metadatas[idx] if idx < len(self._bm25_metadatas) else {}
                    id_to_result[doc_id] = {
                        "id": doc_id,
                        "content": self._bm25_corpus[idx],
                        "metadata": meta or {},
                        "similarity": 0.0,
                        "distance": 1.0,
                    }

            # --- 3. RRF fusion --------------------------------------------
            fused = reciprocal_rank_fusion([dense_ids, bm25_ranked], k=self.rrf_k)
            results = []
            for doc_id, rrf_score in fused[:top_k]:
                entry = dict(id_to_result[doc_id])
                entry["rrf_score"] = round(rrf_score, 6)
                results.append(entry)

        # --- 4. Cross-encoder re-rank -------------------------------------
        if self.rerank and self._cross_encoder and results:
            pairs = [(query, r["content"]) for r in results]
            ce_scores = self._cross_encoder.predict(pairs)
            for r, score in zip(results, ce_scores):
                r["rerank_score"] = float(score)
            results.sort(key=lambda x: x.get("rerank_score", 0.0), reverse=True)
            logger.debug(f"Re-ranked {len(results)} results with cross-encoder")

        return results
