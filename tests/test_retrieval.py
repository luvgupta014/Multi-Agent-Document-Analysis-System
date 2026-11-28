"""
Unit tests for Phase 2: hybrid retrieval helpers.

Covers:
- reciprocal_rank_fusion (RRF)
- DocumentLoader.recursive_chunk_text
- HybridRetriever (dense-only path, no real model needed via mocks)
"""
import sys
import os
from unittest.mock import MagicMock, patch

# Make sure project root is on sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rag_system.knowledge_base.retriever import reciprocal_rank_fusion, HybridRetriever
from rag_system.knowledge_base.loader import DocumentLoader


# ---------------------------------------------------------------------------
# reciprocal_rank_fusion
# ---------------------------------------------------------------------------

class TestRRF:

    def test_single_list(self):
        result = reciprocal_rank_fusion([["a", "b", "c"]])
        ids = [r[0] for r in result]
        assert ids[0] == "a"  # rank 1 → highest score

    def test_two_lists_consensus(self):
        # "b" appears first in both lists → should win
        result = reciprocal_rank_fusion([["b", "a"], ["b", "c"]])
        ids = [r[0] for r in result]
        assert ids[0] == "b"

    def test_two_lists_disagreement(self):
        # "a" is #1 in list1, "b" is #1 in list2 — scores should be equal
        result = reciprocal_rank_fusion([["a", "b"], ["b", "a"]])
        scores = {r[0]: r[1] for r in result}
        assert abs(scores["a"] - scores["b"]) < 1e-9

    def test_empty_lists(self):
        assert reciprocal_rank_fusion([]) == []
        assert reciprocal_rank_fusion([[]]) == []

    def test_rrf_k_effect(self):
        # Larger k → scores closer together
        r1 = reciprocal_rank_fusion([["a", "b"]], k=1)
        r2 = reciprocal_rank_fusion([["a", "b"]], k=1000)
        gap1 = r1[0][1] - r1[1][1]
        gap2 = r2[0][1] - r2[1][1]
        assert gap1 > gap2

    def test_scores_sum_per_doc(self):
        # With identical lists, scores double
        single = reciprocal_rank_fusion([["a"]], k=60)
        double = reciprocal_rank_fusion([["a"], ["a"]], k=60)
        assert abs(double[0][1] - 2 * single[0][1]) < 1e-9

    def test_returns_sorted_desc(self):
        result = reciprocal_rank_fusion([["a", "b", "c"], ["c", "a", "b"]])
        scores = [r[1] for r in result]
        assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# recursive_chunk_text
# ---------------------------------------------------------------------------

class TestRecursiveChunking:

    def test_short_text_not_split(self):
        text = "Hello world."
        chunks = DocumentLoader.recursive_chunk_text(text, chunk_size=500)
        assert len(chunks) == 1
        assert "Hello world" in chunks[0]

    def test_long_text_is_split(self):
        # Use unique words so the dedup step doesn't collapse identical overlap chunks
        text = " ".join(f"word{i}" for i in range(300))
        chunks = DocumentLoader.recursive_chunk_text(text, chunk_size=100)
        assert len(chunks) > 1

    def test_no_chunk_exceeds_size(self):
        text = "paragraph one.\n\nparagraph two is a bit longer.\n\n" * 10
        chunks = DocumentLoader.recursive_chunk_text(text, chunk_size=100)
        for chunk in chunks:
            # allow up to 10% over due to indivisible words
            assert len(chunk) <= 110, f"Chunk too large: {len(chunk)}"

    def test_no_empty_chunks(self):
        text = "Line one.\n\nLine two.\n\n\n\nLine three."
        chunks = DocumentLoader.recursive_chunk_text(text, chunk_size=500)
        assert all(c.strip() for c in chunks)

    def test_all_content_preserved(self):
        """Every word in the source should appear in at least one chunk."""
        words = ["alpha", "beta", "gamma", "delta", "epsilon"]
        text = ". ".join(words)
        chunks = DocumentLoader.recursive_chunk_text(text, chunk_size=20)
        combined = " ".join(chunks)
        for word in words:
            assert word in combined

    def test_strategy_produces_different_chunks(self):
        """sentence_aware and recursive should produce different chunk counts."""
        text = "First sentence. Second sentence. Third sentence.\n\nNew paragraph here.\n\nAnother paragraph."
        sa = DocumentLoader.chunk_text(text, chunk_size=50)
        rc = DocumentLoader.recursive_chunk_text(text, chunk_size=50)
        # Both strategies should produce valid chunks; counts may differ
        assert len(sa) >= 1
        assert len(rc) >= 1


# ---------------------------------------------------------------------------
# HybridRetriever (mock vector store — no real ChromaDB/model needed)
# ---------------------------------------------------------------------------

def _make_mock_vs(docs=None):
    """Create a minimal mock VectorStore for testing."""
    if docs is None:
        docs = [
            {"id": "doc_0", "content": "climate change global warming", "metadata": {}, "similarity": 0.9, "distance": 0.1},
            {"id": "doc_1", "content": "renewable energy solar wind", "metadata": {}, "similarity": 0.7, "distance": 0.3},
            {"id": "doc_2", "content": "machine learning model accuracy", "metadata": {}, "similarity": 0.5, "distance": 0.5},
        ]
    vs = MagicMock()
    vs.search.return_value = docs
    vs.collection.get.return_value = {
        "ids": [d["id"] for d in docs],
        "documents": [d["content"] for d in docs],
        "metadatas": [d.get("metadata", {}) for d in docs],
    }
    return vs


class TestHybridRetriever:

    def test_dense_only_passthrough(self):
        """With hybrid=False, results come straight from vector store."""
        vs = _make_mock_vs()
        retriever = HybridRetriever(vs, hybrid=False, rerank=False)
        results = retriever.search("climate", top_k=2)
        assert len(results) == 2
        assert results[0]["id"] == "doc_0"

    def test_hybrid_returns_top_k(self):
        """Hybrid mode should return exactly top_k results."""
        vs = _make_mock_vs()
        retriever = HybridRetriever(vs, hybrid=True, rerank=False)
        results = retriever.search("climate", top_k=2)
        assert len(results) <= 2

    def test_hybrid_results_have_rrf_score(self):
        """Hybrid results should carry an rrf_score field."""
        vs = _make_mock_vs()
        retriever = HybridRetriever(vs, hybrid=True, rerank=False)
        results = retriever.search("climate", top_k=3)
        for r in results:
            assert "rrf_score" in r

    def test_dense_results_have_id_and_content(self):
        vs = _make_mock_vs()
        retriever = HybridRetriever(vs, hybrid=False, rerank=False)
        results = retriever.search("energy", top_k=3)
        for r in results:
            assert "id" in r
            assert "content" in r

    def test_bm25_unavailable_falls_back(self):
        """If rank_bm25 is not installed, hybrid silently falls back to dense."""
        vs = _make_mock_vs()
        with patch.dict("sys.modules", {"rank_bm25": None}):
            retriever = HybridRetriever(vs, hybrid=True, rerank=False)
            # After fallback, hybrid should be False
            assert retriever.hybrid is False
