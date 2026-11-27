"""
Unit tests for eval/run_eval.py helpers.

Tests the pure-Python metric functions (recall_at_k, reciprocal_rank)
and the YAML loader without touching ChromaDB or the LLM.
"""
import sys
from pathlib import Path

import pytest
import yaml

# Make sure eval/ is importable
EVAL_DIR = Path(__file__).parent.parent / "eval"
sys.path.insert(0, str(EVAL_DIR))

from run_eval import recall_at_k, reciprocal_rank, load_golden_qa


# ---------------------------------------------------------------------------
# recall_at_k
# ---------------------------------------------------------------------------

class TestRecallAtK:

    def test_hit_at_1(self):
        assert recall_at_k(["a", "b", "c"], ["a"], k=1) == 1.0

    def test_miss_at_1_hit_at_3(self):
        assert recall_at_k(["x", "y", "a"], ["a"], k=1) == 0.0
        assert recall_at_k(["x", "y", "a"], ["a"], k=3) == 1.0

    def test_all_miss(self):
        assert recall_at_k(["x", "y", "z"], ["a"], k=5) == 0.0

    def test_multiple_relevant(self):
        # Any one of the relevant chunks in top-k counts as a hit
        assert recall_at_k(["x", "b", "c"], ["a", "b"], k=3) == 1.0

    def test_empty_retrieved(self):
        assert recall_at_k([], ["a"], k=5) == 0.0

    def test_k_larger_than_retrieved(self):
        # Should not crash when k > len(retrieved)
        assert recall_at_k(["a"], ["a"], k=10) == 1.0


# ---------------------------------------------------------------------------
# reciprocal_rank
# ---------------------------------------------------------------------------

class TestReciprocalRank:

    def test_first_position(self):
        assert reciprocal_rank(["a", "b", "c"], ["a"]) == pytest.approx(1.0)

    def test_second_position(self):
        assert reciprocal_rank(["x", "a", "c"], ["a"]) == pytest.approx(0.5)

    def test_third_position(self):
        assert reciprocal_rank(["x", "y", "a"], ["a"]) == pytest.approx(1 / 3)

    def test_no_match(self):
        assert reciprocal_rank(["x", "y", "z"], ["a"]) == 0.0

    def test_empty_retrieved(self):
        assert reciprocal_rank([], ["a"]) == 0.0

    def test_multiple_relevant_first_wins(self):
        # When multiple chunks are relevant, the highest rank (lowest index) wins.
        # retrieved=["b","a"], relevant=["a","b"] -> "b" hits at rank 1 -> RR=1.0
        assert reciprocal_rank(["b", "a"], ["a", "b"]) == pytest.approx(1.0)
        # retrieved=["x","a","b"], relevant=["a","b"] -> "a" hits at rank 2 -> RR=0.5
        assert reciprocal_rank(["x", "a", "b"], ["a", "b"]) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# load_golden_qa
# ---------------------------------------------------------------------------

class TestGoldenQA:

    def test_loads_30_questions(self):
        questions = load_golden_qa()
        assert len(questions) == 30

    def test_each_entry_has_required_fields(self):
        questions = load_golden_qa()
        for q in questions:
            assert "question" in q, f"Missing 'question' in: {q}"
            assert "answer" in q, f"Missing 'answer' in: {q}"
            assert "relevant_chunks" in q, f"Missing 'relevant_chunks' in: {q}"
            assert isinstance(q["relevant_chunks"], list)
            assert len(q["relevant_chunks"]) >= 1

    def test_chunk_ids_have_correct_format(self):
        questions = load_golden_qa()
        valid_stems = {
            "climate_info", "renewable_energy",
            "model_performance_report", "data_pipeline_architecture",
            "future_feature_roadmap",
        }
        for q in questions:
            for chunk_id in q["relevant_chunks"]:
                parts = chunk_id.rsplit("_chunk_", 1)
                assert len(parts) == 2, f"Bad chunk ID format: {chunk_id}"
                assert parts[0] in valid_stems, f"Unknown document stem: {parts[0]}"
                assert parts[1].isdigit(), f"Non-numeric chunk index: {chunk_id}"

    def test_covers_all_five_documents(self):
        questions = load_golden_qa()
        stems = set()
        for q in questions:
            for chunk_id in q["relevant_chunks"]:
                stems.add(chunk_id.rsplit("_chunk_", 1)[0])
        assert stems == {
            "climate_info", "renewable_energy",
            "model_performance_report", "data_pipeline_architecture",
            "future_feature_roadmap",
        }

    def test_no_empty_questions(self):
        questions = load_golden_qa()
        for q in questions:
            assert q["question"].strip(), "Found empty question string"
            assert q["answer"].strip(), "Found empty answer string"
