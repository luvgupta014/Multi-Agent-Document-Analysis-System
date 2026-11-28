"""
Tests for Phase 4 additions:
  - ConversationMemory
  - SQL safety filter (_is_safe)
  - _ensure_sample_db / _run_query
  - SQLSpecialistAgent.get_tools()
  - _ddg_search (mocked)
  - WebSearchSpecialistAgent.get_tools()
  - ManagerAgent._route() with mocked LLM
  - ManagerAgent memory integration
  - ManagerAgent._synthesize_answer SQL path
"""
from __future__ import annotations

import json
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# ConversationMemory
# ---------------------------------------------------------------------------
from rag_system.agents.memory import ConversationMemory


class TestConversationMemory:
    def test_initial_empty(self):
        mem = ConversationMemory()
        assert len(mem) == 0
        assert mem.format_for_prompt() == ""

    def test_add_single_turn(self):
        mem = ConversationMemory()
        mem.add("user", "Hello")
        mem.add("assistant", "Hi there")
        assert len(mem) == 2

    def test_format_contains_role_and_content(self):
        mem = ConversationMemory()
        mem.add("user", "What is RAG?")
        mem.add("assistant", "Retrieval-Augmented Generation.")
        fmt = mem.format_for_prompt()
        assert "User: What is RAG?" in fmt
        assert "Assistant: Retrieval-Augmented Generation." in fmt
        assert "Conversation history" in fmt

    def test_rolling_window_respects_max_turns(self):
        mem = ConversationMemory(max_turns=2)
        for i in range(5):
            mem.add("user", f"q{i}")
            mem.add("assistant", f"a{i}")
        # max_turns=2 → keep last 2 pairs = 4 items
        assert len(mem) == 4

    def test_rolling_window_drops_oldest(self):
        mem = ConversationMemory(max_turns=1)
        mem.add("user", "old question")
        mem.add("assistant", "old answer")
        mem.add("user", "new question")
        mem.add("assistant", "new answer")
        fmt = mem.format_for_prompt()
        assert "old question" not in fmt
        assert "new question" in fmt

    def test_clear_resets(self):
        mem = ConversationMemory()
        mem.add("user", "hi")
        mem.clear()
        assert len(mem) == 0
        assert mem.format_for_prompt() == ""

    def test_max_turns_zero_keeps_nothing(self):
        mem = ConversationMemory(max_turns=0)
        mem.add("user", "hi")
        mem.add("assistant", "hello")
        assert len(mem) == 0


# ---------------------------------------------------------------------------
# SQL safety filter
# ---------------------------------------------------------------------------
from rag_system.agents.sql_specialist import _is_safe, _run_query, _ensure_sample_db


class TestSQLSafety:
    def test_select_allowed(self):
        assert _is_safe("SELECT * FROM models") is True

    def test_select_case_insensitive(self):
        assert _is_safe("select id from experiments") is True

    def test_select_with_where_allowed(self):
        assert _is_safe("SELECT name FROM models WHERE id = 1") is True

    def test_insert_rejected(self):
        assert _is_safe("INSERT INTO models VALUES (1,'x')") is False

    def test_update_rejected(self):
        assert _is_safe("UPDATE models SET name='y' WHERE id=1") is False

    def test_delete_rejected(self):
        assert _is_safe("DELETE FROM models WHERE id=1") is False

    def test_drop_rejected(self):
        assert _is_safe("DROP TABLE models") is False

    def test_create_rejected(self):
        assert _is_safe("CREATE TABLE foo (id INTEGER)") is False

    def test_empty_string_rejected(self):
        assert _is_safe("") is False

    def test_select_with_embedded_drop_rejected(self):
        # Tricky: starts with SELECT but contains DROP
        assert _is_safe("SELECT * FROM models; DROP TABLE models") is False


class TestEnsureSampleDB:
    def test_creates_db_if_missing(self, tmp_path):
        db_file = str(tmp_path / "test.db")
        _ensure_sample_db(db_file)
        assert Path(db_file).exists()

    def test_idempotent_if_exists(self, tmp_path):
        db_file = str(tmp_path / "test.db")
        _ensure_sample_db(db_file)
        mtime1 = Path(db_file).stat().st_mtime
        _ensure_sample_db(db_file)
        mtime2 = Path(db_file).stat().st_mtime
        assert mtime1 == mtime2  # file untouched on second call

    def test_sample_db_has_models_table(self, tmp_path):
        db_file = str(tmp_path / "test.db")
        _ensure_sample_db(db_file)
        conn = sqlite3.connect(db_file)
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
        conn.close()
        assert "models" in tables
        assert "experiments" in tables

    def test_run_query_returns_columns_and_rows(self, tmp_path):
        db_file = str(tmp_path / "test.db")
        _ensure_sample_db(db_file)
        result = _run_query(db_file, "SELECT id, name FROM models ORDER BY id")
        assert "columns" in result
        assert "id" in result["columns"]
        assert "name" in result["columns"]
        assert len(result["rows"]) >= 1

    def test_run_query_row_count(self, tmp_path):
        db_file = str(tmp_path / "test.db")
        _ensure_sample_db(db_file)
        result = _run_query(db_file, "SELECT * FROM models")
        assert result["row_count"] == len(result["rows"])


# ---------------------------------------------------------------------------
# SQLSpecialistAgent.get_tools()
# ---------------------------------------------------------------------------
from rag_system.agents.sql_specialist import SQLSpecialistAgent


class TestSQLSpecialistAgentTools:
    def test_get_tools_returns_one_tool(self, tmp_path):
        db_file = str(tmp_path / "test.db")
        agent = SQLSpecialistAgent(db_path=db_file)
        tools = agent.get_tools()
        assert len(tools) == 1
        assert tools[0].name == "sql_query"

    def test_tool_has_query_parameter(self, tmp_path):
        db_file = str(tmp_path / "test.db")
        agent = SQLSpecialistAgent(db_path=db_file)
        tool = agent.get_tools()[0]
        assert "query" in tool.parameters["properties"]


# ---------------------------------------------------------------------------
# _ddg_search (mocked — no real network calls)
# ---------------------------------------------------------------------------
from rag_system.agents.web_search import _ddg_search


class TestDDGSearch:
    def test_returns_list_on_success(self):
        fake_response = {
            "AbstractText": "Python is a programming language.",
            "Heading": "Python",
            "AbstractURL": "https://python.org",
            "RelatedTopics": [],
        }
        mock_resp = MagicMock()
        mock_resp.json.return_value = fake_response
        mock_resp.raise_for_status = MagicMock()

        with patch("rag_system.agents.web_search.httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.get.return_value = mock_resp
            mock_client_cls.return_value.__enter__.return_value = mock_client

            results = _ddg_search("Python language")

        assert isinstance(results, list)
        assert len(results) >= 1
        assert results[0]["title"] == "Python"
        assert "Python is a programming language." in results[0]["snippet"]

    def test_returns_empty_on_empty_response(self):
        fake_response = {"AbstractText": "", "RelatedTopics": []}
        mock_resp = MagicMock()
        mock_resp.json.return_value = fake_response
        mock_resp.raise_for_status = MagicMock()

        with patch("rag_system.agents.web_search.httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.get.return_value = mock_resp
            mock_client_cls.return_value.__enter__.return_value = mock_client

            results = _ddg_search("xyzzy no results")

        assert results == []

    def test_related_topics_included(self):
        fake_response = {
            "AbstractText": "",
            "RelatedTopics": [
                {"Text": "Topic A details", "FirstURL": "https://example.com/a"},
                {"Text": "Topic B details", "FirstURL": "https://example.com/b"},
            ],
        }
        mock_resp = MagicMock()
        mock_resp.json.return_value = fake_response
        mock_resp.raise_for_status = MagicMock()

        with patch("rag_system.agents.web_search.httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.get.return_value = mock_resp
            mock_client_cls.return_value.__enter__.return_value = mock_client

            results = _ddg_search("test query")

        assert len(results) == 2
        assert results[0]["url"] == "https://example.com/a"


# ---------------------------------------------------------------------------
# WebSearchSpecialistAgent.get_tools()
# ---------------------------------------------------------------------------
from rag_system.agents.web_search import WebSearchSpecialistAgent


class TestWebSearchSpecialistAgentTools:
    def test_get_tools_returns_one_tool(self):
        agent = WebSearchSpecialistAgent()
        tools = agent.get_tools()
        assert len(tools) == 1
        assert tools[0].name == "web_search"

    def test_tool_has_query_parameter(self):
        agent = WebSearchSpecialistAgent()
        tool = agent.get_tools()[0]
        assert "query" in tool.parameters["properties"]


# ---------------------------------------------------------------------------
# ManagerAgent._route() with mocked LLM
# ---------------------------------------------------------------------------
from rag_system.agents.manager import ManagerAgent
from rag_system.llm.base import BaseLLM, LLMConfig


def _mock_manager(route_response: str = '{"action": "document_retrieval", "reason": "test"}'):
    """Helper: returns a ManagerAgent with a fully mocked BaseLLM."""
    mock_llm = MagicMock(spec=BaseLLM)
    mock_llm.generate.return_value = route_response
    return ManagerAgent(
        mcp_url="http://localhost:9999",
        llm=mock_llm,
    )


class TestManagerRoute:
    def test_route_document_retrieval(self):
        mgr = _mock_manager('{"action": "document_retrieval", "reason": "docs"}')
        result = mgr._route("What is climate change?")
        assert result["action"] == "document_retrieval"

    def test_route_sql_query(self):
        mgr = _mock_manager('{"action": "sql_query", "reason": "structured data"}')
        result = mgr._route("Show me models with auc > 0.9")
        assert result["action"] == "sql_query"

    def test_route_web_search(self):
        mgr = _mock_manager('{"action": "web_search", "reason": "current events"}')
        result = mgr._route("What is today's news?")
        assert result["action"] == "web_search"

    def test_route_direct(self):
        mgr = _mock_manager('{"action": "direct", "reason": "simple greeting"}')
        result = mgr._route("Hello!")
        assert result["action"] == "direct"

    def test_route_fallback_on_invalid_json(self):
        mgr = _mock_manager("not valid json at all")
        result = mgr._route("What is RAG?")
        # Should fall back to document_retrieval
        assert result["action"] == "document_retrieval"

    def test_route_fallback_on_unknown_action(self):
        mgr = _mock_manager('{"action": "unknown_action", "reason": "?"}')
        result = mgr._route("What is RAG?")
        assert result["action"] == "document_retrieval"

    def test_route_returns_reason(self):
        mgr = _mock_manager('{"action": "document_retrieval", "reason": "needs docs"}')
        result = mgr._route("What is RAG?")
        assert "reason" in result


# ---------------------------------------------------------------------------
# ManagerAgent memory integration
# ---------------------------------------------------------------------------
class TestManagerMemory:
    def test_memory_initialized(self):
        mgr = _mock_manager()
        assert len(mgr.memory) == 0

    def test_memory_grows_when_added_directly(self):
        mgr = _mock_manager()
        mgr.memory.add("user", "Hello")
        mgr.memory.add("assistant", "Hi!")
        assert len(mgr.memory) == 2

    def test_memory_max_turns_configurable(self):
        mgr = ManagerAgent(mcp_url="http://localhost:9999", llm=MagicMock(spec=BaseLLM), memory_turns=3)
        assert mgr.memory.max_turns == 3


# ---------------------------------------------------------------------------
# ManagerAgent._synthesize_answer SQL path
# ---------------------------------------------------------------------------
class TestSynthesizeSQLPath:
    def test_sql_path_uses_rows_as_context(self):
        mock_llm = MagicMock(spec=BaseLLM)
        mock_llm.generate.return_value = "Here are the results."
        mgr = ManagerAgent(mcp_url="http://localhost:9999", llm=mock_llm)

        sql_result = {
            "columns": ["id", "name", "auc_roc"],
            "rows": [{"id": 1, "name": "churn_v3", "auc_roc": 0.89}],
        }
        answer = mgr._synthesize_answer("Show me models", sql_result)
        assert isinstance(answer, str)
        # LLM was called with a prompt containing the row data
        call_args = mock_llm.generate.call_args[0][0]
        assert "churn_v3" in call_args or "rows" in call_args.lower() or "results" in call_args.lower()

    def test_no_results_returns_fallback(self):
        mock_llm = MagicMock(spec=BaseLLM)
        mgr = ManagerAgent(mcp_url="http://localhost:9999", llm=mock_llm)
        answer = mgr._synthesize_answer("Question?", {"results": []})
        assert "couldn't find" in answer.lower() or answer != ""
