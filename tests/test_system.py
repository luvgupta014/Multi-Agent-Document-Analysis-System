"""
Test suite for the Multi-Agent Document Analysis System.

Covers:
- Pydantic models (Tool, AgentRegistration, Message, MessageResponse)
- AgentRegistry (register, unregister, find, list, stats)
- DocumentLoader (chunking, txt loading, directory loading, error paths)
- VectorStore stats helper
- MCP Server HTTP endpoints (via FastAPI TestClient)
- ManagerAgent logic (_should_use_retrieval, _select_specialist, _synthesize_answer)
- MetricsCollector (record, summary, reset)
- REST API health endpoint
"""
import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from rag_system.mcp_server.models import Tool, AgentRegistration, Message, MessageResponse
from rag_system.mcp_server.registry import AgentRegistry
from rag_system.knowledge_base.loader import Document, DocumentLoader


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_registration(agent_id: str = "agent1", name: str = "Agent 1") -> AgentRegistration:
    return AgentRegistration(
        agent_id=agent_id,
        name=name,
        description="Test agent",
        tools=[Tool(name="document_retriever", description="retrieve docs")],
        endpoint="http://localhost:8001",
    )


# ===========================================================================
# TestModels
# ===========================================================================

class TestModels:

    def test_tool_creation(self):
        tool = Tool(name="search", description="Search docs", parameters={"type": "object"})
        assert tool.name == "search"
        assert tool.description == "Search docs"

    def test_agent_registration(self):
        reg = _make_registration()
        assert reg.agent_id == "agent1"
        assert len(reg.tools) == 1
        assert reg.endpoint == "http://localhost:8001"

    def test_message_creation(self):
        msg = Message(
            sender="manager",
            recipient="specialist",
            tool="document_retriever",
            parameters={"query": "test"},
        )
        assert msg.sender == "manager"
        assert msg.recipient == "specialist"
        assert msg.message_id is not None
        assert msg.timestamp is not None

    def test_message_response_success(self):
        resp = MessageResponse(
            message_id="test-id",
            success=True,
            result={"data": "test"},
        )
        assert resp.success is True
        assert resp.result == {"data": "test"}
        assert resp.error is None

    def test_message_response_failure(self):
        resp = MessageResponse(
            message_id="err-id",
            success=False,
            error="something went wrong",
        )
        assert resp.success is False
        assert resp.error == "something went wrong"


# ===========================================================================
# TestRegistry
# ===========================================================================

class TestRegistry:

    def setup_method(self):
        self.registry = AgentRegistry()

    def test_register_agent(self):
        assert self.registry.register_agent(_make_registration()) is True

    def test_register_duplicate(self):
        reg = _make_registration()
        self.registry.register_agent(reg)
        assert self.registry.register_agent(reg) is False

    def test_get_agent(self):
        self.registry.register_agent(_make_registration())
        agent = self.registry.get_agent("agent1")
        assert agent is not None
        assert agent.name == "Agent 1"

    def test_get_nonexistent_agent(self):
        assert self.registry.get_agent("nonexistent") is None

    def test_list_agents(self):
        self.registry.register_agent(_make_registration("a1", "Agent A"))
        self.registry.register_agent(_make_registration("a2", "Agent B"))
        agents = self.registry.list_agents()
        assert len(agents) == 2

    def test_find_agents_with_tool(self):
        self.registry.register_agent(_make_registration())
        found = self.registry.find_agents_with_tool("document_retriever")
        assert len(found) == 1

    def test_find_agents_no_match(self):
        self.registry.register_agent(_make_registration())
        found = self.registry.find_agents_with_tool("nonexistent_tool")
        assert len(found) == 0

    def test_unregister_agent(self):
        self.registry.register_agent(_make_registration())
        assert self.registry.unregister_agent("agent1") is True
        assert self.registry.get_agent("agent1") is None

    def test_unregister_nonexistent(self):
        assert self.registry.unregister_agent("nonexistent") is False

    def test_stats(self):
        self.registry.register_agent(_make_registration())
        stats = self.registry.get_stats()
        assert stats["total_agents"] == 1
        assert len(stats["agents"]) == 1


# ===========================================================================
# TestDocumentLoader
# ===========================================================================

class TestDocumentLoader:

    def test_document_creation(self):
        doc = Document(content="Test content", metadata={"id": "test1"})
        assert doc.content == "Test content"
        assert doc.doc_id == "test1"

    def test_document_repr(self):
        doc = Document(content="Short text")
        assert "Short text" in repr(doc)

    def test_document_len(self):
        doc = Document(content="Hello World")
        assert len(doc) == 11

    def test_chunk_text_basic(self):
        text = "First sentence. Second sentence. Third sentence. Fourth sentence."
        chunks = DocumentLoader.chunk_text(text, chunk_size=40, overlap=0)
        assert len(chunks) >= 1

    def test_chunk_text_with_overlap(self):
        text = "A very long first sentence here. A second sentence here. A third sentence."
        chunks = DocumentLoader.chunk_text(text, chunk_size=40, overlap=20)
        assert len(chunks) >= 1

    def test_chunk_text_preserves_content(self):
        text = "Alpha. Beta. Gamma."
        chunks = DocumentLoader.chunk_text(text, chunk_size=1000, overlap=0)
        # All content should fit in one chunk
        assert len(chunks) == 1
        full = " ".join(chunks)
        assert "Alpha" in full

    def test_load_nonexistent_file(self):
        docs = DocumentLoader.load_text_file("nonexistent_file.txt")
        assert docs == []

    def test_load_nonexistent_directory(self):
        docs = DocumentLoader.load_directory("nonexistent_dir")
        assert docs == []

    def test_load_pdf_missing_pypdf(self):
        """load_pdf_file should return [] when pypdf is not importable."""
        with patch.dict("sys.modules", {"pypdf": None}):
            docs = DocumentLoader.load_pdf_file("some_file.pdf")
            assert docs == []

    def test_load_pdf_nonexistent_file(self):
        """load_pdf_file should return [] for a missing file."""
        docs = DocumentLoader.load_pdf_file("does_not_exist.pdf")
        assert docs == []

    def test_load_text_file_produces_metadata(self, tmp_path):
        """Check that loaded documents carry expected metadata keys."""
        f = tmp_path / "sample.txt"
        f.write_text("Hello world. This is a test sentence. And another one.")
        docs = DocumentLoader.load_text_file(str(f))
        assert len(docs) >= 1
        meta = docs[0].metadata
        assert "filename" in meta
        assert "chunk_index" in meta
        assert "file_type" in meta
        assert meta["file_type"] == "txt"

    def test_load_directory_mixed_extensions(self, tmp_path):
        """Directory loader should process .txt and skip unsupported types."""
        (tmp_path / "a.txt").write_text("Some text here. More text. Even more.")
        (tmp_path / "b.md").write_text("# Markdown — should be ignored")
        docs = DocumentLoader.load_directory(str(tmp_path))
        filenames = [d.metadata["filename"] for d in docs]
        assert any("a.txt" in f for f in filenames)
        assert not any("b.md" in f for f in filenames)


# ===========================================================================
# TestMCPServer
# ===========================================================================

class TestMCPServer:

    @pytest.fixture
    def client(self):
        from rag_system.mcp_server.server import app
        from fastapi.testclient import TestClient
        return TestClient(app)

    def test_health(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "MCP Server"

    def test_register_agent(self, client):
        payload = {
            "agent_id": "test_specialist",
            "name": "Test Specialist",
            "description": "Test description",
            "tools": [{"name": "document_retriever", "description": "retrieve docs", "parameters": {}}],
            "endpoint": "http://localhost:9999",
        }
        response = client.post("/register", json=payload)
        assert response.status_code == 201
        assert response.json()["status"] == "registered"

    def test_register_duplicate(self, client):
        payload = {
            "agent_id": "dup_agent",
            "name": "Dup Agent",
            "description": "Test",
            "tools": [],
            "endpoint": "http://localhost:9999",
        }
        client.post("/register", json=payload)
        response = client.post("/register", json=payload)
        assert response.status_code == 409

    def test_list_agents(self, client):
        response = client.get("/agents")
        assert response.status_code == 200
        assert isinstance(response.json(), list)

    def test_get_nonexistent_agent(self, client):
        response = client.get("/agents/nonexistent_id_12345")
        assert response.status_code == 404

    def test_unregister_nonexistent(self, client):
        response = client.delete("/unregister/nonexistent_id_12345")
        assert response.status_code == 404

    def test_send_to_nonexistent(self, client):
        payload = {
            "sender": "manager",
            "recipient": "nonexistent_agent_xyz",
            "tool": "document_retriever",
            "parameters": {"query": "test"},
        }
        response = client.post("/send", json=payload)
        assert response.status_code == 200
        assert response.json()["success"] is False

    def test_mcp_tools_endpoint(self, client):
        payload = {
            "agent_id": "tools_test_agent",
            "name": "Tools Test Agent",
            "description": "Test",
            "tools": [{"name": "document_retriever", "description": "retrieve docs", "parameters": {}}],
            "endpoint": "http://localhost:9998",
        }
        client.post("/register", json=payload)
        response = client.get("/mcp/v1/tools")
        assert response.status_code == 200
        tools = response.json()
        assert isinstance(tools, list)
        tool_names = [t["name"] for t in tools]
        assert "document_retriever" in tool_names


# ===========================================================================
# TestManagerAgent
# ===========================================================================

class TestManagerAgent:
    """Unit tests for ManagerAgent logic (no network calls)."""

    def _make_manager(self):
        """Build a ManagerAgent with a mock LLM injected — no genai patching needed."""
        from rag_system.agents.manager import ManagerAgent
        from rag_system.llm.base import BaseLLM, LLMConfig

        mock_llm = MagicMock(spec=BaseLLM)
        mock_llm.config = LLMConfig(model_name="fake-model")
        mock_llm.__repr__ = lambda self: "MockLLM(model=fake-model)"
        manager = ManagerAgent(agent_id="test", mcp_url="http://localhost:8000", llm=mock_llm)
        return manager, mock_llm

    def test_should_not_retrieve_for_math(self):
        manager, mock_llm = self._make_manager()
        # _route() now parses JSON — a "direct" action means no retrieval
        mock_llm.generate.return_value = '{"action": "direct", "reason": "math question"}'
        result = manager._should_use_retrieval("What is 2+2?", ["document_retriever"])
        assert result is False

    def test_should_retrieve_for_domain_question(self):
        manager, mock_llm = self._make_manager()
        mock_llm.generate.return_value = "yes"
        result = manager._should_use_retrieval(
            "What is the churn model AUC-ROC?", ["document_retriever"]
        )
        assert result is True

    def test_no_retrieval_when_tool_missing(self):
        manager, _ = self._make_manager()
        result = manager._should_use_retrieval("Any question", [])
        assert result is False

    def test_select_specialist_returns_none_for_empty(self):
        manager, _ = self._make_manager()
        assert manager._select_specialist("test", []) is None

    def test_select_specialist_picks_best_match(self):
        manager, _ = self._make_manager()
        specialists = [
            {
                "agent_id": "climate_spec",
                "name": "Climate Specialist",
                "description": "climate energy environment",
                "tools": [{"name": "document_retriever"}],
            },
            {
                "agent_id": "ml_spec",
                "name": "ML Specialist",
                "description": "machine learning models metrics",
                "tools": [{"name": "document_retriever"}],
            },
        ]
        selected = manager._select_specialist("churn model metrics", specialists)
        assert selected["agent_id"] == "ml_spec"

    def test_synthesize_no_results_returns_fallback(self):
        manager, _ = self._make_manager()
        answer = manager._synthesize_answer("What is X?", {"results": []})
        assert "couldn't find" in answer.lower()

    def test_synthesize_calls_gemini_with_context(self):
        manager, mock_llm = self._make_manager()
        mock_llm.generate.return_value = "The answer is 42."
        results = {
            "results": [
                {
                    "content": "Some relevant text",
                    "metadata": {"filename": "doc.txt", "chunk_index": 0},
                    "similarity": 0.9,
                }
            ]
        }
        answer = manager._synthesize_answer("What is the answer?", results)
        assert answer == "The answer is 42."
        mock_llm.generate.assert_called_once()


# ===========================================================================
# TestMetrics
# ===========================================================================

class TestMetrics:

    def test_metrics_collector(self):
        from rag_system.utils.metrics import MetricsCollector, QueryMetric

        collector = MetricsCollector()
        collector.start_timer()

        metric = QueryMetric(
            question="test question",
            specialist_used="climate",
            retrieval_time=0.1,
            synthesis_time=0.2,
            total_time=0.3,
            results_count=5,
            success=True,
        )
        collector.record_query(metric)

        summary = collector.get_summary()
        assert summary["total_queries"] == 1
        assert summary["successful"] == 1
        assert summary["success_rate"] == 100.0

    def test_empty_metrics(self):
        from rag_system.utils.metrics import MetricsCollector
        collector = MetricsCollector()
        summary = collector.get_summary()
        assert summary["total_queries"] == 0

    def test_metrics_reset(self):
        from rag_system.utils.metrics import MetricsCollector, QueryMetric
        collector = MetricsCollector()
        collector.record_query(
            QueryMetric("q", "s", 0.1, 0.1, 0.2, 3, True)
        )
        assert collector.get_summary()["total_queries"] == 1
        collector.reset()
        assert collector.get_summary()["total_queries"] == 0

    def test_failed_metrics_tracked(self):
        from rag_system.utils.metrics import MetricsCollector, QueryMetric
        collector = MetricsCollector()
        collector.record_query(QueryMetric("q1", "s", 0.1, 0.1, 0.2, 0, success=False))
        collector.record_query(QueryMetric("q2", "s", 0.1, 0.2, 0.3, 5, success=True))
        summary = collector.get_summary()
        assert summary["failed"] == 1
        assert summary["successful"] == 1
        assert summary["success_rate"] == 50.0


# ===========================================================================
# TestRestAPI
# ===========================================================================

class TestRestAPI:

    @pytest.fixture
    async def api_client(self):
        import httpx
        from rag_system.api.app import app
        with patch.dict("os.environ", {"GEMINI_API_KEY": "fake-key"}):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="http://test",
            ) as ac:
                yield ac

    async def test_api_health(self, api_client):
        response = await api_client.get("/health")
        assert response.status_code == 200
