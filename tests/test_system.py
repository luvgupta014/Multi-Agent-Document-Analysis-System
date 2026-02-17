import pytest
import pytest_asyncio
import httpx
import asyncio
import json
from unittest.mock import patch, MagicMock

from syfe_assignment.mcp_server.models import Tool, AgentRegistration, Message, MessageResponse
from syfe_assignment.mcp_server.registry import AgentRegistry
from syfe_assignment.knowledge_base.loader import Document, DocumentLoader


class TestModels:

    def test_tool_creation(self):
        tool = Tool(name="search", description="Search docs", parameters={"type": "object"})
        assert tool.name == "search"
        assert tool.description == "Search docs"

    def test_agent_registration(self):
        reg = AgentRegistration(
            agent_id="test_agent",
            name="Test Agent",
            description="A test agent",
            tools=[Tool(name="search", description="search docs")],
            endpoint="http://localhost:9999"
        )
        assert reg.agent_id == "test_agent"
        assert len(reg.tools) == 1
        assert reg.endpoint == "http://localhost:9999"

    def test_message_creation(self):
        msg = Message(
            sender="manager",
            recipient="specialist",
            tool="document_retriever",
            parameters={"query": "test"}
        )
        assert msg.sender == "manager"
        assert msg.recipient == "specialist"
        assert msg.message_id is not None
        assert msg.timestamp is not None

    def test_message_response(self):
        resp = MessageResponse(
            message_id="test-id",
            success=True,
            result={"data": "test"}
        )
        assert resp.success is True
        assert resp.result == {"data": "test"}
        assert resp.error is None


class TestRegistry:

    def setup_method(self):
        self.registry = AgentRegistry()

    def _make_registration(self, agent_id="agent1", name="Agent 1"):
        return AgentRegistration(
            agent_id=agent_id,
            name=name,
            description="Test agent",
            tools=[Tool(name="document_retriever", description="retrieve docs")],
            endpoint=f"http://localhost:8001"
        )

    def test_register_agent(self):
        reg = self._make_registration()
        assert self.registry.register_agent(reg) is True

    def test_register_duplicate(self):
        reg = self._make_registration()
        self.registry.register_agent(reg)
        assert self.registry.register_agent(reg) is False

    def test_get_agent(self):
        reg = self._make_registration()
        self.registry.register_agent(reg)
        agent = self.registry.get_agent("agent1")
        assert agent is not None
        assert agent.name == "Agent 1"

    def test_get_nonexistent_agent(self):
        assert self.registry.get_agent("nonexistent") is None

    def test_list_agents(self):
        self.registry.register_agent(self._make_registration("a1", "Agent A"))
        self.registry.register_agent(self._make_registration("a2", "Agent B"))
        agents = self.registry.list_agents()
        assert len(agents) == 2

    def test_find_agents_with_tool(self):
        self.registry.register_agent(self._make_registration())
        found = self.registry.find_agents_with_tool("document_retriever")
        assert len(found) == 1

    def test_find_agents_no_match(self):
        self.registry.register_agent(self._make_registration())
        found = self.registry.find_agents_with_tool("nonexistent_tool")
        assert len(found) == 0

    def test_unregister_agent(self):
        self.registry.register_agent(self._make_registration())
        assert self.registry.unregister_agent("agent1") is True
        assert self.registry.get_agent("agent1") is None

    def test_unregister_nonexistent(self):
        assert self.registry.unregister_agent("nonexistent") is False

    def test_stats(self):
        self.registry.register_agent(self._make_registration())
        stats = self.registry.get_stats()
        assert stats["total_agents"] == 1
        assert len(stats["agents"]) == 1


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

    def test_load_nonexistent_file(self):
        docs = DocumentLoader.load_text_file("nonexistent_file.txt")
        assert docs == []

    def test_load_nonexistent_directory(self):
        docs = DocumentLoader.load_directory("nonexistent_dir")
        assert docs == []


class TestMCPServer:

    @pytest.fixture
    def client(self):
        from syfe_assignment.mcp_server.server import app
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
            "endpoint": "http://localhost:9999"
        }
        response = client.post("/register", json=payload)
        assert response.status_code == 201
        data = response.json()
        assert data["status"] == "registered"

    def test_register_duplicate(self, client):
        payload = {
            "agent_id": "dup_agent",
            "name": "Dup Agent",
            "description": "Test",
            "tools": [],
            "endpoint": "http://localhost:9999"
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
            "parameters": {"query": "test"}
        }
        response = client.post("/send", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is False

    def test_mcp_tools_endpoint(self, client):
        payload = {
            "agent_id": "tools_test_agent",
            "name": "Tools Test Agent",
            "description": "Test",
            "tools": [{"name": "document_retriever", "description": "retrieve docs", "parameters": {}}],
            "endpoint": "http://localhost:9998"
        }
        client.post("/register", json=payload)
        response = client.get("/mcp/v1/tools")
        assert response.status_code == 200
        tools = response.json()
        assert isinstance(tools, list)
        tool_names = [t["name"] for t in tools]
        assert "document_retriever" in tool_names


class TestMetrics:

    def test_metrics_collector(self):
        from syfe_assignment.utils.metrics import MetricsCollector, QueryMetric

        collector = MetricsCollector()
        collector.start_timer()

        metric = QueryMetric(
            question="test question",
            specialist_used="climate",
            retrieval_time=0.1,
            synthesis_time=0.2,
            total_time=0.3,
            results_count=5,
            success=True
        )
        collector.record_query(metric)

        summary = collector.get_summary()
        assert summary["total_queries"] == 1
        assert summary["successful"] == 1
        assert summary["success_rate"] == 100.0

    def test_empty_metrics(self):
        from syfe_assignment.utils.metrics import MetricsCollector
        collector = MetricsCollector()
        summary = collector.get_summary()
        assert summary["total_queries"] == 0
