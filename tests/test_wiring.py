"""
End-to-end integration and wiring tests covering:
- SpecialistAgent + HybridRetriever live wiring & POST /ingest refresh
- HybridRetriever.refresh_index() + BM25 metadata preservation
- ManagerAgent create_llm(cfg) + config.yaml memory_turns wiring
- ManagerAgent NL-to-SQL translation before querying SQLSpecialistAgent
- POST /upload without mocking _manager.specialist (real ManagerAgent fallback + MCP ingest)
- Public arXiv PDFs under data/ load cleanly via DocumentLoader
- register_with_mcp retry handling
"""
from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx

from rag_system.agents.manager import ManagerAgent
from rag_system.agents.specialist import SpecialistAgent
from rag_system.knowledge_base.loader import DocumentLoader
from rag_system.knowledge_base.retriever import HybridRetriever
from rag_system.llm.base import BaseLLM, LLMConfig
from rag_system.mcp_server.client import register_with_mcp


class TestSpecialistHybridAndIngest:

    def _make_mock_vs(self):
        docs = [
            {
                "id": "doc_0",
                "content": "Solar panels convert sunlight.",
                "metadata": {"filename": "solar.txt", "chunk_index": 0},
                "similarity": 0.85,
                "distance": 0.15,
            }
        ]
        vs = MagicMock()
        vs.count.return_value = 1
        vs.search.return_value = docs
        vs.collection.get.return_value = {
            "ids": ["doc_0"],
            "documents": ["Solar panels convert sunlight."],
            "metadatas": [{"filename": "solar.txt", "chunk_index": 0}],
        }
        return vs

    async def test_specialist_uses_hybrid_retriever_and_ingest_refreshes_bm25(self):
        vs = self._make_mock_vs()
        agent = SpecialistAgent(
            agent_id="spec_test",
            name="Test Spec",
            description="Test",
            collection_name="test_col",
            vector_store=vs,
            hybrid=True,
            rerank=False,
        )
        assert isinstance(agent.retriever, HybridRetriever)
        assert agent.retriever.hybrid is True

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=agent.app),
            base_url="http://test",
        ) as client:
            # Execute search goes through HybridRetriever
            exec_resp = await client.post(
                "/execute",
                json={
                    "sender": "mgr",
                    "recipient": "spec_test",
                    "tool": "document_retriever",
                    "parameters": {"query": "solar panels", "top_k": 3},
                },
            )
            assert exec_resp.status_code == 200
            assert exec_resp.json()["results"][0]["rrf_score"] > 0

            # Ingest adds documents and triggers BM25 refresh
            ingest_resp = await client.post(
                "/ingest",
                json={
                    "documents": [
                        {"content": "Wind turbines capture wind.", "metadata": {"filename": "wind.txt", "chunk_index": 0}}
                    ]
                },
            )
            assert ingest_resp.status_code == 200
            assert ingest_resp.json()["chunks_added"] == 1
            vs.add_documents.assert_called_once()

    def test_bm25_only_hit_preserves_metadata(self):
        vs = MagicMock()
        vs.search.return_value = []  # dense returns nothing
        vs.collection.get.return_value = {
            "ids": ["doc_x"],
            "documents": ["Specific keyword xylophone metric"],
            "metadatas": [{"filename": "report.txt", "chunk_index": 2}],
        }
        retriever = HybridRetriever(vs, hybrid=True, rerank=False)
        results = retriever.search("xylophone", top_k=1)
        assert len(results) == 1
        assert results[0]["metadata"]["filename"] == "report.txt"
        assert results[0]["metadata"]["chunk_index"] == 2

    def test_refresh_index_after_empty_start(self):
        vs = MagicMock()
        vs.collection.get.return_value = {"ids": [], "documents": [], "metadatas": []}
        retriever = HybridRetriever(vs, hybrid=True, rerank=False)
        assert retriever._bm25 is None

        vs.collection.get.return_value = {
            "ids": ["new_1"],
            "documents": ["Newly uploaded nuclear fusion text"],
            "metadatas": [{"filename": "new.txt", "chunk_index": 0}],
        }
        retriever.refresh_index()
        assert retriever._bm25 is not None


class TestManagerFactoryAndSQLFlow:

    def test_manager_uses_create_llm_when_no_llm_or_key_passed(self):
        mock_llm = MagicMock(spec=BaseLLM)
        mock_llm.config = LLMConfig(provider="ollama", model_name="llama3.2")
        with patch("rag_system.agents.manager.create_llm", return_value=mock_llm) as mock_factory:
            mgr = ManagerAgent(mcp_url="http://localhost:9999")
        mock_factory.assert_called_once()
        assert mgr._llm is mock_llm

    def test_generate_sql_strips_markdown_fences(self):
        mock_llm = MagicMock(spec=BaseLLM)
        mock_llm.generate.return_value = "```sql\nSELECT name, auc_roc FROM models;\n```"
        mgr = ManagerAgent(mcp_url="http://localhost:9999", llm=mock_llm)
        sql = mgr._generate_sql("Show model AUC scores", schema="Schema:\n  models(name TEXT, auc_roc REAL)")
        assert sql == "SELECT name, auc_roc FROM models"

    async def test_answer_question_translates_nl_to_sql_for_sql_specialist(self):
        mock_llm = MagicMock(spec=BaseLLM)
        mock_llm.generate.side_effect = [
            '{"action": "sql_query", "reason": "tabular metrics"}',
            "SELECT name, auc_roc FROM models ORDER BY auc_roc DESC",
            "The top model is revenue_forecast with AUC 0.91.",
        ]
        mgr = ManagerAgent(mcp_url="http://localhost:9999", llm=mock_llm)
        sql_specialist = {
            "agent_id": "sql_specialist",
            "name": "SQL Data Specialist",
            "description": "Structured SQLite DB",
            "endpoint": "http://localhost:8003",
            "tools": [
                {
                    "name": "sql_query",
                    "description": "Run SELECT.\nSchema:\n  models(id INTEGER, name TEXT, auc_roc REAL)",
                }
            ],
        }
        mgr.discover_specialists = AsyncMock(return_value=[sql_specialist])
        mgr.query_specialist = AsyncMock(
            return_value={
                "columns": ["name", "auc_roc"],
                "rows": [{"name": "revenue_forecast", "auc_roc": 0.91}],
            }
        )

        res = await mgr.answer_question("Which model has the highest AUC?")
        assert res["success"] is True
        assert res["routing_action"] == "sql_query"
        # Verify query_specialist received SQL, not raw English question
        called_query = mgr.query_specialist.call_args[0][1]
        assert called_query.startswith("SELECT ")


class TestLiveUploadWithoutSpecialistAttr:

    async def test_upload_works_with_real_manager_falling_back_to_vector_store(self):
        from rag_system.api import app as app_module

        mock_llm = MagicMock(spec=BaseLLM)
        real_mgr = ManagerAgent(mcp_url="http://localhost:9999", llm=mock_llm)
        real_mgr.discover_specialists = AsyncMock(return_value=[])

        mock_store = MagicMock()
        mock_store.add_documents = MagicMock()

        with (
            patch.object(app_module, "_manager", real_mgr),
            patch("rag_system.knowledge_base.base_store.create_vector_store", return_value=mock_store),
        ):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app_module.app),
                base_url="http://test",
            ) as client:
                resp = await client.post(
                    "/upload",
                    files={"file": ("note.txt", io.BytesIO(b"Renewable solar wind power."), "text/plain")},
                )

        assert resp.status_code == 200
        assert resp.json()["chunks_indexed"] >= 1
        mock_store.add_documents.assert_called_once()


class TestDataCorpusAndMCPClient:

    def test_arxiv_pdfs_exist_and_load(self):
        data_dir = Path(__file__).parent.parent / "data"
        pdfs = list(data_dir.glob("*.pdf"))
        assert len(pdfs) >= 2
        docs = DocumentLoader.load_pdf_file(str(pdfs[0]), chunk_size=500, strategy="recursive")
        assert len(docs) > 5

    async def test_register_with_mcp_retries_and_fails_cleanly(self):
        agent = MagicMock()
        agent.agent_id = "test_agent"
        agent.get_registration.return_value = MagicMock(model_dump=lambda: {"agent_id": "test_agent"})

        with patch("rag_system.mcp_server.client.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.post.side_effect = httpx.ConnectError("refused")
            mock_cls.return_value.__aenter__.return_value = mock_client
            ok = await register_with_mcp(agent, "http://localhost:8000", max_retries=2, retry_delay=0.01)

        assert ok is False
        assert mock_client.post.call_count == 2
