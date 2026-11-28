from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel
from typing import List, Dict, Optional
from loguru import logger

from rag_system.knowledge_base.vector_store import VectorStore
from rag_system.knowledge_base.base_store import BaseVectorStore, ChromaVectorStore
from rag_system.knowledge_base.loader import Document
from rag_system.knowledge_base.retriever import HybridRetriever
from rag_system.mcp_server.models import Tool, AgentRegistration, Message
from rag_system.utils.config import load_config


class IngestChunk(BaseModel):
    content: str
    metadata: Dict = {}


class IngestRequest(BaseModel):
    documents: List[IngestChunk]


class SpecialistAgent:
    """Wraps a vector store + HybridRetriever behind a FastAPI service."""

    def __init__(
        self,
        agent_id: str,
        name: str,
        description: str,
        collection_name: str,
        port: int = 8001,
        persist_dir: str = "./chroma_db",
        vector_store: Optional[BaseVectorStore] = None,
        retriever: Optional[HybridRetriever] = None,
        hybrid: Optional[bool] = None,
        rerank: Optional[bool] = None,
        rrf_k: Optional[int] = None,
    ):
        cfg = load_config()
        retrieval_cfg = cfg.get("retrieval", {})
        embed_model = cfg.get("embedding", {}).get("model_name", "all-MiniLM-L6-v2")

        self.agent_id = agent_id
        self.name = name
        self.description = description
        self.port = port

        self.app = FastAPI(
            title=f"Specialist: {name}",
            description=description,
            version="1.0.0"
        )

        if vector_store is not None:
            self.vector_store: BaseVectorStore = vector_store
        else:
            # Default: wrap a ChromaDB VectorStore
            raw_vs = VectorStore(
                collection_name=collection_name,
                persist_dir=persist_dir,
                model_name=embed_model,
            )
            self.vector_store = ChromaVectorStore(raw_vs)

        use_hybrid = bool(hybrid) if hybrid is not None else bool(retrieval_cfg.get("hybrid", True))
        use_rerank = bool(rerank) if rerank is not None else bool(retrieval_cfg.get("rerank", False))
        use_rrf_k = int(rrf_k) if rrf_k is not None else int(retrieval_cfg.get("rrf_k", 60))

        self.retriever = retriever or HybridRetriever(
            self.vector_store,
            hybrid=use_hybrid,
            rerank=use_rerank,
            rrf_k=use_rrf_k,
        )

        doc_count = self.vector_store.count()
        if doc_count == 0:
            logger.warning("Knowledge base is empty. Run setup_knowledge_base.py first.")
        else:
            logger.info(f"Loaded {doc_count} documents")

        self._setup_routes()
        logger.info(
            f"Specialist '{name}' ready (hybrid={self.retriever.hybrid}, rerank={self.retriever.rerank})"
        )

    def get_tools(self) -> List[Tool]:
        return [
            Tool(
                name="document_retriever",
                description=f"Retrieve relevant document snippets from the knowledge base about {self.description}",
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query or question"
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Number of results to return (1-20)",
                            "default": 5
                        }
                    },
                    "required": ["query"]
                }
            )
        ]

    def _setup_routes(self):

        @self.app.post("/execute", response_model=Dict)
        async def execute_tool(message: Message):
            if message.tool != "document_retriever":
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Tool '{message.tool}' not supported"
                )

            query = message.parameters.get("query")
            top_k = message.parameters.get("top_k", 5)

            if not query:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Missing required parameter: query"
                )

            try:
                results = self.retriever.search(query, top_k=top_k)
                return {
                    "agent_id": self.agent_id,
                    "agent_name": self.name,
                    "query": query,
                    "results_count": len(results),
                    "results": results
                }

            except Exception as e:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Search failed: {str(e)}"
                )

        @self.app.post("/ingest", response_model=Dict)
        async def ingest_documents(payload: IngestRequest):
            if not payload.documents:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="No documents provided",
                )
            docs = [
                Document(content=item.content, metadata=item.metadata)
                for item in payload.documents
                if item.content.strip()
            ]
            if not docs:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="All provided documents are empty",
                )
            self.vector_store.add_documents(docs)
            self.retriever.refresh_index()
            return {
                "status": "ingested",
                "chunks_added": len(docs),
                "documents_indexed": self.vector_store.count(),
            }

        @self.app.get("/health", response_model=Dict)
        async def health():
            return {
                "status": "healthy",
                "agent_id": self.agent_id,
                "agent_name": self.name,
                "documents_indexed": self.vector_store.count(),
                "tools": [t.name for t in self.get_tools()]
            }

    def get_registration(self, endpoint_host: str = "localhost") -> AgentRegistration:
        return AgentRegistration(
            agent_id=self.agent_id,
            name=self.name,
            description=self.description,
            tools=self.get_tools(),
            endpoint=f"http://{endpoint_host}:{self.port}"
        )
