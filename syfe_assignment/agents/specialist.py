from fastapi import FastAPI, HTTPException, status
from typing import List, Dict
from loguru import logger
import os

from syfe_assignment.knowledge_base.vector_store import VectorStore
from syfe_assignment.mcp_server.models import Tool, AgentRegistration, Message


class SpecialistAgent:
    """Wraps a ChromaDB vector store behind a FastAPI service"""

    def __init__(
        self,
        agent_id: str,
        name: str,
        description: str,
        collection_name: str,
        port: int = 8001,
        persist_dir: str = "./chroma_db"
    ):
        self.agent_id = agent_id
        self.name = name
        self.description = description
        self.port = port

        self.app = FastAPI(
            title=f"Specialist: {name}",
            description=description,
            version="1.0.0"
        )

        self.vector_store = VectorStore(
            collection_name=collection_name,
            persist_dir=persist_dir
        )

        doc_count = self.vector_store.collection.count()
        if doc_count == 0:
            logger.warning("Knowledge base is empty. Run setup_knowledge_base.py first.")
        else:
            logger.info(f"Loaded {doc_count} documents")

        self._setup_routes()
        logger.info(f"Specialist '{name}' ready")

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
                results = self.vector_store.search(query, top_k=top_k)

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

        @self.app.get("/health", response_model=Dict)
        async def health():
            return {
                "status": "healthy",
                "agent_id": self.agent_id,
                "agent_name": self.name,
                "documents_indexed": self.vector_store.collection.count(),
                "tools": [t.name for t in self.get_tools()]
            }

    def get_registration(self) -> AgentRegistration:
        return AgentRegistration(
            agent_id=self.agent_id,
            name=self.name,
            description=self.description,
            tools=self.get_tools(),
            endpoint=f"http://localhost:{self.port}"
        )
