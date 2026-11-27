"""
REST API interface for the Manager Agent.

Exposes the multi-agent RAG system over HTTP so it can be queried
without running the interactive REPL. Useful for integration, demos,
and deployment behind a reverse proxy.
"""
from contextlib import asynccontextmanager
from typing import List, Optional
import os

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from loguru import logger

from rag_system.agents.manager import ManagerAgent
from rag_system.utils.logging_config import setup_logging

load_dotenv()

# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="The question to answer")
    top_k: int = Field(5, ge=1, le=20, description="Number of document chunks to retrieve")


class SourceChunk(BaseModel):
    id: str
    content: str
    metadata: dict
    similarity: float


class QueryResponse(BaseModel):
    question: str
    answer: str
    specialist_used: str
    total_time_ms: float
    success: bool
    sources: List[dict] = []


# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------

_manager: Optional[ManagerAgent] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _manager
    setup_logging(
        level=os.getenv("LOG_LEVEL", "INFO"),
        log_dir=os.getenv("LOG_DIR", "logs"),
    )

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key == "your_gemini_api_key_here":
        logger.error("GEMINI_API_KEY not set — manager will not initialize")
        _manager = None
    else:
        mcp_host = os.getenv("MCP_SERVER_HOST", "localhost")
        if mcp_host == "0.0.0.0":
            mcp_host = "localhost"
        mcp_port = os.getenv("MCP_SERVER_PORT", "8000")
        mcp_url = os.getenv("MCP_SERVER_URL", f"http://{mcp_host}:{mcp_port}")
        model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

        _manager = ManagerAgent(
            agent_id="manager_api",
            mcp_url=mcp_url,
            api_key=api_key,
            model_name=model_name,
        )
        logger.info(f"Manager API ready → MCP at {mcp_url}")

    yield
    logger.info("Manager API shutting down")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Multi-Agent RAG API",
    description=(
        "REST interface for the multi-agent document analysis system. "
        "Submit a question and receive a synthesized answer with source citations."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post(
    "/query",
    response_model=QueryResponse,
    summary="Submit a question",
    description="Routes the question through the multi-agent pipeline and returns a synthesized answer.",
)
async def query(request: QueryRequest) -> QueryResponse:
    if _manager is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Manager agent not initialized. Check GEMINI_API_KEY.",
        )

    result = await _manager.answer_question(request.question)

    return QueryResponse(
        question=result["question"],
        answer=result["answer"],
        specialist_used=result.get("specialist_used", "N/A"),
        total_time_ms=result.get("total_time_ms", 0.0),
        success=result.get("success", False),
        sources=result.get("sources", []),
    )


@app.get("/health", summary="Health check")
async def health():
    return {
        "status": "healthy",
        "service": "Manager REST API",
        "version": "1.0.0",
        "manager_ready": _manager is not None,
    }
