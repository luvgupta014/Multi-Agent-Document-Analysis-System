"""
REST API interface for the Manager Agent.

Exposes the multi-agent RAG system over HTTP so it can be queried
without running the interactive REPL. Useful for integration, demos,
and deployment behind a reverse proxy.

Authentication
--------------
Protected endpoints require an ``X-API-Key`` header whose value matches
the ``API_KEY`` environment variable (or ``config.yaml`` → ``api.key``).
Leave ``API_KEY`` unset to disable auth (useful during local development).

Rate limiting
-------------
A simple in-memory token-bucket is applied per client IP:
``api.rate_limit_per_minute`` requests per 60-second window (default 60).
Returns HTTP 429 when exceeded.
"""
from contextlib import asynccontextmanager
from typing import AsyncGenerator, List, Optional
import asyncio
import json
import os
import shutil
import tempfile
import time
from collections import defaultdict
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, UploadFile, File, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import httpx
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from loguru import logger

from rag_system.agents.manager import ManagerAgent
from rag_system.llm.base import LLMError
from rag_system.llm.factory import create_llm
from rag_system.utils.config import load_config
from rag_system.utils.logging_config import setup_logging

load_dotenv()

# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

_cfg = load_config()

# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

_API_KEY: Optional[str] = os.getenv("API_KEY") or _cfg.get("api", {}).get("key")


def _check_api_key(request: Request) -> None:
    """Raise HTTP 401 if the X-API-Key header doesn't match _API_KEY."""
    if not _API_KEY:
        return  # auth disabled
    provided = request.headers.get("X-API-Key", "")
    if provided != _API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing X-API-Key header.",
        )


# ---------------------------------------------------------------------------
# Rate limiting (simple in-memory sliding window)
# ---------------------------------------------------------------------------

_RATE_LIMIT: int = int(
    os.getenv("RATE_LIMIT_PER_MINUTE")
    or _cfg.get("api", {}).get("rate_limit_per_minute", 60)
)
_rate_buckets: dict = defaultdict(list)  # ip -> [timestamps]


def _check_rate_limit(request: Request) -> None:
    """Raise HTTP 429 if the client IP has exceeded the per-minute limit."""
    if _RATE_LIMIT <= 0:
        return  # disabled
    ip = request.client.host if request.client else "unknown"
    now = time.monotonic()
    window = 60.0
    _rate_buckets[ip] = [t for t in _rate_buckets[ip] if now - t < window]
    if len(_rate_buckets[ip]) >= _RATE_LIMIT:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded. Max {_RATE_LIMIT} requests per minute.",
        )
    _rate_buckets[ip].append(now)


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


class UploadResponse(BaseModel):
    filename: str
    chunks_indexed: int
    file_type: str
    success: bool


# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------

_manager: Optional[ManagerAgent] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _manager
    setup_logging(
        service_name="manager_api",
        log_dir=os.getenv("LOG_DIR", "logs"),
        level=os.getenv("LOG_LEVEL", "INFO"),
    )

    try:
        provider = _cfg.get("llm", {}).get("provider", "gemini").lower()
        if provider == "gemini" and os.getenv("GEMINI_API_KEY", "").strip() == "your_gemini_api_key_here":
            raise LLMError("GEMINI_API_KEY still has placeholder value.")

        llm = create_llm(_cfg)
        mcp_host = os.getenv("MCP_SERVER_HOST", "localhost")
        if mcp_host == "0.0.0.0":
            mcp_host = "localhost"
        mcp_port = os.getenv("MCP_SERVER_PORT", "8000")
        mcp_url = os.getenv("MCP_SERVER_URL", f"http://{mcp_host}:{mcp_port}")

        _manager = ManagerAgent(
            agent_id="manager_api",
            mcp_url=mcp_url,
            llm=llm,
        )
        logger.info(f"Manager API ready → MCP at {mcp_url} (llm={llm})")
    except LLMError as exc:
        logger.error(f"LLM not configured ({exc}) — manager will not initialize")
        _manager = None

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
# SSE helpers
# ---------------------------------------------------------------------------

async def _sse_stream(question: str, manager: ManagerAgent) -> AsyncGenerator[str, None]:
    """
    Run answer_question() and yield the answer word-by-word as SSE events,
    followed by a final ``[DONE]`` event carrying sources as JSON.

    Format:  data: <token>\\n\\n
    """
    result = await manager.answer_question(question)
    answer: str = result.get("answer", "")
    sources: list = result.get("sources", [])
    specialist: str = result.get("specialist_used", "N/A")

    # Stream the answer token by token (word-level for readability)
    for word in answer.split(" "):
        payload = json.dumps({"token": word + " "})
        yield f"data: {payload}\n\n"
        await asyncio.sleep(0)  # yield to event loop between tokens

    # Final event with metadata
    done_payload = json.dumps({
        "done": True,
        "specialist_used": specialist,
        "sources": sources,
    })
    yield f"data: {done_payload}\n\n"


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get(
    "/chat",
    summary="Streaming chat (SSE)",
    description=(
        "Stream an answer word-by-word via Server-Sent Events. "
        "Pass the question as the ``q`` query parameter. "
        "Requires ``X-API-Key`` header when auth is enabled."
    ),
)
async def chat(request: Request, q: str) -> StreamingResponse:
    _check_api_key(request)
    _check_rate_limit(request)

    if not q.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Query parameter 'q' must not be empty.",
        )

    if _manager is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Manager agent not initialized. Check GEMINI_API_KEY.",
        )

    return StreamingResponse(
        _sse_stream(q, _manager),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.post(
    "/query",
    response_model=QueryResponse,
    summary="Submit a question",
    description="Routes the question through the multi-agent pipeline and returns a synthesized answer.",
)
async def query(request: Request, body: QueryRequest) -> QueryResponse:
    _check_api_key(request)
    _check_rate_limit(request)

    if _manager is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Manager agent not initialized. Check GEMINI_API_KEY.",
        )

    result = await _manager.answer_question(body.question, top_k=body.top_k)

    return QueryResponse(
        question=result["question"],
        answer=result["answer"],
        specialist_used=result.get("specialist_used", "N/A"),
        total_time_ms=result.get("total_time_ms", 0.0),
        success=result.get("success", False),
        sources=result.get("sources", []),
    )


@app.post(
    "/upload",
    response_model=UploadResponse,
    summary="Upload a document",
    description=(
        "Upload a .txt, .pdf, or .docx file. The document is chunked and "
        "indexed into the vector store so it can be retrieved immediately."
    ),
)
async def upload(request: Request, file: UploadFile = File(...)) -> UploadResponse:
    """Ingest a new document into the knowledge base."""
    _check_api_key(request)
    _check_rate_limit(request)

    allowed = {".txt", ".pdf", ".docx"}
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in allowed:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Unsupported file type '{suffix}'. Allowed: {', '.join(sorted(allowed))}",
        )

    if _manager is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Manager agent not initialized. Check GEMINI_API_KEY.",
        )

    tmp_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = Path(tmp.name)

        from rag_system.knowledge_base.loader import DocumentLoader

        chunk_size = int(_cfg.get("knowledge_base", {}).get("chunk_size", 500))
        strategy = str(_cfg.get("chunking", {}).get("strategy", "recursive"))

        loaders = {
            ".txt": DocumentLoader.load_text_file,
            ".pdf": DocumentLoader.load_pdf_file,
            ".docx": DocumentLoader.load_docx_file,
        }
        docs = loaders[suffix](str(tmp_path), chunk_size=chunk_size, strategy=strategy)

        if not docs:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="No text could be extracted from the uploaded file.",
            )

        for doc in docs:
            doc.metadata["filename"] = file.filename or "uploaded"
            doc.metadata["source"] = file.filename or "uploaded"

        # Support direct specialist injection (used in unit tests) as well as live MCP discovery
        if hasattr(_manager, "specialist") and getattr(_manager, "specialist", None) is not None:
            _manager.specialist.vector_store.add_documents(docs)
        else:
            ingested_via_mcp = False
            payload = {
                "documents": [
                    {"content": d.content, "metadata": d.metadata}
                    for d in docs
                ]
            }
            try:
                specialists = await _manager.discover_specialists()
                doc_specialists = [
                    s for s in specialists
                    if any(t.get("name") == "document_retriever" for t in s.get("tools", []))
                ]
                async with httpx.AsyncClient(timeout=30.0) as client:
                    for spec in doc_specialists:
                        resp = await client.post(f"{spec['endpoint'].rstrip('/')}/ingest", json=payload)
                        resp.raise_for_status()
                        ingested_via_mcp = True
            except Exception as exc:
                logger.warning(f"MCP specialist ingest unavailable ({exc}), writing locally")

            if not ingested_via_mcp:
                from rag_system.knowledge_base.base_store import create_vector_store
                create_vector_store(_cfg).add_documents(docs)

        logger.info(f"Uploaded '{file.filename}': {len(docs)} chunks indexed")
        return UploadResponse(
            filename=file.filename or "uploaded",
            chunks_indexed=len(docs),
            file_type=suffix.lstrip("."),
            success=True,
        )
    finally:
        if tmp_path and tmp_path.exists():
            tmp_path.unlink()


@app.get("/health", summary="Health check")
async def health():
    return {
        "status": "healthy",
        "service": "Manager REST API",
        "version": "1.0.0",
        "manager_ready": _manager is not None,
    }
