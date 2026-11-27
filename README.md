# Multi-Agent Document Analysis System

A production-ready multi-agent RAG (Retrieval-Augmented Generation) system using MCP (Model Context Protocol) for orchestrated document retrieval and question answering.

[![Tests](https://github.com/YOUR_USERNAME/multi-agent-rag/actions/workflows/tests.yml/badge.svg)](https://github.com/YOUR_USERNAME/multi-agent-rag/actions)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A Manager Agent orchestrates queries across specialist agents to synthesize answers that span multiple documents, with full source citation and latency tracking.

## How it works

```
User Question
     │
     ▼
Manager Agent ──► decides if retrieval is needed (via Gemini)
     │                          │
     │ (if yes)                 │ (if no, answer directly)
     ▼                          ▼
MCP Server ──► routes to Specialist ──► ChromaDB search
     │
     ▼
Manager gets results back, synthesizes answer with Gemini
```

The system has three moving parts:

1. **MCP Server** — a FastAPI service that acts as the message router and agent registry. Agents register themselves on startup, and the Manager discovers them at query time.
2. **Specialist Agent** — handles the actual document retrieval. It wraps a ChromaDB vector store and exposes a `document_retriever` tool.
3. **Manager Agent** — the user-facing component. It takes a question, asks Gemini whether it even needs to search documents, and if so, routes the query through MCP to the Specialist. Then it synthesizes a final answer with source citations.

## Tech Stack

- Python 3.12
- Google Gemini (gemini-2.5-flash, free tier)
- FastAPI + uvicorn
- ChromaDB for vector storage (cosine similarity)
- sentence-transformers (all-MiniLM-L6-v2) for embeddings
- httpx for async inter-service calls
- tenacity for retry/backoff on API calls
- loguru for structured logging

## Getting Started

### Prerequisites
- Python 3.12+
- A Gemini API key — get one free at https://aistudio.google.com/app/apikey (takes ~30 seconds)

### Setup

**1. Install dependencies:**
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

pip install -r requirements.txt
```

**2. Configure environment:**
```bash
cp .env.example .env   # Mac/Linux
copy .env.example .env  # Windows
```

Open `.env` and **only change** the first line:
```
GEMINI_API_KEY=your_actual_api_key_here
```

**All other variables have working defaults** — ports (8000, 8001, 8002), paths, model name (`gemini-2.5-flash`), etc. No need to touch them unless you want to customize.

**Note on LLM flexibility:** The system currently uses Google Gemini because it has a generous free tier and simple API. You can switch between Gemini models (`gemini-2.5-flash`, `gemini-1.5-flash`, `gemini-1.5-pro`) by changing `GEMINI_MODEL` in `.env`. **To use OpenAI, Anthropic, or other providers**, you'd need to modify [manager.py](rag_system/agents/manager.py) to replace the `google.generativeai` client — the architecture supports this, but Gemini is kept as the default for its free tier.

### Index the knowledge base

```bash
python setup_knowledge_base.py
```

This chunks and embeds the sample documents into ChromaDB. Supports both `.txt` and `.pdf` files.

### Run the system

You need 3 terminals:

```bash
# Terminal 1 — MCP Server (message router)
python run_mcp_server.py

# Terminal 2 — Specialist Agent (wait a few seconds for MCP to start)
python run_specialist.py

# Terminal 3 — Option A: interactive REPL
python run_manager.py

# Terminal 3 — Option B: REST API (query via HTTP)
python run_api.py
```

**REST API usage:**
```bash
curl -X POST http://localhost:8002/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the churn model AUC-ROC?"}'
```

Then just type questions in the manager terminal or use the API. Try:
- `"What is the churn model's AUC-ROC?"`
- `"How does our data pipeline handle streaming?"`
- `"What is 2+2?"` (this one skips retrieval entirely)

## Project Layout

```
rag_system/
├── api/
│   └── app.py           # FastAPI REST interface for the manager
├── mcp_server/
│   ├── models.py        # Pydantic schemas for messages, tools, agents
│   ├── registry.py      # In-memory agent registry
│   └── server.py        # FastAPI server with MCP endpoints
├── knowledge_base/
│   ├── loader.py        # Document loading + chunking (txt + pdf)
│   └── vector_store.py  # ChromaDB wrapper (cosine similarity)
├── agents/
│   ├── specialist.py    # Document retrieval agent
│   └── manager.py       # Orchestrator agent (with retry logic)
└── utils/
    ├── logging_config.py
    └── metrics.py

docs/sample_documents/     # Knowledge base files (txt + pdf supported)
tests/test_system.py       # 35+ unit tests with coverage
config.yaml               # App config
.env                      # Secrets + env vars
Dockerfile + docker-compose.yml
```

## MCP Specification

The server exposes two MCP-compliant endpoints:

- `GET /mcp/v1/tools` — returns all registered tools across agents
- `POST /mcp/v1/tools/{tool_name}/invoke` — invokes a tool by name

Other endpoints handle agent registration (`POST /register`), discovery (`GET /agents`), and message routing (`POST /send`).

## REST API Reference

The Manager REST API (port 8002) exposes:

| Endpoint | Method | Description |
|---|---|---|
| `/query` | POST | Submit a question, get a synthesized answer |
| `/health` | GET | Health check |
| `/docs` | GET | Interactive Swagger UI |

**Query request:**
```json
{
  "question": "string",
  "top_k": 5
}
```

**Query response:**
```json
{
  "question": "string",
  "answer": "string",
  "specialist_used": "string",
  "total_time_ms": 1234.5,
  "success": true,
  "sources": [...]
}
```

## Architecture Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Vector DB | ChromaDB | Native persistence; no separate server process needed for prototype |
| Distance metric | Cosine | Standard for semantic similarity; better calibrated than L2 for text embeddings |
| Embeddings | all-MiniLM-L6-v2 (384-dim) | Fast load time; quality sufficient for small-to-medium corpora |
| LLM | Google Gemini 2.5 Flash | Generous free tier; simple Python SDK; model swappable via env var |
| Agent comms | HTTP/JSON over MCP | Debuggable; allows agents to run as separate processes |
| Query state | Stateless per-query | Keeps context window clean; conversation history is an additive feature |
| Retry strategy | Exponential backoff (tenacity) | Handles 429 rate limits gracefully without crashing |

## Known Limitations & Roadmap

- **Chunking:** Sentence-aware splitting with fixed overlap. Semantic chunking would improve retrieval quality for longer documents.
- **Re-ranking:** Raw similarity results go to Gemini; a cross-encoder re-ranker (e.g. `ms-marco-MiniLM-L-6-v2`) would improve answer precision.
- **Multiple specialists:** Registry supports N specialists; routing logic (beyond keyword scoring) needs an embedding-based similarity approach for production.
- **Conversation history:** Each query is stateless. A session-aware mode with a rolling context window is the next planned feature.
- **Similarity threshold:** Currently configurable via `config.yaml`; threshold tuning on a labeled eval set would improve precision/recall trade-off.

## Knowledge Base

Sample documents covering different domains:

| Document | Topic |
|----------|-------|
| climate_info.txt | Climate change overview |
| renewable_energy.txt | Solar, wind, hydro, geothermal |
| model_performance_report.txt | Q3 ML model metrics (churn, classification, recommendations) |
| data_pipeline_architecture.txt | Kafka, Airflow, Iceberg, ELT pipeline |
| future_feature_roadmap.txt | Q4 improvements planned for RAG + multi-agent |

Cross-document queries (like "What's the churn model's AUC-ROC and how does the pipeline work?") pull from multiple files to test the retrieval properly.

## Features

### Source Citations
Answers include references like `[Source: model_performance_report.txt, chunk 3]` so you can trace where the information came from.

### Logging
Key events are logged with latency tracking:
- Whether the Manager decided to use retrieval or answer directly
- How long the MCP round-trip took
- How long Gemini took to synthesize
- Total end-to-end time

Logs go to `logs/` with automatic rotation at 10MB.

### Docker Support
```bash
docker-compose up --build
```
Spins up all three services. Health checks make sure the MCP server is ready before the Specialist tries to register.

### Configuration
Everything is externalized — API keys, ports, model names, paths, retrieval thresholds — split between `.env` (secrets) and `config.yaml` (app settings). No hardcoded values in the source code.

## Testing

```bash
pytest tests/ -v

# With coverage report:
pytest tests/ -v --cov=rag_system --cov-report=term-missing
```

35+ tests covering models, registry, document loading, MCP endpoints, manager logic, and metrics.

## Troubleshooting

- **"GEMINI_API_KEY not found"** — Make sure you created `.env` from `.env.example` and added your actual API key
- **"No specialist agents available"** — Make sure `run_specialist.py` is running and you see "Registered with MCP server" in its output
- **"Knowledge base is empty"** — Run `setup_knowledge_base.py` first
- **429 rate limit** — The system retries automatically with exponential backoff. If it persists, switch to `gemini-1.5-flash` in `.env`
- **Port conflict** — Kill existing Python processes or change `MCP_SERVER_PORT`/`SPECIALIST_PORT` in `.env`
- **ModuleNotFoundError** — Make sure you activated the virtual environment (`venv\Scripts\activate` on Windows)

**Quick connection test:** After starting all services, visit http://localhost:8000/health in your browser — you should see `{"status":"healthy"}`.
