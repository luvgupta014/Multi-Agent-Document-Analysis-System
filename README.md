# Multi-Agent Document Analysis System

A multi-agent RAG (Retrieval-Augmented Generation) system built with MCP (Model Context Protocol) for the Syfe AI Engineering Intern assignment. The idea is that a Manager Agent orchestrates queries across specialist agents to answer questions that span multiple documents.

I chose MCP as the communication layer because it gives a clean separation between the orchestrator and the retrieval logic — you can swap out specialists without rewriting the manager.

## How it works

The system has three moving parts:

1. **MCP Server** — a FastAPI service that acts as the message router and agent registry. Agents register themselves on startup, and the Manager discovers them at query time.
2. **Specialist Agent** — handles the actual document retrieval. It wraps a ChromaDB vector store and exposes a `document_retriever` tool.
3. **Manager Agent** — the user-facing component. It takes a question, asks Gemini whether it even needs to search documents, and if so, routes the query through MCP to the Specialist. Then it synthesizes a final answer with source citations.

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

## Tech Stack

- Python 3.12
- Google Gemini (gemini-2.5-flash, free tier)
- FastAPI + uvicorn
- ChromaDB for vector storage
- sentence-transformers (all-MiniLM-L6-v2) for embeddings
- httpx for async inter-service calls
- loguru for logging

## Getting Started

### Prerequisites
- Python 3.12+
- A Gemini API key — get one free at https://aistudio.google.com/app/apikey

### Setup

```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

pip install -r requirements.txt
```

Put your API key in `.env`:
```
GEMINI_API_KEY=your_key_here
```

### Index the knowledge base

```bash
python setup_knowledge_base.py
```

This chunks and embeds the 5 sample documents into ChromaDB (~32 chunks total).

### Run the system

You need 3 terminals:

```bash
# Terminal 1
python run_mcp_server.py

# Terminal 2 (wait a few seconds for MCP to start)
python run_specialist.py

# Terminal 3
python run_manager.py
```

Then just type questions in Terminal 3. Try something like:
- "What is the churn model's AUC-ROC?"
- "How does our data pipeline handle streaming?"
- "What is 2+2?" (this one skips retrieval entirely)

## Project Layout

```
syfe_assignment/
├── mcp_server/
│   ├── models.py        # Pydantic schemas for messages, tools, agents
│   ├── registry.py      # In-memory agent registry
│   └── server.py        # FastAPI server with MCP endpoints
├── knowledge_base/
│   ├── loader.py        # Document loading + chunking
│   └── vector_store.py  # ChromaDB wrapper
├── agents/
│   ├── specialist.py    # Document retrieval agent
│   └── manager.py       # Orchestrator agent
└── utils/
    ├── logging_config.py
    └── metrics.py

docs/sample_documents/     # 5 knowledge base files
tests/test_system.py       # Unit tests
config.yaml               # App config
.env                      # Secrets + env vars
Dockerfile + docker-compose.yml
```

## MCP Specification

The server exposes two MCP-compliant endpoints:

- `GET /mcp/v1/tools` — returns all registered tools across agents
- `POST /mcp/v1/tools/{tool_name}/invoke` — invokes a tool by name

Other endpoints handle agent registration (`POST /register`), discovery (`GET /agents`), and message routing (`POST /send`).

## A2A Design Choices

**Why HTTP/JSON?** It's simple, debuggable, and the assignment says "a direct function call or HTTP request is sufficient." I went with HTTP so the agents can run as separate processes, which is closer to how you'd actually deploy this.

**State management:** Each query is stateless — the Manager sends the full context (question + retrieved chunks) to Gemini every time. I didn't implement conversation history since the assignment focuses on single-query analysis, but it would be straightforward to add.

**Specialist personality:** The Specialist uses a system prompt that says "You are a meticulous technical analyst" and enforces citation-only answers. The Manager has a different role — it decides whether to use tools and handles orchestration.

## Knowledge Base

Five sample documents covering different domains:

| Document | Topic |
|----------|-------|
| climate_info.txt | Climate change overview |
| renewable_energy.txt | Solar, wind, hydro, geothermal |
| model_performance_report.txt | Q3 ML model metrics (churn, classification, recommendations) |
| data_pipeline_architecture.txt | Kafka, Airflow, Iceberg, ELT pipeline |
| future_feature_roadmap.txt | Q4 improvements planned for RAG + multi-agent |

Cross-document queries (like "What's the churn model's AUC-ROC and how does the pipeline work?") pull from multiple files to test the retrieval properly.

## Bonus Features

### Source Citations
Answers include references like `[Source: model_performance_report.txt, chunk 3]` so you can trace where the information came from. This is handled through the Specialist's system prompt and by passing chunk metadata through the pipeline.

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
Everything is externalized — API keys, ports, model names, paths — split between `.env` (secrets) and `config.yaml` (app settings). No hardcoded values in the source code.

## Testing

```bash
pytest tests/test_system.py -v
```

31 tests covering models, registry, document loading, MCP endpoints, and metrics.

## Self-Evaluation

**What went well:**
- The MCP abstraction works cleanly — adding a new specialist agent would just mean registering another service
- The LLM-based tool decision is a nice touch. For "What is 2+2?", it correctly skips retrieval entirely
- Source citations make the answers verifiable
- ChromaDB's persistent storage means you don't re-index every time

**What I'd improve with more time:**
- The chunking strategy is basic (sentence-aware splitting with overlap). I'd experiment with semantic chunking or recursive splitting
- Currently there's only one specialist. The architecture supports multiple, but I didn't implement routing between different domain experts
- No re-ranking step — the raw similarity results go straight to Gemini. Adding a cross-encoder re-ranker would improve answer quality
- The tests cover unit-level stuff but don't have integration tests that spin up all three services
- Error handling around the Gemini API could be more robust (retry with backoff instead of just catching exceptions)

**Trade-offs I made:**
- Chose ChromaDB over FAISS because it handles persistence natively, even though FAISS would be faster for pure in-memory search
- Went with stateless queries rather than building conversation history — keeps it simple for the MVP scope
- Used `all-MiniLM-L6-v2` (384-dim) over larger models because it loads fast and the knowledge base is small enough that embedding quality isn't the bottleneck

## Troubleshooting

- **"No specialist agents available"** — Make sure `run_specialist.py` is running and you see "Registered with MCP server" in its output
- **"Knowledge base is empty"** — Run `setup_knowledge_base.py` first
- **429 rate limit** — Gemini free tier has limits. Wait 30s or try a different model in `.env`
- **Port conflict** — Kill existing Python processes or change ports in `.env`
