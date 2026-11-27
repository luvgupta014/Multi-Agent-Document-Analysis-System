# Contributing

Thanks for your interest in contributing to the Multi-Agent RAG project!

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/multi-agent-rag
cd multi-agent-rag

python -m venv venv
source venv/bin/activate      # Mac/Linux
venv\Scripts\activate         # Windows

pip install -r requirements.txt
cp .env.example .env          # add your GEMINI_API_KEY
```

## Running the system

```bash
# 1. Index documents
python setup_knowledge_base.py

# 2. Start services (3 terminals)
python run_mcp_server.py
python run_specialist.py
python run_manager.py          # interactive REPL
# OR
python run_api.py              # REST API on :8002
```

## Running tests

```bash
pytest tests/ -v

# With coverage:
pytest tests/ -v --cov=rag_system --cov-report=term-missing
```

## Project layout

```
rag_system/
├── api/              # FastAPI REST interface for the manager
├── mcp_server/       # MCP protocol: models, registry, server
├── knowledge_base/   # Document loading (txt+pdf) and ChromaDB wrapper
├── agents/           # Manager (orchestrator) and Specialist (retrieval)
└── utils/            # Logging config and metrics collector

tests/                # 35+ unit tests
docs/                 # Sample knowledge-base documents
.github/workflows/    # GitHub Actions CI
```

## Code style

- Follow PEP 8. Black-formatted code preferred (88 char line length).
- All public functions should have a one-line docstring.
- Keep type annotations on all function signatures.

## Submitting a PR

1. Create a branch: `git checkout -b feature/your-feature`
2. Make your changes and add/update tests
3. Make sure tests pass: `pytest tests/ -v`
4. Push and open a Pull Request against `main`
