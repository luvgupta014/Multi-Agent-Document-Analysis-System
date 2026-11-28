# Contributing

Thanks for contributing to the Multi-Agent Document Analysis System.

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/Multi-Agent-Document-Analysis-System
cd Multi-Agent-Document-Analysis-System

python -m venv venv
source venv/bin/activate      # Mac/Linux
venv\Scripts\activate         # Windows

pip install -r requirements.txt
cp .env.example .env          # add your GEMINI_API_KEY
```

## Running Locally

```bash
# 1. Index documents
python setup_knowledge_base.py

# 2. Start services
python run_mcp_server.py
python run_specialist.py
python run_manager.py          # interactive CLI
# OR
python run_api.py              # REST API on :8002
streamlit run ui/app.py        # Streamlit UI on :8501
```

## Lint and Tests

```bash
ruff check rag_system tests
pytest tests/ -v
pytest tests/ -v --cov=rag_system --cov-report=term-missing
```

## Project Layout

```text
rag_system/
├── agents/           # Manager, Document/SQL/Web specialists, ConversationMemory
├── api/              # FastAPI REST & SSE server, auth, rate limit, /upload
├── knowledge_base/   # DocumentLoader (txt/pdf/docx), ChromaVectorStore, HybridRetriever
├── llm/              # BaseLLM factory + Gemini, OpenAI, Ollama backends
├── mcp_server/       # FastAPI MCP server, Pydantic models, registry
└── utils/            # Logging, metrics, prompt-injection filter, Langfuse tracer
ui/                   # Streamlit chat frontend
eval/                 # 30-question golden QA set and evaluation runner
tests/                # Unit and integration tests
.github/workflows/    # CI (ruff lint, pytest coverage, docker build)
```

## Code Style

- Keep functions small and readable; externalize configuration to `config.yaml` and `.env`.
- Run `ruff check rag_system tests` before opening a PR.
- Add tests under `tests/` for new features or bug fixes.

## Submitting a PR

1. Create a branch: `git checkout -b feature/your-feature`
2. Make changes and add tests
3. Verify `ruff check rag_system tests` and `pytest tests/ -v` pass
4. Push and open a Pull Request against `main`
