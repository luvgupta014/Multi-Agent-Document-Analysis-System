# Deploying to Render

This repository includes a [render.yaml](../render.yaml) Blueprint that deploys the four services (`rag-mcp-server`, `rag-specialist`, `rag-api`, and `rag-ui`) on Render.

## 1. Blueprint Deploy (Recommended)

1. Push your repository to GitHub.
2. Open the [Render Dashboard](https://dashboard.render.com/) and select **New → Blueprint**.
3. Connect your repository; Render reads `render.yaml` automatically.
4. Fill in the prompted secrets:
   - `GEMINI_API_KEY` — required by `rag-api` (or configure an alternative provider in `config.yaml`).
   - `API_KEY` — optional shared secret for `X-API-Key` auth (set the same value on `rag-api` and `rag-ui`, or leave blank to disable auth).
5. Click **Apply**.

Every Python entry script (`run_mcp_server.py`, `run_specialist.py`, `run_api.py`) checks Render's injected `PORT` environment variable automatically before falling back to local port defaults.

## 2. Single Web Service (Docker)

To deploy just the API container using the root `Dockerfile`:

1. Select **New → Web Service** and connect your repository.
2. Choose **Docker** as the runtime.
3. Set the start command if overriding the default:
   ```bash
   python run_api.py
   ```
4. Configure environment variables:
   - `GEMINI_API_KEY`
   - `MCP_SERVER_URL` (for example `https://rag-mcp-server.onrender.com`)
   - `API_KEY` (optional)
   - `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` (optional)

## 3. Production Notes

- **Persistence:** Render's free tier uses an ephemeral filesystem, so `chroma_db/` resets on redeploy unless you attach a persistent disk mounted at `/app/chroma_db`.
- **Pre-indexing:** `rag-specialist` runs `FORCE_RELOAD=true python setup_knowledge_base.py` during the build step so the initial corpus is embedded before boot.
- **Cold starts:** Free Render services spin down after 15 minutes of inactivity; the first request after idle takes ~30–60s while services wake and load `all-MiniLM-L6-v2`.
