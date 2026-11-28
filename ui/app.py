"""
Streamlit chat UI for the Multi-Agent RAG System.

Connects to the Manager REST API (default: http://localhost:8002).
Streams answers word-by-word via the /chat SSE endpoint and shows
source citations and agent trace in expandable panels.

Usage:
    streamlit run ui/app.py

Environment variables (optional):
    API_BASE_URL   Manager API base URL   (default: http://localhost:8002)
    API_KEY        X-API-Key header value (default: empty / auth disabled)
"""
import json
import os
import time
from typing import Optional

import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

API_BASE_URL: str = os.getenv("API_BASE_URL", "http://localhost:8002")
API_KEY: str = os.getenv("API_KEY", "")


def _headers() -> dict:
    h = {"Accept": "text/event-stream"}
    if API_KEY:
        h["X-API-Key"] = API_KEY
    return h


# ---------------------------------------------------------------------------
# Page layout
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Multi-Agent RAG",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 Multi-Agent Document Analysis")
st.caption(
    "Ask questions about your documents. The system routes each query to the best "
    "specialist: document retrieval, SQL database, web search, or direct answer."
)

# Sidebar — settings and upload
with st.sidebar:
    st.header("⚙️ Settings")
    base_url = st.text_input("API base URL", value=API_BASE_URL)
    api_key_input = st.text_input("API key (optional)", value=API_KEY, type="password")

    st.divider()
    st.header("📄 Upload Document")
    uploaded_file = st.file_uploader(
        "Upload .txt, .pdf, or .docx", type=["txt", "pdf", "docx"]
    )
    if uploaded_file and st.button("Index document"):
        with st.spinner("Indexing…"):
            headers_upload = {}
            if api_key_input:
                headers_upload["X-API-Key"] = api_key_input
            try:
                resp = requests.post(
                    f"{base_url}/upload",
                    files={"file": (uploaded_file.name, uploaded_file, uploaded_file.type)},
                    headers=headers_upload,
                    timeout=60,
                )
                if resp.ok:
                    data = resp.json()
                    st.success(
                        f"✅ Indexed **{data['filename']}** — "
                        f"{data['chunks_indexed']} chunks ({data['file_type'].upper()})"
                    )
                else:
                    st.error(f"Upload failed: {resp.status_code} — {resp.text}")
            except requests.RequestException as exc:
                st.error(f"Connection error: {exc}")

    st.divider()
    if st.button("🗑️ Clear conversation"):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.markdown(
        f"**API:** `{base_url}`\n\n"
        "**Agents:** Document · SQL · Web · Direct"
    )

# ---------------------------------------------------------------------------
# Chat history
# ---------------------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display prior messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander(f"📚 Sources ({len(msg['sources'])})"):
                for src in msg["sources"]:
                    fn = src.get("metadata", {}).get("filename", "unknown")
                    chunk = src.get("metadata", {}).get("chunk_index", "?")
                    score = src.get("similarity", 0)
                    st.markdown(f"- **{fn}** chunk {chunk} (score: {score:.3f})")
                    st.caption(src.get("content", "")[:200] + "…")
        if msg["role"] == "assistant" and msg.get("specialist_used"):
            st.caption(f"🔀 Agent: **{msg['specialist_used']}**")

# ---------------------------------------------------------------------------
# Chat input
# ---------------------------------------------------------------------------

if prompt := st.chat_input("Ask a question…"):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Stream assistant response
    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_text = ""
        sources: list = []
        specialist_used: str = "N/A"
        error_msg: Optional[str] = None

        try:
            headers_stream = _headers()
            if api_key_input:
                headers_stream["X-API-Key"] = api_key_input

            with requests.get(
                f"{base_url}/chat",
                params={"q": prompt},
                headers=headers_stream,
                stream=True,
                timeout=120,
            ) as resp:
                if resp.status_code == 401:
                    error_msg = "❌ Unauthorized — check your API key."
                elif resp.status_code == 429:
                    error_msg = "⏱️ Rate limit exceeded. Please wait a moment."
                elif resp.status_code == 503:
                    error_msg = "🔴 API not ready — is `GEMINI_API_KEY` set on the server?"
                elif not resp.ok:
                    error_msg = f"❌ Error {resp.status_code}: {resp.text[:200]}"
                else:
                    for raw_line in resp.iter_lines():
                        if not raw_line:
                            continue
                        line = raw_line if isinstance(raw_line, str) else raw_line.decode()
                        if not line.startswith("data:"):
                            continue
                        payload_str = line[5:].strip()
                        try:
                            payload = json.loads(payload_str)
                        except json.JSONDecodeError:
                            continue

                        if payload.get("done"):
                            sources = payload.get("sources", [])
                            specialist_used = payload.get("specialist_used", "N/A")
                        else:
                            full_text += payload.get("token", "")
                            placeholder.markdown(full_text + "▌")

                    placeholder.markdown(full_text)

        except requests.ConnectionError:
            error_msg = (
                f"❌ Cannot connect to `{base_url}`. "
                "Make sure the API server is running."
            )
        except requests.Timeout:
            error_msg = "⏱️ Request timed out."

        if error_msg:
            st.error(error_msg)
            full_text = error_msg

        # Show sources and agent trace
        if sources:
            with st.expander(f"📚 Sources ({len(sources)})"):
                for src in sources:
                    fn = src.get("metadata", {}).get("filename", "unknown")
                    chunk = src.get("metadata", {}).get("chunk_index", "?")
                    score = src.get("similarity", 0)
                    st.markdown(f"- **{fn}** chunk {chunk} (score: {score:.3f})")
                    st.caption(src.get("content", "")[:200] + "…")
        if specialist_used and specialist_used != "N/A":
            st.caption(f"🔀 Agent: **{specialist_used}**")

    # Save to history
    st.session_state.messages.append({
        "role": "assistant",
        "content": full_text,
        "sources": sources,
        "specialist_used": specialist_used,
    })
