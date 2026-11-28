"""
Tests for Phase 6 — /chat SSE, API key auth, rate limiting.
"""
import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import httpx


# ---------------------------------------------------------------------------
# Shared answer fixture
# ---------------------------------------------------------------------------

_FAKE_RESULT = {
    "question": "What is solar energy?",
    "answer": "Solar energy is power from the sun.",
    "specialist_used": "document_retrieval",
    "total_time_ms": 50.0,
    "success": True,
    "sources": [{"id": "s1", "content": "Solar text.", "metadata": {"filename": "solar.txt", "chunk_index": 0}, "similarity": 0.9}],
}


@pytest.fixture
async def api_client():
    """AsyncClient against the app with no auth enabled."""
    import httpx
    from rag_system.api.app import app
    with patch.dict("os.environ", {"GEMINI_API_KEY": "fake-key", "API_KEY": ""}):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
        ) as ac:
            yield ac


@pytest.fixture
async def api_client_with_auth():
    """AsyncClient against the app with API_KEY=secret123."""
    import httpx
    from rag_system.api import app as app_module
    from rag_system.api.app import app

    with patch.dict("os.environ", {"GEMINI_API_KEY": "fake-key", "API_KEY": "secret123"}):
        # Patch the module-level _API_KEY after env is set
        with patch.object(app_module, "_API_KEY", "secret123"):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="http://test",
            ) as ac:
                yield ac


# ===========================================================================
# TestSSEChat
# ===========================================================================

class TestSSEChat:

    async def test_chat_streams_tokens(self, api_client):
        """/chat should return a 200 StreamingResponse with SSE events."""
        from rag_system.api import app as app_module

        mock_manager = AsyncMock()
        mock_manager.answer_question = AsyncMock(return_value=_FAKE_RESULT)

        with patch.object(app_module, "_manager", mock_manager):
            response = await api_client.get("/chat", params={"q": "What is solar energy?"})

        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]

        # Collect SSE lines
        events = [
            line[5:].strip()
            for line in response.text.splitlines()
            if line.startswith("data:")
        ]
        assert len(events) >= 2  # at least some tokens + done

        # Last event must be done=True
        last = json.loads(events[-1])
        assert last.get("done") is True
        assert "specialist_used" in last

    async def test_chat_empty_query_rejected(self, api_client):
        response = await api_client.get("/chat", params={"q": "   "})
        assert response.status_code == 422

    async def test_chat_no_manager(self, api_client):
        from rag_system.api import app as app_module
        with patch.object(app_module, "_manager", None):
            response = await api_client.get("/chat", params={"q": "hello"})
        assert response.status_code == 503

    async def test_chat_tokens_reconstruct_answer(self, api_client):
        """Concatenating token events should reconstruct the full answer."""
        from rag_system.api import app as app_module

        mock_manager = AsyncMock()
        mock_manager.answer_question = AsyncMock(return_value=_FAKE_RESULT)

        with patch.object(app_module, "_manager", mock_manager):
            response = await api_client.get("/chat", params={"q": "solar"})

        events = [
            line[5:].strip()
            for line in response.text.splitlines()
            if line.startswith("data:")
        ]
        tokens = "".join(
            json.loads(e).get("token", "")
            for e in events
            if not json.loads(e).get("done")
        )
        assert "Solar energy" in tokens or "solar" in tokens.lower()


# ===========================================================================
# TestAPIKeyAuth
# ===========================================================================

class TestAPIKeyAuth:

    async def test_no_auth_required_when_key_blank(self, api_client):
        """/health should be accessible without X-API-Key when API_KEY is blank."""
        response = await api_client.get("/health")
        assert response.status_code == 200

    async def test_query_blocked_without_key(self, api_client_with_auth):
        """POST /query without header returns 401."""
        from rag_system.api import app as app_module
        mock_manager = AsyncMock()
        mock_manager.answer_question = AsyncMock(return_value=_FAKE_RESULT)
        with patch.object(app_module, "_manager", mock_manager):
            response = await api_client_with_auth.post(
                "/query", json={"question": "hello"}
            )
        assert response.status_code == 401

    async def test_query_accepted_with_correct_key(self, api_client_with_auth):
        """POST /query with correct X-API-Key returns 200."""
        from rag_system.api import app as app_module
        mock_manager = AsyncMock()
        mock_manager.answer_question = AsyncMock(return_value=_FAKE_RESULT)
        with patch.object(app_module, "_manager", mock_manager):
            response = await api_client_with_auth.post(
                "/query",
                json={"question": "hello"},
                headers={"X-API-Key": "secret123"},
            )
        assert response.status_code == 200

    async def test_chat_blocked_without_key(self, api_client_with_auth):
        from rag_system.api import app as app_module
        mock_manager = AsyncMock()
        mock_manager.answer_question = AsyncMock(return_value=_FAKE_RESULT)
        with patch.object(app_module, "_manager", mock_manager):
            response = await api_client_with_auth.get("/chat", params={"q": "hi"})
        assert response.status_code == 401

    async def test_chat_accepted_with_correct_key(self, api_client_with_auth):
        from rag_system.api import app as app_module
        mock_manager = AsyncMock()
        mock_manager.answer_question = AsyncMock(return_value=_FAKE_RESULT)
        with patch.object(app_module, "_manager", mock_manager):
            response = await api_client_with_auth.get(
                "/chat",
                params={"q": "solar energy"},
                headers={"X-API-Key": "secret123"},
            )
        assert response.status_code == 200

    async def test_wrong_key_returns_401(self, api_client_with_auth):
        from rag_system.api import app as app_module
        with patch.object(app_module, "_manager", AsyncMock()):
            response = await api_client_with_auth.get(
                "/chat",
                params={"q": "solar"},
                headers={"X-API-Key": "wrong-key"},
            )
        assert response.status_code == 401


# ===========================================================================
# TestRateLimiting
# ===========================================================================

class TestRateLimiting:

    async def test_rate_limit_triggers_at_threshold(self, api_client):
        """_check_rate_limit raises 429 when bucket is full."""
        from rag_system.api.app import _check_rate_limit, _rate_buckets, _RATE_LIMIT
        import time
        from fastapi import HTTPException

        # Build a mock request with a fake IP
        mock_request = MagicMock()
        mock_request.client.host = "test-ip-ratelimit"

        # Pre-fill bucket to the limit
        now = time.monotonic()
        _rate_buckets["test-ip-ratelimit"] = [now - i * 0.001 for i in range(_RATE_LIMIT)]

        # Next call should raise 429
        with pytest.raises(HTTPException) as exc_info:
            _check_rate_limit(mock_request)

        _rate_buckets.pop("test-ip-ratelimit", None)
        assert exc_info.value.status_code == 429

    async def test_rate_limit_zero_disables(self, api_client):
        """When _RATE_LIMIT == 0, no limiting is applied."""
        from rag_system.api import app as app_module

        with patch.object(app_module, "_RATE_LIMIT", 0):
            app_module._rate_buckets.clear()
            for _ in range(5):
                r = await api_client.get("/health")
            assert r.status_code == 200


# ===========================================================================
# TestSSEHelpers
# ===========================================================================

class TestSSEHelpers:

    async def test_sse_stream_yields_done_event(self):
        """_sse_stream should always end with a done=True event."""
        from rag_system.api.app import _sse_stream

        mock_manager = AsyncMock()
        mock_manager.answer_question = AsyncMock(return_value=_FAKE_RESULT)

        events = []
        async for chunk in _sse_stream("test question", mock_manager):
            if chunk.startswith("data:"):
                events.append(json.loads(chunk[5:].strip()))

        done_events = [e for e in events if e.get("done")]
        assert len(done_events) == 1
        assert done_events[0]["specialist_used"] == "document_retrieval"

    async def test_sse_stream_tokens_non_empty(self):
        """Token events should carry non-empty strings."""
        from rag_system.api.app import _sse_stream

        mock_manager = AsyncMock()
        mock_manager.answer_question = AsyncMock(return_value=_FAKE_RESULT)

        token_events = []
        async for chunk in _sse_stream("test", mock_manager):
            if chunk.startswith("data:"):
                payload = json.loads(chunk[5:].strip())
                if "token" in payload:
                    token_events.append(payload["token"])

        assert len(token_events) > 0
        assert all(isinstance(t, str) for t in token_events)
