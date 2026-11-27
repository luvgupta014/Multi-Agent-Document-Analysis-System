"""
Unit tests for Phase 3: LLM abstraction and BaseVectorStore.

All tests are mock-based — no API keys, no network, no ChromaDB needed.
"""
import sys
import os
import pytest
from unittest.mock import MagicMock, patch, PropertyMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rag_system.llm.base import BaseLLM, LLMConfig, LLMError
from rag_system.llm.factory import create_llm


# ---------------------------------------------------------------------------
# LLMConfig
# ---------------------------------------------------------------------------

class TestLLMConfig:

    def test_defaults(self):
        cfg = LLMConfig()
        assert cfg.provider == "gemini"
        assert cfg.temperature == 0.3
        assert cfg.max_tokens == 2048
        assert cfg.api_key is None

    def test_custom_values(self):
        cfg = LLMConfig(provider="openai", model_name="gpt-4o", temperature=0.0)
        assert cfg.provider == "openai"
        assert cfg.model_name == "gpt-4o"
        assert cfg.temperature == 0.0


# ---------------------------------------------------------------------------
# BaseLLM is abstract
# ---------------------------------------------------------------------------

class TestBaseLLMAbstract:

    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            BaseLLM(LLMConfig())  # type: ignore

    def test_concrete_subclass_works(self):
        class FakeLLM(BaseLLM):
            def generate(self, prompt: str) -> str:
                return f"echo: {prompt}"

        llm = FakeLLM(LLMConfig(model_name="fake"))
        assert llm.generate("hello") == "echo: hello"
        assert "FakeLLM" in repr(llm)
        assert "fake" in repr(llm)


# ---------------------------------------------------------------------------
# LLM factory
# ---------------------------------------------------------------------------

class TestCreateLLM:

    def test_unknown_provider_raises(self):
        with pytest.raises(LLMError, match="Unknown LLM provider"):
            create_llm({"llm": {"provider": "unknown_provider"}})

    def test_gemini_provider_selected(self):
        cfg = {
            "llm": {"provider": "gemini", "model_name": "gemini-2.5-flash"},
        }
        mock_model = MagicMock()
        with patch("google.generativeai.configure"), \
             patch("google.generativeai.GenerativeModel", return_value=mock_model), \
             patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            from rag_system.llm.gemini import GeminiLLM
            llm = create_llm(cfg)
            assert isinstance(llm, GeminiLLM)

    def test_openai_provider_selected(self):
        cfg = {"llm": {"provider": "openai", "model_name": "gpt-4o-mini"}}
        mock_openai_mod = MagicMock()
        mock_client = MagicMock()
        mock_openai_mod.OpenAI.return_value = mock_client
        with patch.dict("sys.modules", {"openai": mock_openai_mod}), \
             patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            from rag_system.llm import openai_llm
            import importlib; importlib.reload(openai_llm)
            from rag_system.llm.openai_llm import OpenAILLM
            llm = create_llm(cfg)
            assert isinstance(llm, OpenAILLM)

    def test_ollama_provider_selected(self):
        cfg = {"llm": {"provider": "ollama", "model_name": "llama3.2"}}
        from rag_system.llm.ollama import OllamaLLM
        llm = create_llm(cfg)
        assert isinstance(llm, OllamaLLM)

    def test_legacy_gemini_section_fallback(self):
        """If llm section absent, reads from legacy gemini section."""
        cfg = {"gemini": {"model_name": "gemini-1.5-flash", "temperature": 0.5}}
        with patch("google.generativeai.configure"), \
             patch("google.generativeai.GenerativeModel"), \
             patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            llm = create_llm(cfg)
            assert llm.config.model_name == "gemini-1.5-flash"
            assert llm.config.temperature == 0.5


# ---------------------------------------------------------------------------
# GeminiLLM
# ---------------------------------------------------------------------------

class TestGeminiLLM:

    def _make(self):
        from rag_system.llm.gemini import GeminiLLM
        with patch("google.generativeai.configure"), \
             patch("google.generativeai.GenerativeModel") as MockModel:
            mock_model = MagicMock()
            MockModel.return_value = mock_model
            llm = GeminiLLM(LLMConfig(provider="gemini", api_key="fake"))
            llm._model = mock_model
            return llm, mock_model

    def test_generate_returns_text(self):
        llm, mock_model = self._make()
        mock_model.generate_content.return_value = MagicMock(text="Hello world")
        assert llm.generate("say hi") == "Hello world"

    def test_no_api_key_raises(self):
        from rag_system.llm.gemini import GeminiLLM
        with patch("google.generativeai.configure"), \
             patch("google.generativeai.GenerativeModel"):
            with pytest.raises(LLMError, match="GEMINI_API_KEY"):
                GeminiLLM(LLMConfig(provider="gemini", api_key=None))

    def test_exception_wrapped_in_llm_error(self):
        llm, mock_model = self._make()
        mock_model.generate_content.side_effect = RuntimeError("boom")
        with pytest.raises(LLMError, match="Gemini generate failed"):
            llm.generate("test")


# ---------------------------------------------------------------------------
# OllamaLLM
# ---------------------------------------------------------------------------

class TestOllamaLLM:

    def _make(self, base_url="http://localhost:11434"):
        from rag_system.llm.ollama import OllamaLLM
        return OllamaLLM(LLMConfig(provider="ollama", model_name="llama3.2", base_url=base_url))

    def test_generate_ok(self):
        llm = self._make()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"response": "Paris"}
        mock_resp.raise_for_status = MagicMock()
        with patch("httpx.post", return_value=mock_resp):
            assert llm.generate("capital of France?") == "Paris"

    def test_connection_error_raises_llm_error(self):
        import httpx
        llm = self._make()
        with patch("httpx.post", side_effect=httpx.RequestError("refused")):
            with pytest.raises(LLMError, match="Ollama unreachable"):
                llm.generate("test")

    def test_base_url_stripped(self):
        llm = self._make(base_url="http://localhost:11434/")
        assert not llm._base_url.endswith("/")


# ---------------------------------------------------------------------------
# OpenAILLM
# ---------------------------------------------------------------------------

class TestOpenAILLM:

    def test_generate_ok(self):
        from rag_system.llm.openai_llm import OpenAILLM
        # Bypass __init__ (requires openai package) and test generate() directly
        mock_client = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "42"
        mock_client.chat.completions.create.return_value = MagicMock(choices=[mock_choice])

        llm = object.__new__(OpenAILLM)
        llm.config = LLMConfig(provider="openai", model_name="gpt-4o-mini", api_key="k")
        llm._client = mock_client
        assert llm.generate("answer?") == "42"

    def test_missing_package_raises(self):
        from rag_system.llm.openai_llm import OpenAILLM
        with patch.dict("sys.modules", {"openai": None}):
            with pytest.raises(LLMError, match="openai package"):
                OpenAILLM(LLMConfig(provider="openai", api_key="k"))


# ---------------------------------------------------------------------------
# ChromaVectorStore adapter
# ---------------------------------------------------------------------------

class TestChromaVectorStore:

    def _make(self):
        from rag_system.knowledge_base.base_store import ChromaVectorStore
        mock_vs = MagicMock()
        mock_vs.collection.count.return_value = 10
        mock_vs.get_stats.return_value = {"document_count": 10}
        mock_vs.search.return_value = [{"id": "a", "content": "x", "similarity": 0.9}]
        return ChromaVectorStore(mock_vs), mock_vs

    def test_count(self):
        store, _ = self._make()
        assert store.count() == 10

    def test_search_delegates(self):
        store, mock_vs = self._make()
        results = store.search("climate", top_k=3)
        mock_vs.search.assert_called_once_with("climate", top_k=3, filter_metadata=None)
        assert len(results) == 1

    def test_get_stats(self):
        store, _ = self._make()
        stats = store.get_stats()
        assert "document_count" in stats

    def test_collection_property(self):
        store, mock_vs = self._make()
        assert store.collection is mock_vs.collection


# ---------------------------------------------------------------------------
# BaseVectorStore is abstract
# ---------------------------------------------------------------------------

class TestBaseVectorStoreAbstract:

    def test_cannot_instantiate(self):
        from rag_system.knowledge_base.base_store import BaseVectorStore
        with pytest.raises(TypeError):
            BaseVectorStore()  # type: ignore
