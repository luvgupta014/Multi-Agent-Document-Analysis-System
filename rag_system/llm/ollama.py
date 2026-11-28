"""
Ollama backend — calls the local Ollama HTTP API.

Default base_url: http://localhost:11434
Install Ollama: https://ollama.com/download
Pull a model:   ollama pull llama3.2
"""
from __future__ import annotations

import httpx
from loguru import logger
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from .base import BaseLLM, LLMConfig, LLMError


class OllamaLLM(BaseLLM):
    """Ollama local inference server (llama3, mistral, phi3, …)."""

    _DEFAULT_URL = "http://localhost:11434"

    def __init__(self, config: LLMConfig) -> None:
        super().__init__(config)
        self._base_url = (config.base_url or self._DEFAULT_URL).rstrip("/")
        logger.info(f"OllamaLLM initialised: {config.model_name} @ {self._base_url}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.ConnectError)),
        reraise=True,
    )
    def _post_generate(self, url: str, payload: dict) -> dict:
        response = httpx.post(url, json=payload, timeout=120.0)
        response.raise_for_status()
        return response.json()

    def generate(self, prompt: str) -> str:
        url = f"{self._base_url}/api/generate"
        payload = {
            "model": self.config.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.config.temperature,
                "num_predict": self.config.max_tokens,
            },
        }
        try:
            data = self._post_generate(url, payload)
            return data.get("response", "")
        except httpx.HTTPStatusError as exc:
            raise LLMError(f"Ollama HTTP error {exc.response.status_code}: {exc}") from exc
        except httpx.RequestError as exc:
            raise LLMError(
                f"Ollama unreachable at {self._base_url}. Is Ollama running? ({exc})"
            ) from exc
        except Exception as exc:
            raise LLMError(f"Ollama generate failed: {exc}") from exc
