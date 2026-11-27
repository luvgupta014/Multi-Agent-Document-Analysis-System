"""
OpenAI backend — uses the openai Python SDK (>=1.0).

Install: pip install openai
Not in requirements.txt by default to keep the base install lean;
the factory raises a clear error if the package is missing.
"""
from __future__ import annotations

from loguru import logger
from .base import BaseLLM, LLMConfig, LLMError


class OpenAILLM(BaseLLM):
    """OpenAI Chat Completions (gpt-4o, gpt-4o-mini, etc.)."""

    def __init__(self, config: LLMConfig) -> None:
        super().__init__(config)
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise LLMError(
                "openai package not installed. Run: pip install openai"
            ) from exc

        kwargs: dict = {"api_key": config.api_key}
        if config.base_url:
            kwargs["base_url"] = config.base_url  # supports any OpenAI-compatible API

        self._client = OpenAI(**kwargs)
        logger.info(f"OpenAILLM initialised: {config.model_name}")

    def generate(self, prompt: str) -> str:
        try:
            response = self._client.chat.completions.create(
                model=self.config.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
            return response.choices[0].message.content or ""
        except Exception as exc:
            raise LLMError(f"OpenAI generate failed: {exc}") from exc
