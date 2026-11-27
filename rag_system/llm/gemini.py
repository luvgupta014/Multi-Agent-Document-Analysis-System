"""
Gemini backend — wraps google-generativeai with tenacity retry.

The google.generativeai package is deprecated in favour of google.genai;
migration happens in Phase 3 (this file). We keep the existing dependency
to avoid breaking the installed venv, but the abstraction means we can
swap it out without touching any other code.
"""
from __future__ import annotations

import google.generativeai as genai
import google.api_core.exceptions
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from loguru import logger

from .base import BaseLLM, LLMConfig, LLMError


class GeminiLLM(BaseLLM):
    """Google Gemini via the google-generativeai SDK."""

    def __init__(self, config: LLMConfig) -> None:
        super().__init__(config)
        if not config.api_key:
            raise LLMError("GEMINI_API_KEY is required for the Gemini provider.")
        genai.configure(api_key=config.api_key)
        self._model = genai.GenerativeModel(config.model_name)
        logger.info(f"GeminiLLM initialised: {config.model_name}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(google.api_core.exceptions.ResourceExhausted),
        reraise=True,
    )
    def generate(self, prompt: str) -> str:
        try:
            response = self._model.generate_content(prompt)
            return response.text
        except google.api_core.exceptions.ResourceExhausted:
            raise  # let tenacity handle the retry
        except Exception as exc:
            raise LLMError(f"Gemini generate failed: {exc}") from exc
