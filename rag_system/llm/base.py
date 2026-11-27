"""
BaseLLM — the common interface every LLM backend must implement.

Design decisions:
- Synchronous `generate()` only: the manager already wraps calls in
  tenacity retry, so async is not needed at this layer.
- `generate()` returns a plain string — no streaming for now (Phase 6).
- Subclasses raise `LLMError` on unrecoverable failures so callers
  can handle them uniformly regardless of provider.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


class LLMError(Exception):
    """Raised when an LLM call fails after all retries."""


@dataclass
class LLMConfig:
    """Provider-agnostic configuration passed to every backend."""
    provider: str = "gemini"          # gemini | openai | ollama
    model_name: str = "gemini-2.5-flash"
    temperature: float = 0.3
    max_tokens: int = 2048
    api_key: Optional[str] = None
    base_url: Optional[str] = None    # used by Ollama / OpenAI-compatible endpoints
    extra: dict = field(default_factory=dict)


class BaseLLM(ABC):
    """Abstract base for all LLM providers."""

    def __init__(self, config: LLMConfig) -> None:
        self.config = config

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """
        Send *prompt* to the LLM and return the text response.

        Raises:
            LLMError: on any unrecoverable failure.
        """

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.config.model_name})"
