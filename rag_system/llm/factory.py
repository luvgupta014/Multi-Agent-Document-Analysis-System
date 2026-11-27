"""
LLM factory — builds the right backend from config.yaml settings.

Usage:
    from rag_system.llm.factory import create_llm
    llm = create_llm(cfg)          # cfg is the parsed config.yaml dict
    answer = llm.generate(prompt)
"""
from __future__ import annotations

import os
from typing import Dict, Any

from .base import BaseLLM, LLMConfig, LLMError


def create_llm(cfg: Dict[str, Any]) -> BaseLLM:
    """
    Instantiate an LLM from the application config dict.

    Reads:
        llm.provider      (gemini | openai | ollama)
        llm.model_name
        llm.temperature
        llm.max_tokens
        llm.base_url      (optional — Ollama / OpenAI-compatible)

    API keys come from environment variables, never from config files:
        GEMINI_API_KEY    for provider=gemini
        OPENAI_API_KEY    for provider=openai
    """
    llm_cfg = cfg.get("llm", {})

    # Fall back to legacy `gemini` section for backwards compatibility
    legacy = cfg.get("gemini", {})

    provider = llm_cfg.get("provider", "gemini").lower()
    model_name = llm_cfg.get("model_name") or legacy.get("model_name", "gemini-2.5-flash")
    temperature = float(llm_cfg.get("temperature", legacy.get("temperature", 0.3)))
    max_tokens = int(llm_cfg.get("max_tokens", legacy.get("max_output_tokens", 2048)))
    base_url = llm_cfg.get("base_url")

    if provider == "gemini":
        api_key = os.getenv("GEMINI_API_KEY")
        config = LLMConfig(
            provider=provider,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
        )
        from .gemini import GeminiLLM
        return GeminiLLM(config)

    elif provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        config = LLMConfig(
            provider=provider,
            model_name=model_name or "gpt-4o-mini",
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
            base_url=base_url,
        )
        from .openai_llm import OpenAILLM
        return OpenAILLM(config)

    elif provider == "ollama":
        config = LLMConfig(
            provider=provider,
            model_name=model_name or "llama3.2",
            temperature=temperature,
            max_tokens=max_tokens,
            base_url=base_url or "http://localhost:11434",
        )
        from .ollama import OllamaLLM
        return OllamaLLM(config)

    else:
        raise LLMError(
            f"Unknown LLM provider: '{provider}'. "
            "Valid options: gemini, openai, ollama"
        )
