"""
Security utilities for RAG pipelines.

Provides indirect prompt-injection detection and filtering for retrieved
document chunks before they are placed into the LLM synthesis prompt.
"""
from __future__ import annotations

import re
from typing import Dict, List, Pattern
from loguru import logger


# Patterns that signal an embedded instruction-override attempt inside a chunk.
# Kept focused on imperative override phrases so normal domain text is unaffected.
_INJECTION_PATTERNS: List[Pattern[str]] = [
    re.compile(
        r"ignore\s+(?:all\s+|any\s+)?(?:previous|prior|above|earlier)\s+"
        r"(?:instructions?|prompts?|rules?|directions?)",
        re.IGNORECASE,
    ),
    re.compile(
        r"disregard\s+(?:all\s+|any\s+)?(?:previous|prior|above|earlier)\s+"
        r"(?:instructions?|prompts?|rules?)",
        re.IGNORECASE,
    ),
    re.compile(
        r"forget\s+(?:all\s+|your\s+)?(?:previous\s+)?(?:instructions?|prompts?|rules?)",
        re.IGNORECASE,
    ),
    re.compile(
        r"do\s+not\s+follow\s+(?:the\s+)?(?:previous|above|system)\s+"
        r"(?:instructions?|rules?)",
        re.IGNORECASE,
    ),
    re.compile(
        r"override\s+(?:the\s+)?(?:system|safety|security)\s+(?:prompt|instructions?|rules?)",
        re.IGNORECASE,
    ),
    re.compile(
        r"reveal\s+(?:your\s+|the\s+)?system\s+prompt",
        re.IGNORECASE,
    ),
    re.compile(
        r"new\s+system\s+prompt\s*:",
        re.IGNORECASE,
    ),
    re.compile(
        r"<\s*\|\s*(?:im_start|im_end|system)\s*\|\s*>",
        re.IGNORECASE,
    ),
    re.compile(
        r"^\s*(?:system|assistant)\s*:\s*(?:ignore|override|you\s+must|do\s+not)\b",
        re.IGNORECASE | re.MULTILINE,
    ),
]


def is_prompt_injection(text: str) -> bool:
    """Return True if *text* matches any known prompt-injection pattern."""
    if not text:
        return False
    return any(pattern.search(text) for pattern in _INJECTION_PATTERNS)


def redact_injection_lines(text: str) -> str:
    """Replace any line matching a prompt-injection pattern with a placeholder."""
    if not text:
        return text
    cleaned_lines = []
    for line in text.splitlines():
        if is_prompt_injection(line):
            cleaned_lines.append("[REDACTED: potential prompt injection]")
        else:
            cleaned_lines.append(line)
    return "\n".join(cleaned_lines)


def filter_retrieved_chunks(
    chunks: List[Dict],
    enabled: bool = True,
    mode: str = "drop",
) -> List[Dict]:
    """
    Filter or redact retrieved chunks that contain prompt-injection attempts.

    Args:
        chunks: List of retrieved result dicts (expected key: 'content' or 'snippet').
        enabled: When False, returns chunks unchanged.
        mode: 'drop' removes flagged chunks; 'redact' strips flagged lines in-place.

    Returns:
        Filtered list of chunk dicts.
    """
    if not enabled or not chunks:
        return chunks

    safe_chunks: List[Dict] = []
    for chunk in chunks:
        text = chunk.get("content") or chunk.get("snippet") or ""
        if not is_prompt_injection(text):
            safe_chunks.append(chunk)
            continue

        chunk_id = chunk.get("id", "unknown")
        if mode == "redact":
            logger.warning(f"Redacting potential prompt injection in chunk '{chunk_id}'")
            updated = dict(chunk)
            if "content" in updated:
                updated["content"] = redact_injection_lines(updated["content"])
            elif "snippet" in updated:
                updated["snippet"] = redact_injection_lines(updated["snippet"])
            safe_chunks.append(updated)
        else:
            logger.warning(f"Dropped chunk '{chunk_id}' due to potential prompt injection")

    return safe_chunks
