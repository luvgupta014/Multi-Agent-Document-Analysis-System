"""
Conversation memory — a simple rolling window of recent turns.

Keeps the last `max_turns` user/assistant exchanges and formats them
as a plain-text prefix that can be prepended to any LLM prompt.

Design: in-memory only (no persistence).  Each ManagerAgent instance
carries its own Memory object.  Phase 6 can persist to SQLite if needed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class Turn:
    role: str    # "user" or "assistant"
    content: str


class ConversationMemory:
    """Rolling window conversation memory."""

    def __init__(self, max_turns: int = 5) -> None:
        self.max_turns = max_turns
        self._turns: List[Turn] = []

    def add(self, role: str, content: str) -> None:
        self._turns.append(Turn(role=role, content=content))
        # Keep only the last max_turns pairs (each pair = user + assistant)
        max_items = self.max_turns * 2
        if max_items == 0:
            self._turns.clear()
        elif len(self._turns) > max_items:
            self._turns = self._turns[-max_items:]

    def format_for_prompt(self) -> str:
        """Return memory as a text block to prepend to prompts."""
        if not self._turns:
            return ""
        lines = ["--- Conversation history ---"]
        for turn in self._turns:
            lines.append(f"{turn.role.capitalize()}: {turn.content}")
        lines.append("--- End of history ---\n")
        return "\n".join(lines)

    def clear(self) -> None:
        self._turns.clear()

    def __len__(self) -> int:
        return len(self._turns)
