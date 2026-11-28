"""
Optional Langfuse observability integration.

Initialises a Langfuse client only when LANGFUSE_PUBLIC_KEY and
LANGFUSE_SECRET_KEY are both present and the `langfuse` package is installed.
Otherwise all tracing calls degrade cleanly to lightweight no-ops.
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional
from loguru import logger


class _NoOpSpan:
    """No-op span returned when tracing is disabled."""

    def end(self, output: Any = None, metadata: Optional[Dict] = None) -> None:
        return None


class _NoOpTrace:
    """No-op trace returned when tracing is disabled."""

    def span(
        self,
        name: str,
        input: Any = None,
        output: Any = None,
        metadata: Optional[Dict] = None,
    ) -> _NoOpSpan:
        return _NoOpSpan()

    def update(self, output: Any = None, metadata: Optional[Dict] = None) -> None:
        return None


class _LangfuseSpanWrapper:
    """Wraps a real Langfuse span so errors never disrupt the request path."""

    def __init__(self, span_obj: Any) -> None:
        self._span = span_obj

    def end(self, output: Any = None, metadata: Optional[Dict] = None) -> None:
        try:
            kwargs: Dict[str, Any] = {}
            if output is not None:
                kwargs["output"] = output
            if metadata is not None:
                kwargs["metadata"] = metadata
            self._span.end(**kwargs)
        except Exception as exc:
            logger.debug(f"Langfuse span.end failed: {exc}")


class _LangfuseTraceWrapper:
    """Wraps a real Langfuse trace with fail-safe helpers."""

    def __init__(self, trace_obj: Any) -> None:
        self._trace = trace_obj

    def span(
        self,
        name: str,
        input: Any = None,
        output: Any = None,
        metadata: Optional[Dict] = None,
    ) -> _LangfuseSpanWrapper | _NoOpSpan:
        try:
            kwargs: Dict[str, Any] = {"name": name}
            if input is not None:
                kwargs["input"] = input
            if output is not None:
                kwargs["output"] = output
            if metadata is not None:
                kwargs["metadata"] = metadata
            return _LangfuseSpanWrapper(self._trace.span(**kwargs))
        except Exception as exc:
            logger.debug(f"Langfuse trace.span failed: {exc}")
            return _NoOpSpan()

    def update(self, output: Any = None, metadata: Optional[Dict] = None) -> None:
        try:
            kwargs: Dict[str, Any] = {}
            if output is not None:
                kwargs["output"] = output
            if metadata is not None:
                kwargs["metadata"] = metadata
            if kwargs:
                self._trace.update(**kwargs)
        except Exception as exc:
            logger.debug(f"Langfuse trace.update failed: {exc}")


class Tracer:
    """
    Facade over Langfuse with automatic no-op fallback.

    Reads credentials from environment variables:
      LANGFUSE_PUBLIC_KEY
      LANGFUSE_SECRET_KEY
      LANGFUSE_HOST (optional, defaults to https://cloud.langfuse.com)
    """

    def __init__(
        self,
        enabled: bool = True,
        public_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        host: Optional[str] = None,
    ) -> None:
        self._client = None
        self.enabled = False

        pub = public_key if public_key is not None else os.getenv("LANGFUSE_PUBLIC_KEY", "")
        sec = secret_key if secret_key is not None else os.getenv("LANGFUSE_SECRET_KEY", "")
        lf_host = host or os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

        if not enabled or not pub or not sec:
            logger.debug("Langfuse tracing disabled (missing credentials or disabled in config)")
            return

        try:
            from langfuse import Langfuse  # type: ignore

            self._client = Langfuse(
                public_key=pub,
                secret_key=sec,
                host=lf_host,
            )
            self.enabled = True
            logger.info(f"Langfuse tracing enabled ({lf_host})")
        except ImportError:
            logger.debug("langfuse package not installed — tracing disabled")
        except Exception as exc:
            logger.warning(f"Langfuse initialization failed ({exc}) — tracing disabled")

    def start_trace(
        self,
        name: str,
        input: Any = None,
        metadata: Optional[Dict] = None,
    ) -> _LangfuseTraceWrapper | _NoOpTrace:
        if not self.enabled or self._client is None:
            return _NoOpTrace()
        try:
            kwargs: Dict[str, Any] = {"name": name}
            if input is not None:
                kwargs["input"] = input
            if metadata is not None:
                kwargs["metadata"] = metadata
            return _LangfuseTraceWrapper(self._client.trace(**kwargs))
        except Exception as exc:
            logger.debug(f"Langfuse start_trace failed: {exc}")
            return _NoOpTrace()

    def flush(self) -> None:
        if not self.enabled or self._client is None:
            return
        try:
            self._client.flush()
        except Exception as exc:
            logger.debug(f"Langfuse flush failed: {exc}")


def get_tracer(enabled: bool = True) -> Tracer:
    """Create a Tracer using environment variables."""
    return Tracer(enabled=enabled)
