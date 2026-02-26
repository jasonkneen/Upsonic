"""Upsonic integrations with observability platforms."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from upsonic.integrations.tracing import TracingProvider as TracingProvider
    from upsonic.integrations.tracing import DefaultTracingProvider as DefaultTracingProvider
    from upsonic.integrations.langfuse import Langfuse as Langfuse


def __getattr__(name: str):
    if name == "TracingProvider":
        from upsonic.integrations.tracing import TracingProvider
        return TracingProvider
    if name == "DefaultTracingProvider":
        from upsonic.integrations.tracing import DefaultTracingProvider
        return DefaultTracingProvider
    if name == "Langfuse":
        from upsonic.integrations.langfuse import Langfuse
        return Langfuse
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["TracingProvider", "DefaultTracingProvider", "Langfuse"]
