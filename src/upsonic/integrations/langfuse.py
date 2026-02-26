"""Langfuse integration for Upsonic OpenTelemetry instrumentation.

Inherits from :class:`TracingProvider` — only overrides exporter creation
to point at the Langfuse OTLP endpoint with Basic-Auth headers.

Usage::

    from upsonic import Agent, Langfuse

    langfuse = Langfuse(public_key="pk-lf-...", secret_key="sk-lf-...")
    agent = Agent("openai/gpt-4o", instrument=langfuse)
    agent.print_do("Hello!")
    langfuse.shutdown()
"""

from __future__ import annotations

import base64
import os
from typing import Literal, Optional, TYPE_CHECKING

from upsonic.integrations.tracing import TracingProvider

if TYPE_CHECKING:
    from opentelemetry.sdk.trace.export import SpanExporter as _SpanExporter

_LANGFUSE_EU_HOST: str = "https://cloud.langfuse.com"
_LANGFUSE_US_HOST: str = "https://us.cloud.langfuse.com"


class Langfuse(TracingProvider):
    """Langfuse observability integration for Upsonic agents.

    Sends OpenTelemetry traces to the Langfuse ``/api/public/otel`` endpoint
    using HTTP/protobuf with Basic-Auth.

    Args:
        public_key: Langfuse public key (``pk-lf-...``).
            Falls back to ``LANGFUSE_PUBLIC_KEY`` env var.
        secret_key: Langfuse secret key (``sk-lf-...``).
            Falls back to ``LANGFUSE_SECRET_KEY`` env var.
        host: Langfuse host URL.
            Falls back to ``LANGFUSE_HOST`` env var, then defaults to EU cloud.
        region: Shortcut for host selection: ``"eu"`` or ``"us"``.
            Ignored if ``host`` is explicitly provided.
        include_content: Whether to include prompt/response content in traces.
        service_name: Service name reported in traces (default ``"upsonic"``).
        sample_rate: Fraction of traces to sample (default ``1.0``).
        flush_on_exit: Register ``atexit`` handler (default ``True``).
        use_aggregated_usage_attribute_names: Use ``gen_ai.aggregated_usage.*``
            prefix on root spans.

    Raises:
        ValueError: If public_key or secret_key cannot be resolved.
        ImportError: If required OpenTelemetry packages are not installed.

    Example::

        # Minimal — keys from env vars
        langfuse = Langfuse()
        agent = Agent("openai/gpt-4o", instrument=langfuse)

        # Explicit keys, US region, no content in traces
        langfuse = Langfuse(
            public_key="pk-lf-abc",
            secret_key="sk-lf-xyz",
            region="us",
            include_content=False,
        )
        agent = Agent("openai/gpt-4o", instrument=langfuse)
        agent.do("What is 2+2?")
        langfuse.shutdown()
    """

    def __init__(
        self,
        *,
        public_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        host: Optional[str] = None,
        region: Literal["eu", "us"] = "eu",
        include_content: bool = True,
        service_name: str = "upsonic",
        sample_rate: float = 1.0,
        flush_on_exit: bool = True,
        use_aggregated_usage_attribute_names: bool = False,
    ) -> None:
        self._public_key: str = public_key or os.getenv("LANGFUSE_PUBLIC_KEY", "")
        self._secret_key: str = secret_key or os.getenv("LANGFUSE_SECRET_KEY", "")

        if not self._public_key or not self._secret_key:
            raise ValueError(
                "Langfuse public_key and secret_key are required. "
                "Pass them as arguments or set LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY env vars."
            )

        if host is not None:
            self._host: str = host.rstrip("/")
        else:
            env_host: str = os.getenv("LANGFUSE_HOST", "")
            if env_host:
                self._host = env_host.rstrip("/")
            else:
                self._host = _LANGFUSE_US_HOST if region == "us" else _LANGFUSE_EU_HOST

        self._endpoint: str = f"{self._host}/api/public/otel/v1/traces"
        self._auth_header: str = self._build_auth_header(self._public_key, self._secret_key)

        super().__init__(
            service_name=service_name,
            sample_rate=sample_rate,
            include_content=include_content,
            use_aggregated_usage_attribute_names=use_aggregated_usage_attribute_names,
            flush_on_exit=flush_on_exit,
        )

    def _create_exporter(self) -> "_SpanExporter":
        """HTTP/protobuf OTLP exporter aimed at the Langfuse endpoint."""
        try:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                OTLPSpanExporter,
            )
        except ImportError as exc:
            raise ImportError(
                "HTTP OTLP exporter is required for Langfuse (gRPC is not supported). "
                "Install with: pip install opentelemetry-exporter-otlp-proto-http"
            ) from exc

        return OTLPSpanExporter(
            endpoint=self._endpoint,
            headers={"Authorization": self._auth_header},
        )

    @staticmethod
    def _build_auth_header(public_key: str, secret_key: str) -> str:
        raw: str = f"{public_key}:{secret_key}"
        encoded: str = base64.b64encode(raw.encode("utf-8")).decode("ascii")
        return f"Basic {encoded}"

    def __repr__(self) -> str:
        return (
            f"Langfuse(host={self._host!r}, "
            f"service_name={self._service_name!r}, "
            f"include_content={self._include_content}, "
            f"sample_rate={self._sample_rate})"
        )
