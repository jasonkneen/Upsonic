"""Agent-level OpenTelemetry manager.

Centralizes all OTel span creation, attribute setting, and error recording
for the Agent (and later Team) execution flow.  ``InstrumentedModel`` remains
responsible **only** for model-level (``chat``) spans.

Usage inside ``Agent``::

    self._otel = AgentOTelManager(settings, tracing_provider)
    with self._otel.agent_run_span(run_id, ...) as span:
        ...
        self._otel.set_run_attributes(span, ...)
"""

from __future__ import annotations

import json as _json
import time as _time
from contextlib import nullcontext
from typing import Any, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from upsonic.models.instrumented import InstrumentationSettings
    from upsonic.integrations.tracing import TracingProvider

ATTR_RUN_ID: str = "upsonic.run_id"
ATTR_AGENT_NAME: str = "upsonic.agent.name"
ATTR_AGENT_MODEL: str = "upsonic.agent.model"
ATTR_TASK_DESCRIPTION: str = "upsonic.task.description"
ATTR_TOOL_CALL_COUNT: str = "upsonic.tool_call_count"
ATTR_TOTAL_COST: str = "upsonic.total_cost"
ATTR_EXECUTION_TIME: str = "upsonic.execution_time"
ATTR_MODEL_EXECUTION_TIME: str = "upsonic.model_execution_time"
ATTR_INPUT: str = "upsonic.input"
ATTR_OUTPUT: str = "upsonic.output"
ATTR_USER_ID: str = "upsonic.user_id"
ATTR_SESSION_ID: str = "upsonic.session_id"
ATTR_TOOL_NAME: str = "upsonic.tool.name"
ATTR_TOOL_CALL_ID: str = "upsonic.tool.call_id"
ATTR_TOOL_EXECUTION_TIME: str = "upsonic.tool.execution_time"
ATTR_TOOL_SUCCESS: str = "upsonic.tool.success"
ATTR_PIPELINE_TOTAL_STEPS: str = "upsonic.pipeline.total_steps"
ATTR_PIPELINE_STREAMING: str = "upsonic.pipeline.streaming"
ATTR_PIPELINE_DEBUG: str = "upsonic.pipeline.debug"
ATTR_STEP_NAME: str = "upsonic.step.name"
ATTR_STEP_DESCRIPTION: str = "upsonic.step.description"
ATTR_STEP_STATUS: str = "upsonic.step.status"
ATTR_STEP_EXECUTION_TIME: str = "upsonic.step.execution_time"

LF_TRACE_NAME: str = "langfuse.trace.name"
LF_TRACE_INPUT: str = "langfuse.trace.input"
LF_TRACE_OUTPUT: str = "langfuse.trace.output"
LF_OBS_INPUT: str = "langfuse.observation.input"
LF_OBS_OUTPUT: str = "langfuse.observation.output"
LF_USER_ID: str = "langfuse.user.id"
LF_SESSION_ID: str = "langfuse.session.id"
GENERIC_USER_ID: str = "user.id"
GENERIC_SESSION_ID: str = "session.id"


def _is_recording(span: Any) -> bool:
    if span is None:
        return False
    try:
        return span.is_recording()
    except Exception:
        return False


def _set_status_ok(span: Any) -> None:
    """Mark *span* as successfully completed (``StatusCode.OK``)."""
    if not _is_recording(span):
        return
    try:
        from opentelemetry.trace import StatusCode
        span.set_status(StatusCode.OK)
    except Exception:
        pass


class AgentOTelManager:
    """Manages all OTel spans and attributes for an Agent execution.

    Holds a reference to :class:`InstrumentationSettings` and exposes
    methods for every span type the agent pipeline emits.  When
    *settings* is ``None`` every method degrades to a safe no-op so
    callers never need ``if`` guards.
    """

    __slots__ = ("_settings", "_tracing_provider")

    def __init__(
        self,
        settings: Optional["InstrumentationSettings"],
        tracing_provider: Optional["TracingProvider"] = None,
    ) -> None:
        self._settings: Optional["InstrumentationSettings"] = settings
        self._tracing_provider: Optional["TracingProvider"] = tracing_provider

    @property
    def enabled(self) -> bool:
        return self._settings is not None

    @property
    def settings(self) -> Optional["InstrumentationSettings"]:
        return self._settings

    def flush(self) -> None:
        """Force-flush pending spans to the backend without shutting down.

        Call after each agent run to ensure span data appears promptly
        in dashboards (Langfuse, Jaeger, etc.) rather than waiting for
        the ``BatchSpanProcessor``'s scheduled export interval.
        """
        if self._tracing_provider is not None:
            self._tracing_provider.flush()

    def agent_run_span(
        self,
        run_id: str,
        *,
        name: str = "",
        model: str = "",
        task_description: str = "",
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> Any:
        """Create the root ``agent.run`` span, or a no-op context manager."""
        if self._settings is None:
            return nullcontext(None)

        from opentelemetry.trace import SpanKind

        include_content: bool = getattr(self._settings, "include_content", True)
        truncated_desc: str = str(task_description)[:200] if task_description else ""

        trace_name: str = (truncated_desc if include_content else "") or name or "agent.run"

        init_attrs: Dict[str, Any] = {
            ATTR_AGENT_NAME: name,
            ATTR_AGENT_MODEL: model,
            ATTR_RUN_ID: run_id,
            LF_TRACE_NAME: trace_name,
        }

        if include_content and truncated_desc:
            init_attrs[ATTR_TASK_DESCRIPTION] = truncated_desc

        if user_id:
            init_attrs[LF_USER_ID] = user_id
            init_attrs[GENERIC_USER_ID] = user_id
        if session_id:
            init_attrs[LF_SESSION_ID] = session_id
            init_attrs[GENERIC_SESSION_ID] = session_id

        return self._settings.tracer.start_as_current_span(
            "agent.run",
            kind=SpanKind.SERVER,
            attributes=init_attrs,
        )

    def tool_span(self, tool_name: str, tool_call_id: str) -> Any:
        """Create a ``tool.execute`` span (``INTERNAL``), or a no-op context manager."""
        if self._settings is None:
            return nullcontext(None)

        from opentelemetry.trace import SpanKind

        return self._settings.tracer.start_as_current_span(
            "tool.execute",
            kind=SpanKind.INTERNAL,
            attributes={
                ATTR_TOOL_NAME: tool_name,
                ATTR_TOOL_CALL_ID: tool_call_id,
            },
        )

    def pipeline_span(
        self,
        total_steps: int,
        is_streaming: bool,
        debug: bool,
    ) -> Any:
        """Create a ``pipeline.execute`` span (``INTERNAL``), or a no-op context manager."""
        if self._settings is None:
            return nullcontext(None)

        from opentelemetry.trace import SpanKind

        return self._settings.tracer.start_as_current_span(
            "pipeline.execute",
            kind=SpanKind.INTERNAL,
            attributes={
                ATTR_PIPELINE_TOTAL_STEPS: total_steps,
                ATTR_PIPELINE_STREAMING: is_streaming,
                ATTR_PIPELINE_DEBUG: debug,
            },
        )

    def step_span(self, step_name: str, step_description: str) -> Any:
        """Create a ``pipeline.step.<name>`` span (``INTERNAL``), or a no-op context manager."""
        if self._settings is None:
            return nullcontext(None)

        from opentelemetry.trace import SpanKind

        return self._settings.tracer.start_as_current_span(
            f"pipeline.step.{step_name}",
            kind=SpanKind.INTERNAL,
            attributes={
                ATTR_STEP_NAME: step_name,
                ATTR_STEP_DESCRIPTION: step_description,
            },
        )

    def set_run_attributes(
        self,
        span: Any,
        *,
        input_tokens: int = 0,
        output_tokens: int = 0,
        tool_call_count: int = 0,
        total_cost: Optional[float] = None,
        execution_time: Optional[float] = None,
        model_execution_time: Optional[float] = None,
        input_text: Optional[str] = None,
        output_text: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> None:
        """Set final result attributes on the root ``agent.run`` span.

        Token attributes use ``gen_ai.*`` (OTel GenAI semantic conventions)
        so that backends like Langfuse recognise them.  When
        ``use_aggregated_usage_attribute_names`` is set, the
        ``gen_ai.aggregated_usage.*`` prefix is used instead.

        Also sets Langfuse-compatible attributes (``langfuse.trace.*``,
        ``langfuse.observation.*``, ``user.id``, ``session.id``) so that
        Langfuse dashboards display Input, Output, User, and Session
        correctly.  Non-Langfuse backends simply ignore these extras.
        """
        if not _is_recording(span) or self._settings is None:
            return

        use_aggregated: bool = getattr(
            self._settings, "use_aggregated_usage_attribute_names", False
        )
        prefix: str = "gen_ai.aggregated_usage" if use_aggregated else "gen_ai.usage"

        attrs: Dict[str, Any] = {}

        if input_tokens:
            attrs[f"{prefix}.input_tokens"] = input_tokens
        if output_tokens:
            attrs[f"{prefix}.output_tokens"] = output_tokens
        if tool_call_count:
            attrs[ATTR_TOOL_CALL_COUNT] = tool_call_count
        if total_cost is not None and total_cost > 0:
            attrs[ATTR_TOTAL_COST] = total_cost
        if execution_time is not None and execution_time > 0:
            attrs[ATTR_EXECUTION_TIME] = round(execution_time, 3)
        if model_execution_time is not None and model_execution_time > 0:
            attrs[ATTR_MODEL_EXECUTION_TIME] = round(model_execution_time, 3)

        include_content: bool = getattr(self._settings, "include_content", True)
        if include_content:
            if input_text:
                truncated_input: str = str(input_text)[:5000]
                attrs[ATTR_INPUT] = truncated_input
                json_input: str = _json.dumps(truncated_input)
                attrs[LF_TRACE_INPUT] = json_input
                attrs[LF_OBS_INPUT] = json_input
            if output_text:
                truncated_output: str = str(output_text)[:5000]
                attrs[ATTR_OUTPUT] = truncated_output
                json_output: str = _json.dumps(truncated_output)
                attrs[LF_TRACE_OUTPUT] = json_output
                attrs[LF_OBS_OUTPUT] = json_output

        if user_id:
            attrs[ATTR_USER_ID] = user_id
            attrs[LF_USER_ID] = user_id
            attrs[GENERIC_USER_ID] = user_id
        if session_id:
            attrs[ATTR_SESSION_ID] = session_id
            attrs[LF_SESSION_ID] = session_id
            attrs[GENERIC_SESSION_ID] = session_id

        if attrs:
            span.set_attributes(attrs)

    def set_tool_result(
        self,
        span: Any,
        execution_time: float,
        success: bool,
        error: Optional[Exception] = None,
    ) -> None:
        """Set result attributes on a ``tool.execute`` span."""
        if not _is_recording(span):
            return

        span.set_attributes({
            ATTR_TOOL_EXECUTION_TIME: execution_time,
            ATTR_TOOL_SUCCESS: success,
        })
        if success:
            _set_status_ok(span)
        elif error is not None:
            self.record_error(span, error)

    def set_step_result(
        self,
        span: Any,
        status: str,
        execution_time: float,
        error_message: Optional[str] = None,
    ) -> None:
        """Set result attributes on a ``pipeline.step.*`` span."""
        if not _is_recording(span):
            return

        span.set_attributes({
            ATTR_STEP_STATUS: status,
            ATTR_STEP_EXECUTION_TIME: execution_time,
        })
        if error_message:
            try:
                from opentelemetry.trace import StatusCode
                span.set_status(StatusCode.ERROR, error_message)
            except Exception:
                pass
        else:
            _set_status_ok(span)

    @staticmethod
    def mark_success(span: Any) -> None:
        """Explicitly mark *span* as successfully completed (``StatusCode.OK``)."""
        _set_status_ok(span)

    @staticmethod
    def record_error(span: Any, exc: Exception) -> None:
        """Record an exception and ``ERROR`` status on any span."""
        if not _is_recording(span):
            return
        try:
            from opentelemetry.trace import StatusCode
            span.set_status(StatusCode.ERROR, str(exc))
            span.record_exception(exc)
        except Exception:
            pass

    def finalize_agent_run(
        self,
        span: Any,
        output: Any,
        *,
        agent_user_id: Optional[str] = None,
        agent_session_id: Optional[str] = None,
        total_cost: Optional[float] = None,
    ) -> None:
        """Extract data from an ``AgentRunOutput`` and set span attributes.

        This is the single call-site an Agent (or Team) needs at the end of
        a run to populate all OTel attributes on the root span.
        """
        if not _is_recording(span) or self._settings is None or output is None:
            return

        run_usage = getattr(output, "usage", None)
        input_tokens: int = (getattr(run_usage, "input_tokens", 0) or 0) if run_usage else 0
        output_tokens: int = (getattr(run_usage, "output_tokens", 0) or 0) if run_usage else 0

        created_at: int = getattr(output, "created_at", 0)
        execution_time: Optional[float] = None
        if created_at:
            execution_time = _time.time() - created_at

        model_execution_time: Optional[float] = None
        exec_stats = getattr(output, "execution_stats", None)
        if exec_stats is not None:
            step_timing: Dict[str, float] = getattr(exec_stats, "step_timing", {})
            model_execution_time = step_timing.get("model_execution")

        task_obj = getattr(output, "task", None)
        input_text: Optional[str] = str(getattr(task_obj, "description", "")) if task_obj else None

        result_output = getattr(output, "output", None)
        output_text: Optional[str] = str(result_output) if result_output is not None else None

        self.set_run_attributes(
            span,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            tool_call_count=getattr(output, "tool_call_count", 0) or 0,
            total_cost=total_cost,
            execution_time=execution_time,
            model_execution_time=model_execution_time,
            input_text=input_text,
            output_text=output_text,
            user_id=getattr(output, "user_id", None) or agent_user_id,
            session_id=getattr(output, "session_id", None) or agent_session_id,
        )

        _set_status_ok(span)
        self.flush()
