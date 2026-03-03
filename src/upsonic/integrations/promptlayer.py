"""PromptLayer integration for Upsonic prompt management and observability.

Provides prompt registry, versioning, request tracking, scoring, and metadata
through PromptLayer's REST API.  When passed to ``Agent(promptlayer=...)``,
every agent execution is automatically logged to PromptLayer with full model
parameters, token counts, and cost.

Usage::

    from upsonic import Agent
    from upsonic.integrations.promptlayer import PromptLayer

    pl = PromptLayer(api_key="pl_...")

    agent = Agent(
        "openai/gpt-4o",
        system_prompt=pl.get_prompt("my-agent-v2"),
        promptlayer=pl,
    )
    result = agent.do("What is the capital of Japan?")

    pl.shutdown()
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union, TYPE_CHECKING

if TYPE_CHECKING:
    import httpx as _httpx


class PromptLayer:
    """PromptLayer integration for prompt management and observability.

    Connects to PromptLayer's REST API for prompt registry (versioned prompt
    templates), request tracking, scoring, and metadata tagging.

    Args:
        api_key: PromptLayer API key (``pl_...``).
            Falls back to ``PROMPTLAYER_API_KEY`` env var.
        base_url: PromptLayer API base URL.
            Falls back to ``PROMPTLAYER_BASE_URL`` env var,
            then defaults to ``https://api.promptlayer.com``.

    Raises:
        ValueError: If api_key cannot be resolved.
    """

    _DEFAULT_BASE_URL: str = "https://api.promptlayer.com"

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> None:
        self._api_key: str = api_key or os.getenv("PROMPTLAYER_API_KEY", "")
        if not self._api_key:
            raise ValueError(
                "PromptLayer api_key is required. "
                "Pass it as an argument or set PROMPTLAYER_API_KEY env var."
            )
        resolved_url: str = base_url or os.getenv(
            "PROMPTLAYER_BASE_URL", self._DEFAULT_BASE_URL
        )
        self._base_url: str = resolved_url.rstrip("/")
        self._client: Optional[_httpx.Client] = None
        self._async_client: Optional[_httpx.AsyncClient] = None
        self._last_prompt_id: Optional[int] = None
        self._last_prompt_version: Optional[int] = None

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _stringify_metadata(metadata: Dict[str, Any]) -> Dict[str, str]:
        import json as _json

        out: Dict[str, str] = {}
        for k, v in metadata.items():
            if isinstance(v, str):
                out[str(k)] = v
            elif isinstance(v, (dict, list, tuple)):
                out[str(k)] = _json.dumps(v, default=str)
            else:
                out[str(k)] = str(v)
        return out

    @staticmethod
    def _parse_provider_model(name: str) -> Tuple[str, str]:
        """Extract ``(provider, model)`` from ``provider/model`` format.

        Handles ``"openai/gpt-4o"``, ``"accuracy_eval:anthropic/claude-sonnet-4-6"``,
        and plain names like ``"reliability_eval"``.
        """
        cleaned: str = name
        if ":" in cleaned:
            cleaned = cleaned.split(":", 1)[1].strip()
        if "/" in cleaned:
            parts: List[str] = cleaned.split("/", 1)
            return parts[0], parts[1]
        return "custom", name

    @staticmethod
    def _epoch_to_iso(epoch: float) -> str:
        from datetime import datetime, timezone
        return datetime.fromtimestamp(epoch, tz=timezone.utc).isoformat()

    # ------------------------------------------------------------------
    # HTTP plumbing
    # ------------------------------------------------------------------

    def _get_client(self) -> "_httpx.Client":
        if self._client is None:
            import httpx
            self._client = httpx.Client(base_url=self._base_url, timeout=30.0)
        return self._client

    def _get_async_client(self) -> "_httpx.AsyncClient":
        if self._async_client is None:
            import httpx
            self._async_client = httpx.AsyncClient(base_url=self._base_url, timeout=30.0)
        return self._async_client

    def _post(self, path: str, body: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        headers: Dict[str, str] = {"X-API-KEY": self._api_key}
        response = self._get_client().post(path, json=body, headers=headers)
        response.raise_for_status()
        data: Dict[str, Any] = response.json()
        return data

    async def _apost(self, path: str, body: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        headers: Dict[str, str] = {"X-API-KEY": self._api_key}
        response = await self._get_async_client().post(path, json=body, headers=headers)
        response.raise_for_status()
        data: Dict[str, Any] = response.json()
        return data

    def _log_post(self, body: Dict[str, Any]) -> Dict[str, Any]:
        headers: Dict[str, str] = {"X-API-KEY": self._api_key}
        response = self._get_client().post("/log-request", json=body, headers=headers)
        response.raise_for_status()
        data: Dict[str, Any] = response.json()
        return data

    async def _alog_post(self, body: Dict[str, Any]) -> Dict[str, Any]:
        headers: Dict[str, str] = {"X-API-KEY": self._api_key}
        response = await self._get_async_client().post("/log-request", json=body, headers=headers)
        response.raise_for_status()
        data: Dict[str, Any] = response.json()
        return data

    # ------------------------------------------------------------------
    # Prompt registry
    # ------------------------------------------------------------------

    def get_prompt(
        self,
        prompt_name: str,
        *,
        version: Optional[int] = None,
        label: Optional[str] = None,
        variables: Optional[Dict[str, str]] = None,
        return_metadata: bool = False,
    ) -> Union[str, Tuple[str, Dict[str, Any]]]:
        """Fetch a prompt template from PromptLayer by name.

        Args:
            prompt_name: Name of the prompt in PromptLayer registry.
            version: Specific version number. ``None`` for latest.
            label: Label to fetch (e.g. ``"production"``, ``"staging"``).
            variables: Template variables to fill in the prompt template.
            return_metadata: If ``True``, returns ``(prompt_text, metadata)`` tuple.

        Returns:
            The rendered prompt string, or ``(prompt_string, metadata)``
            when *return_metadata* is ``True``.
        """
        body: Dict[str, Any] = {}
        if version is not None:
            body["version"] = version
        if label is not None:
            body["label"] = label
        if variables is not None:
            body["input_variables"] = variables

        result: Dict[str, Any] = self._post(
            f"/prompt-templates/{prompt_name}", body or None
        )
        prompt_text: str = self._extract_prompt_text(result)

        self._last_prompt_id = result.get("id")
        self._last_prompt_version = result.get("version")

        if return_metadata:
            metadata: Dict[str, Any] = {
                "id": result.get("id"),
                "version": result.get("version"),
                "label": result.get("label"),
            }
            return prompt_text, metadata
        return prompt_text

    async def aget_prompt(
        self,
        prompt_name: str,
        *,
        version: Optional[int] = None,
        label: Optional[str] = None,
        variables: Optional[Dict[str, str]] = None,
        return_metadata: bool = False,
    ) -> Union[str, Tuple[str, Dict[str, Any]]]:
        """Async variant of :meth:`get_prompt`."""
        body: Dict[str, Any] = {}
        if version is not None:
            body["version"] = version
        if label is not None:
            body["label"] = label
        if variables is not None:
            body["input_variables"] = variables

        result: Dict[str, Any] = await self._apost(
            f"/prompt-templates/{prompt_name}", body or None
        )
        prompt_text: str = self._extract_prompt_text(result)

        self._last_prompt_id = result.get("id")
        self._last_prompt_version = result.get("version")

        if return_metadata:
            metadata: Dict[str, Any] = {
                "id": result.get("id"),
                "version": result.get("version"),
                "label": result.get("label"),
            }
            return prompt_text, metadata
        return prompt_text

    @staticmethod
    def _extract_prompt_text(result: Dict[str, Any]) -> str:
        prompt_template: Dict[str, Any] = result.get("prompt_template", {})

        messages: Optional[List[Dict[str, Any]]] = prompt_template.get("messages")
        if messages and isinstance(messages, list):
            parts: List[str] = []
            for msg in messages:
                content: Any = msg.get("content", "")
                if isinstance(content, list):
                    text_parts: List[str] = [
                        p.get("text", "")
                        for p in content
                        if isinstance(p, dict) and p.get("type") == "text"
                    ]
                    content = "\n".join(text_parts)
                parts.append(str(content))
            return "\n\n".join(parts)

        content_list: Optional[List[Dict[str, Any]]] = prompt_template.get("content")
        if content_list and isinstance(content_list, list):
            text_parts: List[str] = [
                p.get("text", "")
                for p in content_list
                if isinstance(p, dict) and p.get("type") == "text"
            ]
            if text_parts:
                return "\n".join(text_parts)

        template: Optional[str] = prompt_template.get("template")
        if template:
            return str(template)

        return str(prompt_template)

    # ------------------------------------------------------------------
    # Unified log / alog
    # ------------------------------------------------------------------

    def _build_log_body(
        self,
        *,
        provider: str,
        model: str,
        input_text: str,
        output_text: str,
        start_time: Optional[float],
        end_time: Optional[float],
        input_tokens: int,
        output_tokens: int,
        price: float,
        parameters: Optional[Dict[str, Any]],
        tags: Optional[List[str]],
        metadata: Optional[Dict[str, Any]],
        score: Optional[int],
        status: str,
        function_name: Optional[str],
        prompt_name: Optional[str],
        prompt_id: Optional[int],
        prompt_version: Optional[int],
        system_prompt: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_calls: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        now: float = time.time()

        input_messages: List[Dict[str, Any]] = []
        if system_prompt:
            input_messages.append(
                {"role": "system", "content": [{"type": "text", "text": system_prompt}]}
            )
        input_messages.append(
            {"role": "user", "content": [{"type": "text", "text": input_text}]}
        )

        input_body: Dict[str, Any] = {"type": "chat", "messages": input_messages}
        if tools:
            input_body["tools"] = tools

        assistant_message: Dict[str, Any] = {
            "role": "assistant",
            "content": [{"type": "text", "text": output_text}],
        }
        if tool_calls:
            assistant_message["tool_calls"] = tool_calls

        body: Dict[str, Any] = {
            "provider": provider,
            "model": model,
            "input": input_body,
            "output": {"type": "chat", "messages": [assistant_message]},
            "request_start_time": self._epoch_to_iso(start_time or now),
            "request_end_time": self._epoch_to_iso(end_time or now),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "price": price,
            "status": status,
        }

        if function_name is not None:
            body["function_name"] = function_name
        if parameters:
            body["parameters"] = parameters
        if tags:
            body["tags"] = tags
        if metadata:
            body["metadata"] = self._stringify_metadata(metadata)
        if score is not None:
            body["score"] = max(0, min(100, int(round(score))))
        if prompt_name is not None:
            body["prompt_name"] = prompt_name
        if prompt_id is not None:
            body["prompt_id"] = prompt_id
        if prompt_version is not None:
            body["prompt_version_number"] = prompt_version

        return body

    def log(
        self,
        *,
        provider: str,
        model: str,
        input_text: str,
        output_text: str,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        input_tokens: int = 0,
        output_tokens: int = 0,
        price: float = 0.0,
        parameters: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        score: Optional[int] = None,
        status: str = "SUCCESS",
        function_name: Optional[str] = None,
        prompt_name: Optional[str] = None,
        prompt_id: Optional[int] = None,
        prompt_version: Optional[int] = None,
        scores: Optional[Dict[str, int]] = None,
        system_prompt: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_calls: Optional[List[Dict[str, Any]]] = None,
    ) -> int:
        """Log a request to PromptLayer via ``/log-request``.

        This is the single entry point for all PromptLayer logging -- agent
        runs, accuracy evals, reliability evals, performance evals.  Callers
        construct the appropriate arguments for their use case.

        Args:
            provider: LLM provider (e.g. ``"openai"``, ``"anthropic"``).
            model: Model name (e.g. ``"gpt-4o"``, ``"claude-sonnet-4-6"``).
            input_text: The input prompt or query.
            output_text: The model's output or response.
            start_time: Request start time (epoch seconds).
            end_time: Request end time (epoch seconds).
            input_tokens: Number of input/prompt tokens used.
            output_tokens: Number of output/completion tokens used.
            price: Cost of the request in USD.
            parameters: Model parameters (temperature, max_tokens, etc.).
            tags: Tags for organizing requests.
            metadata: Metadata dictionary (values are stringified).
            score: Primary score (0--100, clamped).
            status: Request status (``SUCCESS``, ``WARNING``, ``ERROR``).
            function_name: Function name for dashboard display.
            prompt_name: PromptLayer prompt template name.
            prompt_id: PromptLayer prompt template ID.
            prompt_version: Prompt template version number.
            scores: Named scores dict (``{name: value}``) to attach via
                ``/rest/track-score`` after the initial log.
            system_prompt: System prompt text (added as a system message).
            tools: Tool definitions available to the model.
            tool_calls: Tool calls made by the model during the run.

        Returns:
            The PromptLayer ``request_id``.
        """
        body: Dict[str, Any] = self._build_log_body(
            provider=provider,
            model=model,
            input_text=input_text,
            output_text=output_text,
            start_time=start_time,
            end_time=end_time,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            price=price,
            parameters=parameters,
            tags=tags,
            metadata=metadata,
            score=score,
            status=status,
            function_name=function_name,
            prompt_name=prompt_name,
            prompt_id=prompt_id,
            prompt_version=prompt_version,
            system_prompt=system_prompt,
            tools=tools,
            tool_calls=tool_calls,
        )
        result: Dict[str, Any] = self._log_post(body)
        request_id: int = result.get("id", 0)

        if scores:
            for score_name, score_value in scores.items():
                self.score(request_id, score_value, name=score_name)

        return request_id

    async def alog(
        self,
        *,
        provider: str,
        model: str,
        input_text: str,
        output_text: str,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        input_tokens: int = 0,
        output_tokens: int = 0,
        price: float = 0.0,
        parameters: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        score: Optional[int] = None,
        status: str = "SUCCESS",
        function_name: Optional[str] = None,
        prompt_name: Optional[str] = None,
        prompt_id: Optional[int] = None,
        prompt_version: Optional[int] = None,
        scores: Optional[Dict[str, int]] = None,
        system_prompt: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_calls: Optional[List[Dict[str, Any]]] = None,
    ) -> int:
        """Async variant of :meth:`log`."""
        body: Dict[str, Any] = self._build_log_body(
            provider=provider,
            model=model,
            input_text=input_text,
            output_text=output_text,
            start_time=start_time,
            end_time=end_time,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            price=price,
            parameters=parameters,
            tags=tags,
            metadata=metadata,
            score=score,
            status=status,
            function_name=function_name,
            prompt_name=prompt_name,
            prompt_id=prompt_id,
            prompt_version=prompt_version,
            system_prompt=system_prompt,
            tools=tools,
            tool_calls=tool_calls,
        )
        result: Dict[str, Any] = await self._alog_post(body)
        request_id: int = result.get("id", 0)

        if scores:
            for score_name, score_value in scores.items():
                await self.ascore(request_id, score_value, name=score_name)

        return request_id

    # ------------------------------------------------------------------
    # Post-hoc score / metadata
    # ------------------------------------------------------------------

    def score(
        self,
        request_id: int,
        score: Union[int, float],
        *,
        name: str = "quality",
    ) -> bool:
        """Score a previously logged request.

        PromptLayer requires integer scores in the range 0--100.
        Float values are rounded automatically and clamped to [0, 100].

        Args:
            request_id: The PromptLayer request ID to score.
            score: Numerical score value (rounded and clamped to 0--100).
            name: Name of the score metric (e.g. ``"accuracy"``, ``"quality"``).

        Returns:
            ``True`` if scoring was successful.
        """
        clamped: int = max(0, min(100, int(round(score))))
        body: Dict[str, Any] = {
            "request_id": request_id,
            "score": clamped,
            "score_name": name,
        }
        result: Dict[str, Any] = self._post("/rest/track-score", body)
        return bool(result.get("success", False))

    async def ascore(
        self,
        request_id: int,
        score: Union[int, float],
        *,
        name: str = "quality",
    ) -> bool:
        """Async variant of :meth:`score`."""
        clamped: int = max(0, min(100, int(round(score))))
        body: Dict[str, Any] = {
            "request_id": request_id,
            "score": clamped,
            "score_name": name,
        }
        result: Dict[str, Any] = await self._apost("/rest/track-score", body)
        return bool(result.get("success", False))

    def add_metadata(
        self,
        request_id: int,
        metadata: Dict[str, Any],
    ) -> bool:
        """Add metadata to a previously logged request.

        Args:
            request_id: The PromptLayer request ID.
            metadata: Metadata dictionary to attach (values are stringified).

        Returns:
            ``True`` if metadata was added successfully.
        """
        body: Dict[str, Any] = {
            "request_id": request_id,
            "metadata": self._stringify_metadata(metadata),
        }
        result: Dict[str, Any] = self._post("/rest/track-metadata", body)
        return bool(result.get("success", False))

    async def aadd_metadata(
        self,
        request_id: int,
        metadata: Dict[str, Any],
    ) -> bool:
        """Async variant of :meth:`add_metadata`."""
        body: Dict[str, Any] = {
            "request_id": request_id,
            "metadata": self._stringify_metadata(metadata),
        }
        result: Dict[str, Any] = await self._apost("/rest/track-metadata", body)
        return bool(result.get("success", False))

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def shutdown(self) -> None:
        """Close HTTP clients. Safe to call multiple times."""
        if self._client is not None:
            self._client.close()
            self._client = None

    async def ashutdown(self) -> None:
        """Async variant of :meth:`shutdown`."""
        if self._async_client is not None:
            await self._async_client.aclose()
            self._async_client = None

    def __repr__(self) -> str:
        return f"PromptLayer(base_url={self._base_url!r})"
