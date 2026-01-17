from __future__ import annotations as _annotations

import dataclasses
from copy import copy
from dataclasses import dataclass, fields
from typing import Annotated, Any, Dict, Optional, TYPE_CHECKING

from genai_prices.data_snapshot import get_snapshot
from pydantic import AliasChoices, BeforeValidator, Field
from typing_extensions import deprecated, overload

from upsonic import _utils
from upsonic.utils.package.exception import UsageLimitExceeded

if TYPE_CHECKING:
    from upsonic.utils.timer import Timer

__all__ = 'RequestUsage', 'RunUsage', 'Usage', 'UsageLimits'


@dataclass(repr=False, kw_only=True)
class UsageBase:
    input_tokens: Annotated[
        int,
        # `request_tokens` is deprecated, but we still want to support deserializing model responses stored in a DB before the name was changed
        Field(validation_alias=AliasChoices('input_tokens', 'request_tokens')),
    ] = 0
    """Number of input/prompt tokens."""

    cache_write_tokens: int = 0
    """Number of tokens written to the cache."""
    cache_read_tokens: int = 0
    """Number of tokens read from the cache."""

    output_tokens: Annotated[
        int,
        # `response_tokens` is deprecated, but we still want to support deserializing model responses stored in a DB before the name was changed
        Field(validation_alias=AliasChoices('output_tokens', 'response_tokens')),
    ] = 0
    """Number of output/completion tokens."""

    input_audio_tokens: int = 0
    """Number of audio input tokens."""
    cache_audio_read_tokens: int = 0
    """Number of audio tokens read from the cache."""
    output_audio_tokens: int = 0
    """Number of audio output tokens."""

    details: Annotated[
        dict[str, int],
        # `details` can not be `None` any longer, but we still want to support deserializing model responses stored in a DB before this was changed
        BeforeValidator(lambda d: d or {}),
    ] = dataclasses.field(default_factory=dict)
    """Any extra details returned by the model."""

    @property
    @deprecated('`request_tokens` is deprecated, use `input_tokens` instead')
    def request_tokens(self) -> int:
        return self.input_tokens

    @property
    @deprecated('`response_tokens` is deprecated, use `output_tokens` instead')
    def response_tokens(self) -> int:
        return self.output_tokens

    @property
    def total_tokens(self) -> int:
        """Sum of `input_tokens + output_tokens`."""
        return self.input_tokens + self.output_tokens

    def opentelemetry_attributes(self) -> dict[str, int]:
        """Get the token usage values as OpenTelemetry attributes."""
        result: dict[str, int] = {}
        if self.input_tokens:
            result['gen_ai.usage.input_tokens'] = self.input_tokens
        if self.output_tokens:
            result['gen_ai.usage.output_tokens'] = self.output_tokens

        details = self.details.copy()
        if self.cache_write_tokens:
            details['cache_write_tokens'] = self.cache_write_tokens
        if self.cache_read_tokens:
            details['cache_read_tokens'] = self.cache_read_tokens
        if self.input_audio_tokens:
            details['input_audio_tokens'] = self.input_audio_tokens
        if self.cache_audio_read_tokens:
            details['cache_audio_read_tokens'] = self.cache_audio_read_tokens
        if self.output_audio_tokens:
            details['output_audio_tokens'] = self.output_audio_tokens
        if details:
            prefix = 'gen_ai.usage.details.'
            for key, value in details.items():
                # Skipping check for value since spec implies all detail values are relevant
                if value:
                    result[prefix + key] = value
        return result

    def __repr__(self):
        kv_pairs = (f'{f.name}={value!r}' for f in fields(self) if (value := getattr(self, f.name)))
        return f'{self.__class__.__qualname__}({", ".join(kv_pairs)})'

    def has_values(self) -> bool:
        """Whether any values are set and non-zero."""
        return any(dataclasses.asdict(self).values())


@dataclass(repr=False, kw_only=True)
class RequestUsage(UsageBase):
    """LLM usage associated with a single request.

    This is an implementation of `genai_prices.types.AbstractUsage` so it can be used to calculate the price of the
    request using genai-prices.
    """

    @property
    def requests(self):
        return 1

    def incr(self, incr_usage: RequestUsage) -> None:
        """Increment the usage in place.

        Args:
            incr_usage: The usage to increment by.
        """
        return _incr_usage_tokens(self, incr_usage)

    def __add__(self, other: RequestUsage) -> RequestUsage:
        """Add two RequestUsages together.

        This is provided so it's trivial to sum usage information from multiple parts of a response.

        **WARNING:** this CANNOT be used to sum multiple requests without breaking some pricing calculations.
        """
        new_usage = copy(self)
        new_usage.incr(other)
        return new_usage

    @classmethod
    def extract(
        cls,
        data: Any,
        *,
        provider: str,
        provider_url: str,
        provider_fallback: str,
        api_flavor: str = 'default',
        details: dict[str, Any] | None = None,
    ) -> RequestUsage:
        """Extract usage information from the response data using genai-prices.

        Args:
            data: The response data from the model API.
            provider: The actual provider ID
            provider_url: The provider base_url
            provider_fallback: The fallback provider ID to use if the actual provider is not found in genai-prices.
                For example, an OpenAI model should set this to "openai" in case it has an obscure provider ID.
            api_flavor: The API flavor to use when extracting usage information,
                e.g. 'chat' or 'responses' for OpenAI.
            details: Becomes the `details` field on the returned `RequestUsage` for convenience.
        """
        details = details or {}
        for provider_id, provider_api_url in [(None, provider_url), (provider, None), (provider_fallback, None)]:
            try:
                provider_obj = get_snapshot().find_provider(None, provider_id, provider_api_url)
                _model_ref, extracted_usage = provider_obj.extract_usage(data, api_flavor=api_flavor)
                return cls(**{k: v for k, v in extracted_usage.__dict__.items() if v is not None}, details=details)
            except Exception:
                pass
        return cls(details=details)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization.
        
        Excludes None/zero values for cleaner output.
        
        Returns:
            Dictionary representation of the usage metrics.
        """
        result: Dict[str, Any] = {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "input_audio_tokens": self.input_audio_tokens,
            "cache_audio_read_tokens": self.cache_audio_read_tokens,
            "output_audio_tokens": self.output_audio_tokens,
        }
        
        # Add optional fields only if they have values
        if self.details:
            result["details"] = self.details
        
        # Remove zero values for cleaner output
        result = {
            k: v for k, v in result.items()
            if v is not None and (not isinstance(v, (int, float)) or v != 0) and (not isinstance(v, dict) or len(v) > 0)
        }
        
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RequestUsage":
        """Reconstruct RequestUsage from dictionary.
        
        Args:
            data: Dictionary containing RequestUsage fields.
            
        Returns:
            RequestUsage instance.
        """
        return cls(
            input_tokens=data.get("input_tokens", 0),
            output_tokens=data.get("output_tokens", 0),
            cache_write_tokens=data.get("cache_write_tokens", 0),
            cache_read_tokens=data.get("cache_read_tokens", 0),
            input_audio_tokens=data.get("input_audio_tokens", 0),
            cache_audio_read_tokens=data.get("cache_audio_read_tokens", 0),
            output_audio_tokens=data.get("output_audio_tokens", 0),
            details=data.get("details", {}),
        )


@dataclass(repr=False, kw_only=True)
class RunUsage(UsageBase):
    """Agent usage associated with an agent run.

    Responsibility for calculating request usage is on the model; Upsonic simply sums the usage information across requests.
    """

    requests: int = 0
    """Number of requests made to the LLM API."""

    tool_calls: int = 0
    """Number of successful tool calls executed during the run."""

    input_tokens: int = 0
    """Total number of input/prompt tokens."""

    cache_write_tokens: int = 0
    """Total number of tokens written to the cache."""

    cache_read_tokens: int = 0
    """Total number of tokens read from the cache."""

    input_audio_tokens: int = 0
    """Total number of audio input tokens."""

    cache_audio_read_tokens: int = 0
    """Total number of audio tokens read from the cache."""

    output_tokens: int = 0
    """Total number of output/completion tokens."""

    output_audio_tokens: int = 0
    """Total number of audio output tokens."""

    reasoning_tokens: int = 0
    """Number of tokens employed in reasoning."""

    details: dict[str, int] = dataclasses.field(default_factory=dict)
    """Any extra details returned by the model."""

    # Timer and timing metrics
    timer: Optional["Timer"] = None
    """Internal timer utility for tracking execution time."""

    time_to_first_token: Optional[float] = None
    """Time from run start to first token generation, in seconds."""

    duration: Optional[float] = None
    """Total run time, in seconds."""

    # Cost metrics
    cost: Optional[float] = None
    """Estimated cost of the run (provider-specific)."""

    # Provider-specific and additional metrics
    provider_metrics: Optional[Dict[str, Any]] = None
    """Provider-specific metrics (e.g., latency breakdown, model info)."""

    additional_metrics: Optional[Dict[str, Any]] = None
    """Any additional custom metrics."""

    def incr(self, incr_usage: RunUsage | RequestUsage) -> None:
        """Increment the usage in place.

        Args:
            incr_usage: The usage to increment by.
        """
        if isinstance(incr_usage, RunUsage):
            self.requests += incr_usage.requests
            self.tool_calls += incr_usage.tool_calls
            self.reasoning_tokens += incr_usage.reasoning_tokens
            
            # Handle cost
            if incr_usage.cost is not None:
                if self.cost is None:
                    self.cost = incr_usage.cost
                else:
                    self.cost += incr_usage.cost
            
            # Handle duration - sum durations
            if incr_usage.duration is not None:
                if self.duration is None:
                    self.duration = incr_usage.duration
                else:
                    self.duration += incr_usage.duration
            
            # Handle time_to_first_token - keep the first/smallest
            if incr_usage.time_to_first_token is not None:
                if self.time_to_first_token is None:
                    self.time_to_first_token = incr_usage.time_to_first_token
            
            # Handle provider_metrics - merge dicts
            if incr_usage.provider_metrics:
                if self.provider_metrics is None:
                    self.provider_metrics = {}
                self.provider_metrics.update(incr_usage.provider_metrics)
            
            # Handle additional_metrics - merge dicts
            if incr_usage.additional_metrics:
                if self.additional_metrics is None:
                    self.additional_metrics = {}
                self.additional_metrics.update(incr_usage.additional_metrics)
        elif isinstance(incr_usage, RequestUsage):
            # RequestUsage counts as 1 request
            self.requests += 1
        
        # Extract special token types from details dict
        # These are provider-specific fields that should be tracked at top level
        if incr_usage.details:
            # Extract reasoning tokens (OpenAI o1/o3 models)
            if 'reasoning_tokens' in incr_usage.details:
                self.reasoning_tokens += incr_usage.details.get('reasoning_tokens', 0)
        
        # _incr_usage_tokens handles all token fields including output_audio_tokens
        return _incr_usage_tokens(self, incr_usage)

    def __add__(self, other: RunUsage | RequestUsage) -> RunUsage:
        """Add two RunUsages together.

        This is provided so it's trivial to sum usage information from multiple runs.
        """
        new_usage = copy(self)
        # Deep copy mutable fields to avoid sharing references
        if self.details:
            new_usage.details = self.details.copy()
        if self.provider_metrics:
            new_usage.provider_metrics = self.provider_metrics.copy()
        if self.additional_metrics:
            new_usage.additional_metrics = self.additional_metrics.copy()
        new_usage.incr(other)
        return new_usage

    def __radd__(self, other: RunUsage | RequestUsage | int) -> RunUsage:
        """Right add to support sum() starting with 0."""
        if other == 0:
            return self
        return self + other

    # Timer methods
    def start_timer(self) -> None:
        """Start the internal timer for tracking execution time."""
        from upsonic.utils.timer import Timer
        if self.timer is None:
            self.timer = Timer()
        self.timer.start()

    def stop_timer(self, set_duration: bool = True) -> None:
        """Stop the internal timer and optionally set duration.
        
        Args:
            set_duration: If True, set self.duration from timer.elapsed
        """
        if self.timer is not None:
            self.timer.stop()
            if set_duration:
                self.duration = self.timer.elapsed

    def set_time_to_first_token(self) -> None:
        """Record the time to first token from timer's elapsed time."""
        if self.timer is not None:
            self.time_to_first_token = self.timer.elapsed

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization.
        
        Excludes the timer object and any None/zero values for cleaner output.
        
        Returns:
            Dictionary representation of the usage metrics.
        """
        result: Dict[str, Any] = {
            "requests": self.requests,
            "tool_calls": self.tool_calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "input_audio_tokens": self.input_audio_tokens,
            "cache_audio_read_tokens": self.cache_audio_read_tokens,
            "output_audio_tokens": self.output_audio_tokens,
            "reasoning_tokens": self.reasoning_tokens,
        }
        
        # Add optional fields only if they have values
        if self.details:
            result["details"] = self.details
        if self.cost is not None:
            result["cost"] = self.cost
        if self.duration is not None:
            result["duration"] = self.duration
        if self.time_to_first_token is not None:
            result["time_to_first_token"] = self.time_to_first_token
        if self.provider_metrics:
            result["provider_metrics"] = self.provider_metrics
        if self.additional_metrics:
            result["additional_metrics"] = self.additional_metrics
        
        # Remove zero values for cleaner output
        result = {
            k: v for k, v in result.items()
            if v is not None and (not isinstance(v, (int, float)) or v != 0) and (not isinstance(v, dict) or len(v) > 0)
        }
        
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RunUsage":
        """Reconstruct RunUsage from dictionary.
        
        Args:
            data: Dictionary containing RunUsage fields.
            
        Returns:
            RunUsage instance.
        """
        return cls(
            requests=data.get("requests", 0),
            tool_calls=data.get("tool_calls", 0),
            input_tokens=data.get("input_tokens", 0),
            output_tokens=data.get("output_tokens", 0),
            cache_write_tokens=data.get("cache_write_tokens", 0),
            cache_read_tokens=data.get("cache_read_tokens", 0),
            input_audio_tokens=data.get("input_audio_tokens", 0),
            cache_audio_read_tokens=data.get("cache_audio_read_tokens", 0),
            output_audio_tokens=data.get("output_audio_tokens", 0),
            reasoning_tokens=data.get("reasoning_tokens", 0),
            details=data.get("details", {}),
            cost=data.get("cost"),
            duration=data.get("duration"),
            time_to_first_token=data.get("time_to_first_token"),
            provider_metrics=data.get("provider_metrics"),
            additional_metrics=data.get("additional_metrics"),
        )

    def update_from_request_usage(self, request_usage: RequestUsage) -> None:
        """Update this RunUsage from a RequestUsage (from model response).
        
        This is a convenience method for updating usage from model responses.
        
        Args:
            request_usage: The RequestUsage from a model response.
        """
        self.incr(request_usage)


def _incr_usage_tokens(slf: RunUsage | RequestUsage, incr_usage: RunUsage | RequestUsage) -> None:
    """Increment the usage in place.

    Args:
        slf: The usage to increment.
        incr_usage: The usage to increment by.
    """
    slf.input_tokens += incr_usage.input_tokens
    slf.cache_write_tokens += incr_usage.cache_write_tokens
    slf.cache_read_tokens += incr_usage.cache_read_tokens
    slf.input_audio_tokens += incr_usage.input_audio_tokens
    slf.cache_audio_read_tokens += incr_usage.cache_audio_read_tokens
    slf.output_tokens += incr_usage.output_tokens
    slf.output_audio_tokens += incr_usage.output_audio_tokens

    for key, value in incr_usage.details.items():
        slf.details[key] = slf.details.get(key, 0) + value


@dataclass(repr=False, kw_only=True)
@deprecated('`Usage` is deprecated, use `RunUsage` instead')
class Usage(RunUsage):
    """Deprecated alias for `RunUsage`."""


@dataclass(repr=False, kw_only=True)
class UsageLimits:
    """Limits on model usage.

    The request count is tracked by upsonic, and the request limit is checked before each request to the model.
    Token counts are provided in responses from the model, and the token limits are checked after each response.

    Each of the limits can be set to `None` to disable that limit.
    """

    request_limit: int | None = 50
    """The maximum number of requests allowed to the model."""
    tool_calls_limit: int | None = None
    """The maximum number of successful tool calls allowed to be executed."""
    input_tokens_limit: int | None = None
    """The maximum number of input/prompt tokens allowed."""
    output_tokens_limit: int | None = None
    """The maximum number of output/response tokens allowed."""
    total_tokens_limit: int | None = None
    """The maximum number of tokens allowed in requests and responses combined."""
    count_tokens_before_request: bool = False
    """If True, perform a token counting pass before sending the request to the model,
    to enforce `request_tokens_limit` ahead of time.

    This may incur additional overhead (from calling the model's `count_tokens` API before making the actual request) and is disabled by default.

    Supported by:

    - Anthropic
    - Google
    - Bedrock Converse

    Support for OpenAI is in development
    """

    @property
    @deprecated('`request_tokens_limit` is deprecated, use `input_tokens_limit` instead')
    def request_tokens_limit(self) -> int | None:
        return self.input_tokens_limit

    @property
    @deprecated('`response_tokens_limit` is deprecated, use `output_tokens_limit` instead')
    def response_tokens_limit(self) -> int | None:
        return self.output_tokens_limit

    @overload
    def __init__(
        self,
        *,
        request_limit: int | None = 50,
        tool_calls_limit: int | None = None,
        input_tokens_limit: int | None = None,
        output_tokens_limit: int | None = None,
        total_tokens_limit: int | None = None,
        count_tokens_before_request: bool = False,
    ) -> None:
        self.request_limit = request_limit
        self.tool_calls_limit = tool_calls_limit
        self.input_tokens_limit = input_tokens_limit
        self.output_tokens_limit = output_tokens_limit
        self.total_tokens_limit = total_tokens_limit
        self.count_tokens_before_request = count_tokens_before_request

    @overload
    @deprecated(
        'Use `input_tokens_limit` instead of `request_tokens_limit` and `output_tokens_limit` and `total_tokens_limit`'
    )
    def __init__(
        self,
        *,
        request_limit: int | None = 50,
        tool_calls_limit: int | None = None,
        request_tokens_limit: int | None = None,
        response_tokens_limit: int | None = None,
        total_tokens_limit: int | None = None,
        count_tokens_before_request: bool = False,
    ) -> None:
        self.request_limit = request_limit
        self.tool_calls_limit = tool_calls_limit
        self.input_tokens_limit = request_tokens_limit
        self.output_tokens_limit = response_tokens_limit
        self.total_tokens_limit = total_tokens_limit
        self.count_tokens_before_request = count_tokens_before_request

    def __init__(
        self,
        *,
        request_limit: int | None = 50,
        tool_calls_limit: int | None = None,
        input_tokens_limit: int | None = None,
        output_tokens_limit: int | None = None,
        total_tokens_limit: int | None = None,
        count_tokens_before_request: bool = False,
        # deprecated:
        request_tokens_limit: int | None = None,
        response_tokens_limit: int | None = None,
    ):
        self.request_limit = request_limit
        self.tool_calls_limit = tool_calls_limit
        self.input_tokens_limit = input_tokens_limit or request_tokens_limit
        self.output_tokens_limit = output_tokens_limit or response_tokens_limit
        self.total_tokens_limit = total_tokens_limit
        self.count_tokens_before_request = count_tokens_before_request

    def has_token_limits(self) -> bool:
        """Returns `True` if this instance places any limits on token counts.

        If this returns `False`, the `check_tokens` method will never raise an error.

        This is useful because if we have token limits, we need to check them after receiving each streamed message.
        If there are no limits, we can skip that processing in the streaming response iterator.
        """
        return any(
            limit is not None for limit in (self.input_tokens_limit, self.output_tokens_limit, self.total_tokens_limit)
        )

    def check_before_request(self, usage: RunUsage) -> None:
        """Raises a `UsageLimitExceeded` exception if the next request would exceed any of the limits."""
        request_limit = self.request_limit
        if request_limit is not None and usage.requests >= request_limit:
            raise UsageLimitExceeded(f'The next request would exceed the request_limit of {request_limit}')

        input_tokens = usage.input_tokens
        if self.input_tokens_limit is not None and input_tokens > self.input_tokens_limit:
            raise UsageLimitExceeded(
                f'The next request would exceed the input_tokens_limit of {self.input_tokens_limit} ({input_tokens=})'
            )

        total_tokens = usage.total_tokens
        if self.total_tokens_limit is not None and total_tokens > self.total_tokens_limit:
            raise UsageLimitExceeded(  # pragma: lax no cover
                f'The next request would exceed the total_tokens_limit of {self.total_tokens_limit} ({total_tokens=})'
            )

    def check_tokens(self, usage: RunUsage) -> None:
        """Raises a `UsageLimitExceeded` exception if the usage exceeds any of the token limits."""
        input_tokens = usage.input_tokens
        if self.input_tokens_limit is not None and input_tokens > self.input_tokens_limit:
            raise UsageLimitExceeded(f'Exceeded the input_tokens_limit of {self.input_tokens_limit} ({input_tokens=})')

        output_tokens = usage.output_tokens
        if self.output_tokens_limit is not None and output_tokens > self.output_tokens_limit:
            raise UsageLimitExceeded(
                f'Exceeded the output_tokens_limit of {self.output_tokens_limit} ({output_tokens=})'
            )

        total_tokens = usage.total_tokens
        if self.total_tokens_limit is not None and total_tokens > self.total_tokens_limit:
            raise UsageLimitExceeded(f'Exceeded the total_tokens_limit of {self.total_tokens_limit} ({total_tokens=})')

    def check_before_tool_call(self, projected_usage: RunUsage) -> None:
        """Raises a `UsageLimitExceeded` exception if the next tool call(s) would exceed the tool call limit."""
        tool_calls_limit = self.tool_calls_limit
        tool_calls = projected_usage.tool_calls
        if tool_calls_limit is not None and tool_calls > tool_calls_limit:
            raise UsageLimitExceeded(
                f'The next tool call(s) would exceed the tool_calls_limit of {tool_calls_limit} ({tool_calls=}).'
            )

    __repr__ = _utils.dataclasses_no_defaults_repr