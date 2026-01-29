import time
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

# Heavy imports moved to lazy loading for faster startup
if TYPE_CHECKING:
    from upsonic.models import Model
    from upsonic.utils.printing import call_end
    from upsonic.utils.llm_usage import llm_usage
    from upsonic.utils.tool_usage import tool_usage
else:
    # Use string annotations to avoid importing heavy modules
    Model = "Model"
    call_end = "call_end"
    llm_usage = "llm_usage"
    tool_usage = "tool_usage"


class CallManager:
    def __init__(self, model: "Model", task, debug=False, show_tool_calls=True, print_output=False):
        """
        Initializes the CallManager.

        Args:
            model: The instantiated model object for this call.
            task: The task being executed.
            debug: Whether debug mode is enabled.
            show_tool_calls: Whether to show tool calls.
            print_output: Whether to print output to console.
        """
        self.model = model
        self.task = task
        self.show_tool_calls = show_tool_calls
        self.debug = debug
        self.print_output = print_output
        self.start_time = None
        self.end_time = None
        self.model_response = None
        
    def process_response(self, model_response):
        self.model_response = model_response
        return self.model_response
    
    async def aprepare(self) -> None:
        """Prepare the call by recording start time."""
        self.start_time = time.time()
    
    async def afinalize(self) -> None:
        """Finalize the call by setting end time."""
        # Only set end_time if not already set (may be set from context.model_end_time)
        if self.end_time is None:
            self.end_time = time.time()
        
        # Ensure start_time is set (fallback to end_time if not set)
        if self.start_time is None:
            self.start_time = self.end_time
    
    async def alog_completion(self, context) -> None:
        """Log the completion with usage tracking and call_end.
        
        Args:
            context: AgentRunOutput object containing messages and output.
        """
        # Only call call_end if we have a context
        if context is not None:
            # Lazy import for heavy modules
            from upsonic.utils.llm_usage import llm_usage
            from upsonic.utils.tool_usage import tool_usage
            from upsonic.utils.printing import call_end
            
            # Calculate usage and tool usage from context (AgentRunOutput)
            usage = llm_usage(context)
            if self.show_tool_calls:
                tool_usage_result = tool_usage(context, self.task)
            else:
                tool_usage_result = None
            # Call the end logging
            call_end(
                context.output,
                self.model,
                self.task.response_format,
                self.start_time,
                self.end_time,
                usage,
                tool_usage_result,
                self.debug,
                self.task.price_id,
                print_output=self.print_output
            )
    
    def prepare(self) -> None:
        """Synchronous version of aprepare."""
        import asyncio
        asyncio.get_event_loop().run_until_complete(self.aprepare())
    
    def finalize(self) -> None:
        """Synchronous version of afinalize."""
        import asyncio
        asyncio.get_event_loop().run_until_complete(self.afinalize())
    
    def log_completion(self, context) -> None:
        """Synchronous version of alog_completion."""
        import asyncio
        asyncio.get_event_loop().run_until_complete(self.alog_completion(context))
    
    @asynccontextmanager
    async def manage_call(self):
        """
        Async context manager for call lifecycle.
        
        Note: This context manager is kept for backward compatibility.
        For step-based architecture, use aprepare() and afinalize() directly.
        """
        await self.aprepare()
        
        try:
            yield self
        finally:
            await self.afinalize()
            # Note: alog_completion requires context, so it's not called here
            # It should be called explicitly from the step that has access to context