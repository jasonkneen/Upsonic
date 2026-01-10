"""
Base Step class for pipeline execution.

Steps are the building blocks of the agent execution pipeline.
Each step performs a specific operation and can modify the output context.
PipelineManager passes task, agent, model, and step_number to each step.

Architecture:
- execute() is the core logic with full step lifecycle (try-except-finally)
- execute_stream() is for streaming steps, yields AgentEvent objects
- run() is a minimal wrapper that just calls execute() and returns result
- run_stream() wraps execute_stream() for streaming execution
"""

from abc import ABC, abstractmethod
from typing import Optional, Any, AsyncIterator, Dict, TYPE_CHECKING
from enum import Enum
from pydantic import BaseModel, Field, ConfigDict

if TYPE_CHECKING:
    from upsonic.run.agent.output import AgentRunOutput
    from upsonic.tasks.tasks import Task
    from upsonic.models import Model
    from upsonic.agent.agent import Agent
    from upsonic.run.base import RunStatus
    from upsonic.run.events.events import AgentEvent
else:
    AgentRunOutput = "AgentRunOutput"
    Task = "Task"
    Model = "Model"
    Agent = "Agent"
    RunStatus = "RunStatus"
    AgentEvent = "AgentEvent"


# Error injection registry for testing durable execution
# Maps step_name -> (error_class, error_message, trigger_count)
# trigger_count: number of times to trigger before stopping (0 = infinite)
_ERROR_INJECTION_REGISTRY: dict = {}
_ERROR_INJECTION_COUNTERS: dict = {}


def inject_error_into_step(step_name: str, error_class: type = Exception, error_message: str = "Injected test error", trigger_count: int = 1):
    """
    Inject an error into a specific step for testing durable execution.
    
    Args:
        step_name: Name of the step to inject error into (e.g., "model_execution")
        error_class: Exception class to raise
        error_message: Error message
        trigger_count: Number of times to trigger the error (1 = once, then step works)
    """
    _ERROR_INJECTION_REGISTRY[step_name] = (error_class, error_message, trigger_count)
    _ERROR_INJECTION_COUNTERS[step_name] = 0


def clear_error_injection(step_name: str = None):
    """Clear error injection for a step or all steps."""
    if step_name:
        _ERROR_INJECTION_REGISTRY.pop(step_name, None)
        _ERROR_INJECTION_COUNTERS.pop(step_name, None)
    else:
        _ERROR_INJECTION_REGISTRY.clear()
        _ERROR_INJECTION_COUNTERS.clear()


def check_and_raise_injected_error(step_name: str):
    """Check if an error should be raised for this step and raise it if so."""
    if step_name in _ERROR_INJECTION_REGISTRY:
        error_class, error_message, trigger_count = _ERROR_INJECTION_REGISTRY[step_name]
        current_count = _ERROR_INJECTION_COUNTERS.get(step_name, 0)
        
        # Check if we should still trigger
        if trigger_count == 0 or current_count < trigger_count:
            _ERROR_INJECTION_COUNTERS[step_name] = current_count + 1
            raise error_class(f"[INJECTED ERROR] {step_name}: {error_message}")


class StepStatus(str, Enum):
    """Status of step execution - synced with RunStatus."""
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    PAUSED = "PAUSED"
    CANCELLED = "CANCELLED"
    ERROR = "ERROR"
    SKIPPED = "SKIPPED"
    
    @staticmethod
    def to_run_status(step_status: "StepStatus") -> "RunStatus":
        from upsonic.run.base import RunStatus
        
        mapping = {
            StepStatus.RUNNING: RunStatus.running,
            StepStatus.COMPLETED: RunStatus.completed,
            StepStatus.PAUSED: RunStatus.paused,
            StepStatus.CANCELLED: RunStatus.cancelled,
            StepStatus.ERROR: RunStatus.error,
            StepStatus.SKIPPED: RunStatus.completed,
        }
        return mapping.get(step_status, RunStatus.running)


class StepResult(BaseModel):
    """Result of a step execution. All attributes are set in steps.py execute() method."""
    name: str = Field(default="", description="Step name")
    step_number: int = Field(default=0, description="Step index in pipeline (0-based)")
    status: StepStatus = Field(description="Step execution status")
    message: Optional[str] = Field(default=None, description="Step message")
    execution_time: float = Field(default=0.0, description="Execution time in seconds")

    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize result to dictionary."""
        return {
            "name": self.name,
            "step_number": self.step_number,
            "status": self.status.value,
            "message": self.message,
            "execution_time": self.execution_time,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StepResult":
        """Deserialize result from dictionary."""
        return cls(
            name=data["name"],
            step_number=data["step_number"],
            status=StepStatus(data["status"]),
            message=data.get("message"),
            execution_time=data.get("execution_time", 0.0),
        )


class Step(ABC):
    """
    Base class for pipeline steps.
    
    Each step performs a specific operation in the agent execution pipeline.
    
    Architecture:
    - execute() is the core logic, returns StepResult
    - execute_stream() is for streaming steps, yields AgentEvent objects
    - run() wraps execute() with timing and metadata, returns StepResult
    - run_stream() wraps execute_stream() with timing and metadata, yields events
    """
    
    def __init__(self):
        """Initialize the step."""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of this step."""
        pass
    
    @property
    def description(self) -> str:
        """Return a description of what this step does."""
        return f"Executes {self.name}"
    
    @property
    def supports_streaming(self) -> bool:
        """Whether this step supports streaming via execute_stream()."""
        return False
    
    def _finalize_step_result(
        self,
        step_result: Optional[StepResult],
        context: "AgentRunOutput",
    ) -> None:
        """
        Finalize step result by updating context.
        
        This method handles the common step lifecycle finalization:
        - Appends step_result to context.step_results
        - Updates context.execution_stats
        
        Args:
            step_result: The step result to finalize, or None if no result
            context: The agent run output (single source of truth)
        """
        if step_result:
            context.step_results.append(step_result)
            if context.execution_stats:
                context.execution_stats.executed_steps += 1
                context.execution_stats.step_timing[self.name] = step_result.execution_time
                context.execution_stats.step_statuses[self.name] = step_result.status.value
    
    @abstractmethod
    async def execute(
        self, 
        context: "AgentRunOutput", 
        task: "Task", 
        agent: "Agent", 
        model: "Model",
        step_number: int,
        pipeline_manager: Optional[Any] = None
    ) -> StepResult:
        """
        Execute the step's main logic with full lifecycle management.
        
        This is the core method that subclasses must implement. Each step
        should use try-except-finally pattern to:
        1. Check for cancellation via raise_if_cancelled
        2. Check for injected errors via check_and_raise_injected_error
        3. Execute business logic
        4. Create and return StepResult with all attributes set
        5. In finally: append result to context.step_results, update stats
        
        Args:
            context: The agent run output (single source of truth)
            task: The task being executed
            agent: The agent instance
            model: The model being used
            step_number: Index of this step in the pipeline (0-based)
            pipeline_manager: Optional PipelineManager instance for accessing manager registry
            
        Returns:
            StepResult: The result of the step execution with all attributes set
        """
        pass
    
    async def execute_stream(
        self, 
        context: "AgentRunOutput", 
        task: "Task", 
        agent: "Agent", 
        model: "Model",
        step_number: int,
        pipeline_manager: Optional[Any] = None
    ) -> AsyncIterator["AgentEvent"]:
        """
        Execute the step with streaming event emission.
        
        Override this method in steps that need to emit events during execution.
        The method should yield AgentEvent objects as they occur.
        At the end, it should set context.current_step_result with the StepResult.
        
        Default implementation calls execute() and yields any events that were
        appended to context.events during execution.
        
        Args:
            context: The agent run output (single source of truth)
            task: The task being executed
            agent: The agent instance
            model: The model being used
            step_number: Index of this step in the pipeline (0-based)
            pipeline_manager: Optional PipelineManager instance for accessing manager registry
            
        Yields:
            AgentEvent: Events generated during execution
        """
        # Track events before execution
        events_before = len(context.events) if context.events else 0
        
        # Execute the step
        result = await self.execute(context, task, agent, model, step_number, pipeline_manager=pipeline_manager)
        context.current_step_result = result
        
        # Yield any events that were appended during execution
        if context.events:
            for event in context.events[events_before:]:
                yield event
    
    async def run(
        self, 
        context: "AgentRunOutput", 
        task: "Task", 
        agent: "Agent", 
        model: "Model",
        step_number: int,
        pipeline_manager: Optional[Any] = None
    ) -> StepResult:
        """
        Wrapper that calls execute() and handles common step lifecycle management.
        
        This method handles the common finally block logic:
        - Appends step_result to context.step_results
        - Updates context.execution_stats
        
        Args:
            context: The agent run output (single source of truth)
            task: The task being executed
            agent: The agent instance
            model: The model being used
            step_number: Index of this step in the pipeline (0-based)
            pipeline_manager: Optional PipelineManager instance for accessing manager registry
            
        Returns:
            StepResult: The result of the execution
        """

        step_result = await self.execute(context, task, agent, model, step_number, pipeline_manager=pipeline_manager)
        return step_result

    
    async def run_stream(
        self, 
        context: "AgentRunOutput", 
        task: "Task", 
        agent: "Agent", 
        model: "Model",
        step_number: int,
        pipeline_manager: Optional[Any] = None
    ) -> AsyncIterator["AgentEvent"]:
        """
        Minimal wrapper for streaming execution.
        
        Yields step start/end events and calls execute_stream().
        All step lifecycle management is handled in execute_stream() in steps.py.
        
        Args:
            context: The agent run output (single source of truth)
            task: The task being executed
            agent: The agent instance
            model: The model being used
            step_number: Index of this step in the pipeline (0-based)
            
        Yields:
            AgentEvent: Events from step execution (including step start/end)
        """
        from upsonic.run.events.events import StepStartEvent, StepEndEvent
        
        run_id = agent.run_id if agent and hasattr(agent, 'run_id') else ""
        total_steps = context.execution_stats.total_steps if context.execution_stats else 1
        
        # Yield step start event
        yield StepStartEvent(
            run_id=run_id,
            step_name=self.name,
            step_description=self.description,
            step_index=step_number,
            total_steps=total_steps
        )
        
        # Initialize result holder in context
        context.current_step_result = None
        
        # Execute step - consume generator and yield all events
        async for event in self.execute_stream(context, task, agent, model, step_number, pipeline_manager=pipeline_manager):
            yield event
        
        # Get result from context (set by execute_stream())
        result = context.current_step_result
        
        # Yield step end event
        if result:
            yield StepEndEvent(
                run_id=run_id,
                step_name=self.name,
                step_index=step_number,
                status=result.status.value,
                execution_time=result.execution_time,
                message=result.message or ""
            )
