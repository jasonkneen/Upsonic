from __future__ import annotations

from dataclasses import dataclass, field
from time import time as current_time
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
    Type,
    Union,
    Literal
)

from upsonic.run.base import RunStatus

if TYPE_CHECKING:
    from pydantic import BaseModel
    from upsonic.messages.messages import (
        BinaryContent,
        ModelMessage,
        ModelRequest,
        ModelResponse,
        ThinkingPart,
    )
    from upsonic.run.agent.input import AgentRunInput
    from upsonic.run.events.events import AgentEvent
    from upsonic.run.requirements import RunRequirement
    from upsonic.run.tools.tools import ToolExecution
    from upsonic.usage import RequestUsage
    from upsonic.profiles import ModelProfile
    from upsonic.agent.pipeline.step import StepResult
    from upsonic.run.pipeline.stats import PipelineExecutionStats
    from upsonic.tasks.tasks import Task
    from upsonic.schemas.kb_filter import KBFilterExpr


@dataclass
class AgentRunOutput:
    """Complete output and runtime context for an agent run.
    
    This is the SINGLE SOURCE OF TRUTH for agent run state. It combines:
    - Runtime execution context
    - User-facing output and results
    
    Handles:
    - Message tracking with run boundaries
    - Chat history for LLM execution
    - Tool execution results and tracking
    - HITL requirements (external tool calls only)
    - Streaming state with async context manager
    - Event streaming
    - Text output streaming
    - Step execution tracking
    - Comprehensive serialization
    
    Attributes:
        run_id: Unique identifier for this run
        session_id: Session identifier
        user_id: User identifier
        task: Embedded task (single source of truth for HITL)
        step_results: List of StepResult tracking each step's execution
        execution_stats: Pipeline execution statistics
        requirements: HITL requirements for external tool calls
        agent_knowledge_base_filter: Vector search filters
        session_state: Session state persisted across runs
        output_schema: Output schema constraint
        is_streaming: Whether this is streaming execution
        accumulated_text: Text accumulated during streaming
        chat_history: Full conversation history (historical + current run messages) for LLM
        messages: Only NEW messages from THIS run
        response: Current ModelResponse (single, not list)
        output: Final processed output (str or bytes)
        events: All events emitted during execution
    """
    
    # --- Identity ---
    run_id: Optional[str] = None
    agent_id: Optional[str] = None
    agent_name: Optional[str] = None
    session_id: Optional[str] = None
    parent_run_id: Optional[str] = None
    user_id: Optional[str] = None
    
    # --- Task (embedded for single source of truth) ---
    task: Optional["Task"] = None
    
    # Input reference
    input: Optional["AgentRunInput"] = None
    
    # --- Output ---
    output: Optional[Union[str, bytes]] = None  # Final agent output
    output_schema: Optional[Union[str, Type["BaseModel"]]] = None
    
    # Thinking/reasoning
    thinking_content: Optional[str] = None
    thinking_parts: Optional[List["ThinkingPart"]] = None
    
    # --- Model info ---
    model_name: Optional[str] = None
    model_provider: Optional[str] = None
    model_provider_profile: Optional["ModelProfile"] = None
    
    # --- Messages ---
    # chat_history: Full conversation history (historical + current) for LLM execution FOR THE SESSION, ALL RUNS
    chat_history: List["ModelMessage"] = field(default_factory=list)
    # messages: Only NEW messages from THIS run
    messages: Optional[List["ModelMessage"]] = None
    # response: Current ModelResponse
    response: Optional["ModelResponse"] = None
    usage: Optional["RequestUsage"] = None
    additional_input_message: Optional[List["ModelRequest"]] = None
    
    # Memory tracking
    memory_message_count: int = 0
    
    # --- Tool executions ---
    tools: Optional[List["ToolExecution"]] = None
    tool_call_count: int = 0
    tool_limit_reached: bool = False
    
    # --- Media outputs ---
    images: Optional[List["BinaryContent"]] = None
    files: Optional[List["BinaryContent"]] = None
    
    # --- Status and HITL ---
    status: RunStatus = field(default_factory=lambda: RunStatus.running)
    requirements: Optional[List["RunRequirement"]] = None  # External tool requirements only
    step_results: List["StepResult"] = field(default_factory=list)

    # Pipeline execution statistics
    execution_stats: Optional["PipelineExecutionStats"] = None
    
    # --- Events (for streaming) ---
    events: Optional[List["AgentEvent"]] = None
    
    # --- Configuration ---
    agent_knowledge_base_filter: Optional["KBFilterExpr"] = None
    
    # --- Metadata ---
    metadata: Optional[Dict[str, Any]] = None
    session_state: Optional[Dict[str, Any]] = None
    
    # --- Execution state ---
    is_streaming: bool = False
    accumulated_text: str = ""
    
    # Current step result (set by Step.execute, read by Step.run)
    current_step_result: Optional["StepResult"] = None
    
    # --- User-facing pause information ---
    pause_reason: Optional[Literal["external_tool"]] = None  # "external_tool" only now
    error_details: Optional[str] = None
    
    # --- Message tracking (internal) ---
    _run_boundaries: List[int] = field(default_factory=list)
    
    # --- Timestamps ---
    created_at: int = field(default_factory=lambda: int(current_time()))
    updated_at: Optional[int] = None
    
    # ========================================================================
    # Properties
    # ========================================================================
    
    @property
    def is_paused(self) -> bool:
        """Check if the run is paused (external tool execution)."""
        return self.status == RunStatus.paused
    
    @property
    def is_cancelled(self) -> bool:
        """Check if the run was cancelled."""
        return self.status == RunStatus.cancelled
    
    @property
    def is_complete(self) -> bool:
        """Check if the run is complete."""
        return self.status == RunStatus.completed
    
    @property
    def is_error(self) -> bool:
        """Check if the run has an error."""
        return self.status == RunStatus.error
    
    @property
    def is_problematic(self) -> bool:
        """Check if the run is problematic (paused, cancelled, or error) and needs continue_run_async."""
        return self.status in (RunStatus.paused, RunStatus.cancelled, RunStatus.error)
    
    @property
    def active_requirements(self) -> List["RunRequirement"]:
        """Get unresolved external tool requirements."""
        if not self.requirements:
            return []
        return [req for req in self.requirements if req.needs_external_execution]
    
    @property
    def tools_requiring_confirmation(self) -> List["ToolExecution"]:
        """Get tools that require user confirmation."""
        if not self.tools:
            return []
        return [t for t in self.tools if t.requires_confirmation and not t.confirmed]
    
    @property
    def tools_requiring_user_input(self) -> List["ToolExecution"]:
        """Get tools that require user input."""
        if not self.tools:
            return []
        return [t for t in self.tools if t.requires_user_input and not t.answered]
    
    @property
    def tools_awaiting_external_execution(self) -> List["ToolExecution"]:
        """Get tools awaiting external execution."""
        if not self.tools:
            return []
        return [t for t in self.tools if t.external_execution_required and t.result is None]
    
    # ========================================================================
    # Requirement Methods (External Tool Only)
    # ========================================================================
    
    def add_requirement(self, requirement: "RunRequirement") -> None:
        """Add an external tool requirement."""
        if self.requirements is None:
            self.requirements = []
        self.requirements.append(requirement)
    
    def get_external_tool_requirements(self) -> List["RunRequirement"]:
        """Get all external tool requirements."""
        if not self.requirements:
            return []
        return list(self.requirements)
    
    def get_external_tool_requirements_with_results(self) -> List["RunRequirement"]:
        """Get external tool requirements that have results."""
        if not self.requirements:
            return []
        return [
            r for r in self.requirements 
            if r.tool_execution and r.tool_execution.result is not None
        ]
    
    def has_pending_external_tools(self) -> bool:
        """Check if there are any external tools awaiting execution."""
        return len(self.active_requirements) > 0
    
    # ========================================================================
    # Step Result Methods
    # ========================================================================
    
    def get_step_results(self) -> List["StepResult"]:
        """Get step results from execution."""
        if self.step_results:
            return self.step_results
        return []
    
    def get_execution_stats(self) -> Optional["PipelineExecutionStats"]:
        """Get execution statistics."""
        return self.execution_stats
    
    def get_last_successful_step(self) -> Optional["StepResult"]:
        """Get the last successfully completed step."""
        from upsonic.agent.pipeline.step import StepStatus
        for result in reversed(self.step_results):
            if result.status == StepStatus.COMPLETED:
                return result
        return None
    
    def get_error_step(self) -> Optional["StepResult"]:
        """Get the step that failed with ERROR status (for durable execution)."""
        from upsonic.agent.pipeline.step import StepStatus
        for result in self.step_results:
            if result.status == StepStatus.ERROR:
                return result
        return None
    
    def get_cancelled_step(self) -> Optional["StepResult"]:
        """Get the step that was CANCELLED (for cancel run resumption)."""
        from upsonic.agent.pipeline.step import StepStatus
        for result in self.step_results:
            if result.status == StepStatus.CANCELLED:
                return result
        return None
    
    def get_paused_step(self) -> Optional["StepResult"]:
        """Get the step that is PAUSED (for external tool execution)."""
        from upsonic.agent.pipeline.step import StepStatus
        for result in self.step_results:
            if result.status == StepStatus.PAUSED:
                return result
        return None
    
    def get_problematic_step(self) -> Optional["StepResult"]:
        """Get the step that caused the run to be problematic (error, cancelled, or paused)."""
        return self.get_error_step() or self.get_cancelled_step() or self.get_paused_step()
    
    # ========================================================================
    # Message Tracking Methods
    # ========================================================================
    
    def all_messages(self) -> List["ModelMessage"]:
        """Get all messages from the run."""
        return (self.messages or []).copy()
    
    def new_messages(self) -> List["ModelMessage"]:
        """Get messages from the last iteration only."""
        if not self._run_boundaries or not self.messages:
            return (self.messages or []).copy()
        last_start = self._run_boundaries[-1]
        return self.messages[last_start:].copy()
    
    def add_messages(self, messages: List["ModelMessage"]) -> None:
        """Add messages to the run."""
        if self.messages is None:
            self.messages = []
        self.messages.extend(messages)
    
    def add_message(self, message: "ModelMessage") -> None:
        """Add a single message to the run."""
        if self.messages is None:
            self.messages = []
        self.messages.append(message)
    
    def start_new_run(self) -> None:
        """Initialize a new run by starting a new iteration."""
        self.start_new_iteration()
    
    def start_new_iteration(self) -> None:
        """Mark the start of a new tool call iteration."""
        if self.messages:
            self._run_boundaries.append(len(self.messages))
        else:
            self._run_boundaries.append(0)
    
    def get_last_model_response(self) -> Optional["ModelResponse"]:
        """Get the last ModelResponse from the messages."""
        from upsonic.messages.messages import ModelResponse
        
        messages = self.new_messages()
        for msg in reversed(messages):
            if isinstance(msg, ModelResponse):
                return msg
        return None
    
    # ========================================================================
    # Status Methods
    # ========================================================================
    
    def _sync_status_to_task(self) -> None:
        """Sync the current status to the embedded task."""
        if self.task is not None:
            self.task.status = self.status
    
    def mark_paused(self, reason: Literal["external_tool"] = "external_tool") -> None:
        """Mark the run as paused for external tool execution."""
        self.status = RunStatus.paused
        self.pause_reason = reason
        self.updated_at = int(current_time())
        self._sync_status_to_task()
    
    def mark_cancelled(self) -> None:
        """Mark the run as cancelled."""
        self.status = RunStatus.cancelled
        self.updated_at = int(current_time())
        self._sync_status_to_task()
    
    def mark_completed(self) -> None:
        """Mark the run as completed and finalize output."""
        self.status = RunStatus.completed
        self.updated_at = int(current_time())
        # Set output from accumulated_text if streaming
        if self.output is None and self.accumulated_text and self.is_streaming:
            self.output = self.accumulated_text
        self._sync_status_to_task()
    
    def mark_error(self, error: Optional[str] = None) -> None:
        """Mark the run as having an error."""
        self.status = RunStatus.error
        if error:
            self.error_details = error
            if self.metadata is None:
                self.metadata = {}
            self.metadata["error"] = error
        self.updated_at = int(current_time())
        self._sync_status_to_task()
    
    # ========================================================================
    # Serialization
    # ========================================================================
    
    def _serialize_message(self, msg: Any) -> Any:
        """Serialize a message to dict if it has to_dict method."""
        if msg is None:
            return None
        if hasattr(msg, 'to_dict'):
            return msg.to_dict()
        return msg
    
    def _serialize_item(self, item: Any) -> Any:
        """Serialize an item to dict if it has to_dict method."""
        if item is None:
            return None
        if hasattr(item, 'to_dict'):
            return item.to_dict()
        return item
    
    def _serialize_list(self, items: Optional[List[Any]]) -> Optional[List[Any]]:
        """Serialize a list of items, calling to_dict on each if available."""
        if items is None:
            return None
        return [self._serialize_item(item) for item in items]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (pure transformation, nested objects call their to_dict())."""
        return {
            "run_id": self.run_id,
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "session_id": self.session_id,
            "parent_run_id": self.parent_run_id,
            "user_id": self.user_id,
            "task": self.task.to_dict() if self.task and hasattr(self.task, 'to_dict') else None,
            "status": self.status.value if hasattr(self.status, 'value') else self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "input": self.input.to_dict() if self.input else None,
            "output": self.output,
            "output_schema": str(self.output_schema) if self.output_schema else None,
            "thinking_content": self.thinking_content,
            "thinking_parts": self._serialize_list(self.thinking_parts),
            "model_name": self.model_name,
            "model_provider": self.model_provider,
            "model_provider_profile": self._serialize_item(self.model_provider_profile),
            "chat_history": [self._serialize_message(m) for m in self.chat_history] if self.chat_history else [],
            "messages": self._serialize_list(self.messages),
            "response": self._serialize_message(self.response),
            "usage": self._serialize_item(self.usage),
            "additional_input_message": self._serialize_list(self.additional_input_message),
            "memory_message_count": self.memory_message_count,
            "tools": [t.to_dict() for t in self.tools] if self.tools else None,
            "tool_call_count": self.tool_call_count,
            "tool_limit_reached": self.tool_limit_reached,
            "images": self._serialize_list(self.images),
            "files": self._serialize_list(self.files),
            "requirements": [r.to_dict() for r in self.requirements] if self.requirements else None,
            "step_results": [sr.to_dict() for sr in self.step_results] if self.step_results else [],
            "execution_stats": self.execution_stats.to_dict() if self.execution_stats else None,
            "events": self._serialize_list(self.events),
            "agent_knowledge_base_filter": self.agent_knowledge_base_filter,
            "metadata": self.metadata,
            "session_state": self.session_state,
            "is_streaming": self.is_streaming,
            "accumulated_text": self.accumulated_text,
            "_run_boundaries": self._run_boundaries,
            "pause_reason": self.pause_reason,
            "error_details": self.error_details,
        }
    
    def to_json(self, indent: Optional[int] = 2) -> str:
        """Serialize to JSON string."""
        import json
        return json.dumps(self.to_dict(), indent=indent)
    
    @classmethod
    def _deserialize_messages(cls, messages_data: List[Any]) -> List[Any]:
        """Helper to deserialize a list of messages."""
        messages = []
        for m in messages_data:
            if isinstance(m, dict) and 'parts' in m:
                # It's a serialized ModelRequest/ModelResponse - keep as-is for cloudpickle
                messages.append(m)
            else:
                messages.append(m)
        return messages
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentRunOutput":
        """Reconstruct from dictionary (pure transformation, nested objects reconstructed)."""
        from upsonic.run.agent.input import AgentRunInput
        from upsonic.run.requirements import RunRequirement
        from upsonic.run.tools.tools import ToolExecution
        from upsonic.run.events.events import AgentEvent
        from upsonic.run.pipeline.stats import PipelineExecutionStats
        from upsonic.agent.pipeline.step import StepResult
        from upsonic.tasks.tasks import Task
        from upsonic.profiles import ModelProfile

        # Handle step_results (list of dicts)
        step_results: List[Any] = []
        step_results_data = data.get("step_results", [])
        for sr in step_results_data:
            if isinstance(sr, dict):
                step_results.append(StepResult.from_dict(sr))
            else:
                step_results.append(sr)

        # Handle execution_stats (dict)
        execution_stats = None
        execution_stats_data = data.get("execution_stats")
        if isinstance(execution_stats_data, dict):
            execution_stats = PipelineExecutionStats.from_dict(execution_stats_data)
        else:
            execution_stats = execution_stats_data
        
        # Handle input (dict or object)
        input_data = data.get("input")
        input_obj = None
        if isinstance(input_data, dict):
            input_obj = AgentRunInput.from_dict(input_data)
        else:
            input_obj = input_data
        
        # Handle task (dict or Task object)
        task = None
        task_data = data.get("task")
        if isinstance(task_data, dict):
            if hasattr(Task, 'from_dict'):
                task = Task.from_dict(task_data)
            else:
                task = Task.model_validate(task_data)
        else:
            task = task_data
        
        # Handle model_provider_profile (dict or ModelProfile)
        model_provider_profile = None
        model_provider_profile_data = data.get("model_provider_profile")
        if isinstance(model_provider_profile_data, dict):
            model_provider_profile = ModelProfile.from_dict(model_provider_profile_data)
        else:
            model_provider_profile = model_provider_profile_data
        
        # Handle tools (list of dicts or objects)
        tools = None
        tools_data = data.get("tools")
        if tools_data:
            tools = []
            for t in tools_data:
                if isinstance(t, dict):
                    tools.append(ToolExecution.from_dict(t))
                else:
                    tools.append(t)
        
        # Handle requirements (list of dicts or objects)
        requirements = None
        requirements_data = data.get("requirements")
        if requirements_data:
            requirements = []
            for r in requirements_data:
                if isinstance(r, dict):
                    requirements.append(RunRequirement.from_dict(r))
                else:
                    requirements.append(r)
        
        # Handle status (str or RunStatus)
        status = data.get("status", RunStatus.running)
        if isinstance(status, str):
            status = RunStatus(status)
        
        # Handle events (list of dicts or objects)
        events = None
        events_data = data.get("events")
        if events_data:
            events = []
            for e in events_data:
                if isinstance(e, dict):
                    events.append(AgentEvent.from_dict(e))
                else:
                    events.append(e)
        
        # Handle chat_history (list of dicts or objects)
        chat_history = cls._deserialize_messages(data.get("chat_history", []))
        
        return cls(
            run_id=data.get("run_id"),
            agent_id=data.get("agent_id"),
            agent_name=data.get("agent_name"),
            session_id=data.get("session_id"),
            parent_run_id=data.get("parent_run_id"),
            user_id=data.get("user_id"),
            task=task,
            status=status,
            created_at=data.get("created_at", int(current_time())),
            updated_at=data.get("updated_at"),
            input=input_obj,
            output=data.get("output"),
            output_schema=data.get("output_schema"),
            thinking_content=data.get("thinking_content"),
            thinking_parts=data.get("thinking_parts"),
            model_name=data.get("model_name"),
            model_provider=data.get("model_provider"),
            model_provider_profile=model_provider_profile,
            chat_history=chat_history,
            messages=data.get("messages"),
            response=data.get("response"),
            usage=data.get("usage"),
            additional_input_message=data.get("additional_input_message"),
            memory_message_count=data.get("memory_message_count", 0),
            tools=tools,
            tool_call_count=data.get("tool_call_count", 0),
            tool_limit_reached=data.get("tool_limit_reached", False),
            images=data.get("images"),
            files=data.get("files"),
            requirements=requirements,
            step_results=step_results,
            execution_stats=execution_stats,
            events=events,
            agent_knowledge_base_filter=data.get("agent_knowledge_base_filter"),
            metadata=data.get("metadata"),
            session_state=data.get("session_state"),
            is_streaming=data.get("is_streaming", False),
            accumulated_text=data.get("accumulated_text", ""),
            _run_boundaries=data.get("_run_boundaries", []),
            pause_reason=data.get("pause_reason"),
            error_details=data.get("error_details"),
        )
    
    def __str__(self) -> str:
        return str(self.output if self.output is not None else "")
    
    def __repr__(self) -> str:
        return f"AgentRunOutput(run_id={self.run_id!r}, status={self.status.value}, output_length={len(str(self.output or ''))})"
    
    def serialize(self) -> bytes:
        """Serialize to bytes for storage."""
        import cloudpickle
        return cloudpickle.dumps(self.to_dict())
    
    @classmethod
    def deserialize(cls, data: bytes) -> "AgentRunOutput":
        """Deserialize from bytes."""
        import cloudpickle
        dict_data = cloudpickle.loads(data)
        return cls.from_dict(dict_data)
