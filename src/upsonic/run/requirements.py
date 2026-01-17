import cloudpickle
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4

from upsonic.run.tools.tools import ToolExecution



@dataclass
class RunRequirement:
    """
    Requirement for external tool execution (HITL flows).
    
    This class ONLY handles external tool calls that require user execution
    outside the agent framework. For error recovery (durable execution) and
    cancel run resumption, use the AgentRunOutput status directly.
    """

    id: str = field(default_factory=lambda: str(uuid4()))
    tool_execution: Optional[ToolExecution] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # User confirmation
    confirmation: Optional[bool] = None
    confirmation_note: Optional[str] = None

    # User input
    user_input_schema: Optional[List[Dict[str, Any]]] = None

    def __init__(
        self,
        tool_execution: Optional[ToolExecution] = None,
        id: Optional[str] = None,
        created_at: Optional[datetime] = None,
    ):
        self.id = id or str(uuid4())
        self.tool_execution = tool_execution
        self.user_input_schema = getattr(tool_execution, 'user_input_schema', None) if tool_execution else None
        self.created_at = created_at or datetime.now(timezone.utc)
        self.confirmation = None
        self.confirmation_note = None

    @property
    def needs_confirmation(self) -> bool:
        """Check if this requirement needs user confirmation."""
        if self.confirmation is not None:
            return False
        if not self.tool_execution:
            return False
        if self.tool_execution.confirmed is True:
            return True
        return self.tool_execution.requires_confirmation or False

    @property
    def needs_user_input(self) -> bool:
        """Check if this requirement needs user input."""
        if not self.tool_execution:
            return False
        if self.tool_execution.answered is True:
            return False
        if self.user_input_schema and not all(value is not None for value in self.user_input_schema.values()):
            return True
        return self.tool_execution.requires_user_input or False

    @property
    def needs_external_execution(self) -> bool:
        """Check if this requirement needs external execution."""
        if not self.tool_execution:
            return False
        if self.tool_execution.result is not None:
            return False
        return self.tool_execution.external_execution_required or False

    @property
    def is_external_tool_execution(self) -> bool:
        """Check if this requirement is for external tool execution."""
        return self.needs_external_execution

    @property
    def has_result(self) -> bool:
        """Check if this requirement has a result."""
        return self.tool_execution is not None and self.tool_execution.result is not None

    @property
    def is_resolved(self) -> bool:
        """
        Check if this requirement has been resolved.
        
        A requirement is resolved when:
        - It no longer needs confirmation
        - It no longer needs user input
        - It no longer needs external execution (result has been provided)
        """
        return not self.needs_confirmation and not self.needs_user_input and not self.needs_external_execution

    def confirm(self) -> None:
        """Confirm the tool execution."""
        if not self.needs_confirmation:
            raise ValueError("This requirement does not require confirmation")
        self.confirmation = True
        if self.tool_execution:
            self.tool_execution.confirmed = True

    def reject(self) -> None:
        """Reject the tool execution."""
        if not self.needs_confirmation:
            raise ValueError("This requirement does not require confirmation")
        self.confirmation = False
        if self.tool_execution:
            self.tool_execution.confirmed = False

    def set_external_execution_result(self, result: str) -> None:
        """Set the result from external execution."""
        if not self.tool_execution:
            raise ValueError("No tool execution to set result for")
        self.tool_execution.result = result

    def mark_for_external_execution(self) -> None:
        """Mark this requirement's tool execution as requiring external execution."""
        if self.tool_execution:
            self.tool_execution.external_execution_required = True

    def _serialize_message(self, msg: Any) -> Any:
        """Serialize a message to dict if it has to_dict method."""
        if msg is None:
            return None
        if hasattr(msg, 'to_dict'):
            return msg.to_dict()
        return msg

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "confirmation": self.confirmation,
            "confirmation_note": self.confirmation_note,
            "user_input_schema": self.user_input_schema,
            "tool_execution": self.tool_execution.to_dict() if self.tool_execution else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RunRequirement":
        """Reconstruct from dictionary."""
        if data is None:
            raise ValueError("RunRequirement.from_dict() requires a non-None dict")

        # Handle tool_execution (dict only)
        tool_data = data.get("tool_execution")
        tool_execution: Optional[ToolExecution] = None
        if isinstance(tool_data, dict):
            tool_execution = ToolExecution.from_dict(tool_data)

        # Handle created_at (ISO string from dict)
        created_at_raw = data.get("created_at")
        created_at: Optional[datetime] = None
        if isinstance(created_at_raw, str):
            created_at = datetime.fromisoformat(created_at_raw)
        elif isinstance(created_at_raw, datetime):
            created_at = created_at_raw

        # Build requirement
        requirement = cls(
            tool_execution=tool_execution,
            id=data.get("id"),
            created_at=created_at,
        )
        
        # Set optional fields
        requirement.confirmation = data.get("confirmation")
        requirement.confirmation_note = data.get("confirmation_note")
        requirement.user_input_schema = data.get("user_input_schema")

        return requirement
