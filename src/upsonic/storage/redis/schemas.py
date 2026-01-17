"""Schema definitions for Redis storage tables.

This module defines the structure of tables stored in Redis.
Each schema describes the fields, their types, and constraints.

SESSION_SCHEMA: Schema for storing agent/team/workflow sessions.
USER_MEMORY_SCHEMA: Schema for storing user memory/profile data.
"""
from typing import Any, Dict, Final

# Session table schema - stores AgentSession, TeamSession, WorkflowSession
SESSION_SCHEMA: Final[Dict[str, Dict[str, Any]]] = {
    "session_id": {"type": "string", "primary_key": True},
    "session_type": {"type": "string"},  # "agent", "team", or "workflow"
    "agent_id": {"type": "string"},
    "team_id": {"type": "string"},
    "workflow_id": {"type": "string"},
    "user_id": {"type": "string"},
    "session_data": {"type": "json"},
    "agent_data": {"type": "json"},
    "team_data": {"type": "json"},
    "workflow_data": {"type": "json"},
    "metadata": {"type": "json"},
    "runs": {"type": "json"},
    "messages": {"type": "json"},
    "summary": {"type": "string"},
    "usage": {"type": "json"},
    "created_at": {"type": "integer"},
    "updated_at": {"type": "integer"},
}

# User memory table schema - stores user profile/memory data
USER_MEMORY_SCHEMA: Final[Dict[str, Dict[str, Any]]] = {
    "user_id": {"type": "string", "primary_key": True},
    "user_memory": {"type": "json"},
    "agent_id": {"type": "string"},
    "team_id": {"type": "string"},
    "created_at": {"type": "integer"},
    "updated_at": {"type": "integer"},
}

# Index fields for faster lookups
SESSION_INDEX_FIELDS: Final[list[str]] = [
    "user_id",
    "agent_id", 
    "team_id",
    "workflow_id",
    "session_type",
]

USER_MEMORY_INDEX_FIELDS: Final[list[str]] = [
    "user_id",
    "agent_id",
    "team_id",
]

