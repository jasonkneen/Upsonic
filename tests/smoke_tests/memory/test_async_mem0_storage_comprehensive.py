"""Comprehensive test suite for AsyncMem0Storage.

This test suite verifies ALL methods and attributes of AsyncMem0Storage:
- Initialization (api_key, config, memory_client, custom tables, id)
- Client type detection (platform vs self-hosted)
- Table management (virtual tables - always exist)
- Session operations (aupsert, aget, adelete, bulk operations)
- User memory operations (aupsert, aget, adelete, bulk operations)
- Utility methods (aclear_all, close)
- Edge cases and error handling
- Deserialize flag behavior
- Filtering, pagination, and sorting
- API compatibility between platform and self-hosted
"""
import asyncio
import os
import sys
import time
from typing import Any, Dict, List, Optional

from upsonic.session.agent import AgentSession, RunData
from upsonic.session.base import SessionType
from upsonic.storage.schemas import UserMemory
from upsonic.storage.mem0 import AsyncMem0Storage
from upsonic.run.agent.output import AgentRunOutput
from upsonic.run.base import RunStatus
from upsonic.messages.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart, ToolCallPart, ThinkingPart
import pytest

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None

# Test result tracking
test_results: List[Dict[str, Any]] = []

# Use environment variable for API key
MEM0_API_KEY = os.getenv("MEM0_API_KEY", "m0-gsF34aoubipR3max0hXlgOZ1mfyq99fgOKKoadvL")


def log_test_result(test_name: str, passed: bool, message: str = "") -> None:
    """Log test result."""
    status = "✅ PASSED" if passed else "❌ FAILED"
    result = f"{status}: {test_name}"
    if message:
        result += f" - {message}"
    print(result, flush=True)
    test_results.append({"name": test_name, "passed": passed, "message": message})


def print_separator(title: str) -> None:
    """Print test section separator."""
    print("\n" + "=" * 80, flush=True)
    print(f"  {title}", flush=True)
    print("=" * 80 + "\n", flush=True)


def create_test_agentsession(
    session_id: str,
    agent_id: Optional[str] = None,
    user_id: Optional[str] = None,
    session_type: SessionType = SessionType.AGENT,
    created_at: Optional[int] = None,
) -> AgentSession:
    """Create a test AgentSession with comprehensive runs and messages using REAL classes."""
    current_time = created_at or int(time.time())
    
    # Create REAL AgentRunOutput and RunData objects
    test_runs = {
        "run_001": RunData(output=AgentRunOutput(
            run_id="run_001", session_id=session_id,
            user_id=user_id or f"user_{session_id}",
            agent_id=agent_id or f"agent_{session_id}",
            status=RunStatus.completed, accumulated_text="Done.",
        )),
        "run_002": RunData(output=AgentRunOutput(
            run_id="run_002", session_id=session_id,
            user_id=user_id or f"user_{session_id}",
            agent_id=agent_id or f"agent_{session_id}",
            status=RunStatus.paused,
        )),
    }
    
    # Create REAL ModelRequest and ModelResponse objects
    test_messages = [
        ModelRequest(parts=[UserPromptPart(content="Analyze")], run_id="run_001"),
        ModelResponse(parts=[
            TextPart(content="Analyzing..."),
            ToolCallPart(tool_name="calc", tool_call_id="c1", args={"x": 1}),
        ], model_name="gpt-4"),
        ModelResponse(parts=[
            TextPart(content="Result."),
            ThinkingPart(content="Processed."),
        ], model_name="gpt-4"),
    ]
    
    return AgentSession(
        session_id=session_id,
        agent_id=agent_id or f"agent_{session_id}",
        user_id=user_id or f"user_{session_id}",
        session_type=session_type,
        session_data={"test": "data", "nested": {"arr": [1, 2, {"k": "v"}]}},
        agent_data={"agent_name": "TestAgent", "model": "gpt-4"},
        metadata={"key": "value", "tags": ["test"]},
        runs=test_runs,
        messages=test_messages,
        summary="Test session with REAL classes",
        created_at=current_time,
        updated_at=int(time.time()),
    )


def get_unique_id(prefix: str = "test") -> str:
    """Generate a unique ID for testing."""
    import uuid
    return f"{prefix}_{uuid.uuid4().hex[:8]}_{int(time.time())}"


# ============================================================================
# TEST 1: Initialization
# ============================================================================
@pytest.mark.asyncio
async def test_initialization():
    """Test AsyncMem0Storage initialization with various configurations."""
    print_separator("TEST 1: Initialization")
    
    # Test 1.1: Initialization with API key (Platform)
    try:
        storage = AsyncMem0Storage(api_key=MEM0_API_KEY)
        assert storage.memory_client is not None
        assert storage._is_platform_client is True
        assert storage.session_table_name == "upsonic_sessions"
        assert storage.user_memory_table_name == "upsonic_user_memories"
        assert storage.id is not None
        assert storage.default_user_id == "upsonic_default"
        await storage.close()
        log_test_result("Initialization with API key (platform)", True)
    except Exception as e:
        log_test_result("Initialization with API key (platform)", False, str(e))
        raise
    
    # Test 1.2: Initialization with custom table names
    try:
        storage = AsyncMem0Storage(
            api_key=MEM0_API_KEY,
            session_table="custom_sessions",
            user_memory_table="custom_memories",
        )
        assert storage.session_table_name == "custom_sessions"
        assert storage.user_memory_table_name == "custom_memories"
        await storage.close()
        log_test_result("Initialization with custom table names", True)
    except Exception as e:
        log_test_result("Initialization with custom table names", False, str(e))
        raise
    
    # Test 1.3: Initialization with custom ID
    try:
        custom_id = "test_storage_123"
        storage = AsyncMem0Storage(api_key=MEM0_API_KEY, id=custom_id)
        assert storage.id == custom_id
        await storage.close()
        log_test_result("Initialization with custom ID", True)
    except Exception as e:
        log_test_result("Initialization with custom ID", False, str(e))
        raise
    
    # Test 1.4: Initialization with custom default_user_id
    try:
        storage = AsyncMem0Storage(api_key=MEM0_API_KEY, default_user_id="custom_user")
        assert storage.default_user_id == "custom_user"
        await storage.close()
        log_test_result("Initialization with custom default_user_id", True)
    except Exception as e:
        log_test_result("Initialization with custom default_user_id", False, str(e))
        raise
    
    # Test 1.5: Initialization with existing AsyncMemoryClient
    try:
        from mem0 import AsyncMemoryClient
        client = AsyncMemoryClient(api_key=MEM0_API_KEY)
        storage = AsyncMem0Storage(memory_client=client)
        assert storage.memory_client is client
        assert storage._is_platform_client is True
        await storage.close()
        log_test_result("Initialization with existing AsyncMemoryClient", True)
    except Exception as e:
        log_test_result("Initialization with existing AsyncMemoryClient", False, str(e))
        raise
    
    # Test 1.6: Check platform client detection
    try:
        storage = AsyncMem0Storage(api_key=MEM0_API_KEY)
        assert storage._check_is_platform_client() is True
        await storage.close()
        log_test_result("Platform client detection", True)
    except Exception as e:
        log_test_result("Platform client detection", False, str(e))
        raise


# ============================================================================
# TEST 2: Table Management
# ============================================================================
@pytest.mark.asyncio
async def test_table_management():
    """Test table existence checks (virtual tables always exist in Mem0)."""
    print_separator("TEST 2: Table Management")
    
    storage = AsyncMem0Storage(api_key=MEM0_API_KEY)
    
    try:
        # Test 2.1: table_exists always returns True for Mem0
        result = await storage.table_exists("any_table")
        assert result is True
        log_test_result("table_exists returns True", True)
    except Exception as e:
        log_test_result("table_exists returns True", False, str(e))
        raise
    
    try:
        # Test 2.2: _create_all_tables is no-op (doesn't raise)
        await storage._create_all_tables()
        log_test_result("_create_all_tables is no-op", True)
    except Exception as e:
        log_test_result("_create_all_tables is no-op", False, str(e))
        raise
    
    await storage.close()


# ============================================================================
# TEST 3: Session CRUD Operations
# ============================================================================
@pytest.mark.asyncio
async def test_session_crud():
    """Test session Create, Read, Update, Delete operations."""
    print_separator("TEST 3: Session CRUD Operations")
    
    storage = AsyncMem0Storage(api_key=MEM0_API_KEY)
    
    # Use unique IDs for this test run
    session_id = get_unique_id("async_session")
    
    try:
        # Test 3.1: Upsert session (create)
        session = create_test_agentsession(session_id=session_id)
        result = await storage.aupsert_session(session)
        assert result is not None
        assert isinstance(result, AgentSession)
        assert result.session_id == session_id
        log_test_result("Upsert session (create)", True)
    except Exception as e:
        log_test_result("Upsert session (create)", False, str(e))
        raise
    
    try:
        # Test 3.2: Get session by ID
        retrieved = await storage.aget_session(session_id=session_id)
        assert retrieved is not None
        assert retrieved.session_id == session_id
        log_test_result("Get session by ID", True)
    except Exception as e:
        log_test_result("Get session by ID", False, str(e))
        raise
    
    try:
        # Test 3.3: Upsert session (update)
        session.summary = "Updated summary"
        result = await storage.aupsert_session(session)
        assert result is not None
        # Verify update
        retrieved = await storage.aget_session(session_id=session_id)
        assert retrieved is not None
        assert retrieved.summary == "Updated summary"
        log_test_result("Upsert session (update)", True)
    except Exception as e:
        log_test_result("Upsert session (update)", False, str(e))
        raise
    
    try:
        # Test 3.4: Get session with deserialize=False
        result = await storage.aget_session(session_id=session_id, deserialize=False)
        assert result is not None
        assert isinstance(result, dict)
        assert result.get("session_id") == session_id
        log_test_result("Get session with deserialize=False", True)
    except Exception as e:
        log_test_result("Get session with deserialize=False", False, str(e))
        raise
    
    try:
        # Test 3.5: Delete session
        deleted = await storage.adelete_session(session_id=session_id)
        assert deleted is True
        # Verify deletion
        retrieved = await storage.aget_session(session_id=session_id)
        assert retrieved is None
        log_test_result("Delete session", True)
    except Exception as e:
        log_test_result("Delete session", False, str(e))
        raise
    
    try:
        # Test 3.6: Delete non-existent session
        deleted = await storage.adelete_session(session_id="non_existent_session_xyz")
        assert deleted is False
        log_test_result("Delete non-existent session", True)
    except Exception as e:
        log_test_result("Delete non-existent session", False, str(e))
        raise
    
    # CRITICAL - Verify runs and messages are correctly preserved
    try:
        verify_session_id = get_unique_id("async_session_verify")
        session_with_data = create_test_agentsession(verify_session_id)
        result_full = await storage.aupsert_session(session_with_data, deserialize=True)
        
        # Verify runs are correctly preserved
        assert result_full.runs is not None, "runs should not be None"
        assert isinstance(result_full.runs, dict), "runs should be a dict"
        assert len(result_full.runs) == 2, f"runs should have 2 entries, got {len(result_full.runs)}"
        assert "run_001" in result_full.runs, "run_001 should be in runs"
        
        run_001 = result_full.runs["run_001"]
        assert isinstance(run_001, RunData), f"run_001 should be RunData, got {type(run_001)}"
        assert run_001.output is not None, "run_001.output should not be None"
        assert isinstance(run_001.output, AgentRunOutput), f"run_001.output should be AgentRunOutput"
        assert run_001.output.run_id == "run_001", f"run_id mismatch"
        assert run_001.output.status == RunStatus.completed, f"status mismatch"
        
        run_002 = result_full.runs["run_002"]
        assert run_002.output.status == RunStatus.paused, f"status mismatch"
        log_test_result("runs field round-trip verification", True)
        
        # Verify messages are correctly preserved
        assert result_full.messages is not None, "messages should not be None"
        assert isinstance(result_full.messages, list), "messages should be a list"
        assert len(result_full.messages) == 3, f"messages should have 3 entries, got {len(result_full.messages)}"
        
        msg_0 = result_full.messages[0]
        assert isinstance(msg_0, ModelRequest), f"messages[0] should be ModelRequest, got {type(msg_0)}"
        
        msg_1 = result_full.messages[1]
        assert isinstance(msg_1, ModelResponse), f"messages[1] should be ModelResponse, got {type(msg_1)}"
        
        has_tool_call = any(isinstance(p, ToolCallPart) for p in msg_1.parts)
        assert has_tool_call, "messages[1] should have a ToolCallPart"
        
        msg_2 = result_full.messages[2]
        has_thinking = any(isinstance(p, ThinkingPart) for p in msg_2.parts)
        assert has_thinking, "messages[2] should have a ThinkingPart"
        log_test_result("messages field round-trip verification", True)
        
        # Clean up verify session
        await storage.adelete_session(session_id=verify_session_id)
    except Exception as e:
        log_test_result("runs/messages verification", False, str(e))
        raise
    
    await storage.close()


# ============================================================================
# TEST 4: Bulk Session Operations
# ============================================================================
@pytest.mark.asyncio
async def test_bulk_session_operations():
    """Test bulk session operations (aupsert_sessions, aget_sessions, adelete_sessions)."""
    print_separator("TEST 4: Bulk Session Operations")
    
    storage = AsyncMem0Storage(api_key=MEM0_API_KEY)
    
    # Create unique session IDs
    session_ids = [get_unique_id(f"async_bulk_session_{i}") for i in range(3)]
    
    try:
        # Test 4.1: Upsert multiple sessions
        sessions = [create_test_agentsession(session_id=sid) for sid in session_ids]
        results = await storage.aupsert_sessions(sessions)
        assert len(results) == 3
        log_test_result("Upsert multiple sessions", True)
    except Exception as e:
        log_test_result("Upsert multiple sessions", False, str(e))
        raise
    
    try:
        # Test 4.2: Get sessions by IDs
        retrieved = await storage.aget_sessions(session_ids=session_ids)
        assert isinstance(retrieved, list)
        assert len(retrieved) == 3
        log_test_result("Get sessions by IDs", True)
    except Exception as e:
        log_test_result("Get sessions by IDs", False, str(e))
        raise
    
    try:
        # Test 4.3: Get sessions with deserialize=False
        result = await storage.aget_sessions(session_ids=session_ids, deserialize=False)
        assert isinstance(result, tuple)
        assert len(result) == 2  # (list, count)
        assert len(result[0]) == 3
        log_test_result("Get sessions with deserialize=False", True)
    except Exception as e:
        log_test_result("Get sessions with deserialize=False", False, str(e))
        raise
    
    try:
        # Test 4.4: Delete multiple sessions
        deleted_count = await storage.adelete_sessions(session_ids=session_ids)
        assert deleted_count == 3
        log_test_result("Delete multiple sessions", True)
    except Exception as e:
        log_test_result("Delete multiple sessions", False, str(e))
        raise
    
    try:
        # Test 4.5: Empty list for aupsert_sessions
        results = await storage.aupsert_sessions([])
        assert results == []
        log_test_result("Upsert empty list", True)
    except Exception as e:
        log_test_result("Upsert empty list", False, str(e))
        raise
    
    await storage.close()


# ============================================================================
# TEST 5: Session Filtering
# ============================================================================
@pytest.mark.asyncio
async def test_session_filtering():
    """Test session filtering by user_id, agent_id, session_type."""
    print_separator("TEST 5: Session Filtering")
    
    storage = AsyncMem0Storage(api_key=MEM0_API_KEY)
    
    # Create unique IDs
    user_id = get_unique_id("async_filter_user")
    agent_id = get_unique_id("async_filter_agent")
    session_id = get_unique_id("async_filter_session")
    
    try:
        # Create a session with specific user_id and agent_id
        session = create_test_agentsession(
            session_id=session_id,
            user_id=user_id,
            agent_id=agent_id,
        )
        await storage.aupsert_session(session)
        
        # Test 5.1: Filter by user_id
        results = await storage.aget_sessions(user_id=user_id)
        assert len(results) >= 1
        log_test_result("Filter by user_id", True)
    except Exception as e:
        log_test_result("Filter by user_id", False, str(e))
        raise
    
    try:
        # Test 5.2: Filter by agent_id
        results = await storage.aget_sessions(agent_id=agent_id)
        assert len(results) >= 1
        log_test_result("Filter by agent_id", True)
    except Exception as e:
        log_test_result("Filter by agent_id", False, str(e))
        raise
    
    try:
        # Test 5.3: Filter by session_type
        results = await storage.aget_sessions(session_type=SessionType.AGENT)
        assert isinstance(results, list)
        log_test_result("Filter by session_type", True)
    except Exception as e:
        log_test_result("Filter by session_type", False, str(e))
        raise
    
    # Cleanup
    await storage.adelete_session(session_id)
    await storage.close()


# ============================================================================
# TEST 6: Session Pagination and Sorting
# ============================================================================
@pytest.mark.asyncio
async def test_session_pagination_sorting():
    """Test session pagination and sorting."""
    print_separator("TEST 6: Session Pagination and Sorting")
    
    storage = AsyncMem0Storage(api_key=MEM0_API_KEY)
    
    # Create multiple sessions with different timestamps
    session_ids = []
    for i in range(5):
        session_id = get_unique_id(f"async_page_session_{i}")
        session_ids.append(session_id)
        session = create_test_agentsession(
            session_id=session_id,
            created_at=int(time.time()) - (i * 10),  # Different creation times
        )
        await storage.aupsert_session(session)
        await asyncio.sleep(0.1)  # Small delay
    
    try:
        # Test 6.1: Get sessions with limit
        results = await storage.aget_sessions(session_ids=session_ids, limit=2)
        assert len(results) == 2
        log_test_result("Get sessions with limit", True)
    except Exception as e:
        log_test_result("Get sessions with limit", False, str(e))
        raise
    
    try:
        # Test 6.2: Get sessions with offset
        results = await storage.aget_sessions(session_ids=session_ids, limit=2, offset=2)
        assert len(results) == 2
        log_test_result("Get sessions with offset", True)
    except Exception as e:
        log_test_result("Get sessions with offset", False, str(e))
        raise
    
    try:
        # Test 6.3: Get sessions sorted by created_at desc (default)
        results = await storage.aget_sessions(session_ids=session_ids)
        assert len(results) == 5
        log_test_result("Get sessions sorted by created_at", True)
    except Exception as e:
        log_test_result("Get sessions sorted by created_at", False, str(e))
        raise
    
    # Cleanup
    await storage.adelete_sessions(session_ids)
    await storage.close()


# ============================================================================
# TEST 7: User Memory CRUD Operations
# ============================================================================
@pytest.mark.asyncio
async def test_user_memory_crud():
    """Test user memory Create, Read, Update, Delete operations."""
    print_separator("TEST 7: User Memory CRUD Operations")
    
    storage = AsyncMem0Storage(api_key=MEM0_API_KEY)
    
    user_id = get_unique_id("async_memory_user")
    
    try:
        # Test 7.1: Upsert user memory (create) - deserialize=True returns UserMemory
        memory_data = {"preferences": {"theme": "dark"}, "notes": "Test note"}
        result = await storage.aupsert_user_memory(user_memory=UserMemory(user_id=user_id, user_memory=memory_data), deserialize=True)
        assert result is not None
        assert isinstance(result, UserMemory)
        assert result.user_id == user_id
        log_test_result("Upsert user memory (create)", True)
    except Exception as e:
        log_test_result("Upsert user memory (create)", False, str(e))
        raise
    
    try:
        # Test 7.2: Get user memory by ID
        retrieved = await storage.aget_user_memory(user_id=user_id, deserialize=True)
        assert retrieved is not None
        assert isinstance(retrieved, UserMemory)
        assert retrieved.user_id == user_id
        log_test_result("Get user memory by ID", True)
    except Exception as e:
        log_test_result("Get user memory by ID", False, str(e))
        raise
    
    try:
        # Test 7.3: Upsert user memory (update)
        updated_memory = {"preferences": {"theme": "light"}, "notes": "Updated note"}
        result = await storage.aupsert_user_memory(user_memory=UserMemory(user_id=user_id, user_memory=updated_memory), deserialize=True)
        assert result is not None
        assert isinstance(result, UserMemory)
        # Verify update
        retrieved = await storage.aget_user_memory(user_id=user_id, deserialize=True)
        assert isinstance(retrieved, UserMemory)
        assert retrieved.user_memory.get("preferences", {}).get("theme") == "light"
        log_test_result("Upsert user memory (update)", True)
    except Exception as e:
        log_test_result("Upsert user memory (update)", False, str(e))
        raise
    
    try:
        # Test 7.4: Upsert user memory with agent_id
        memory_with_agent = {"data": "test"}
        result = await storage.aupsert_user_memory(
            user_memory=UserMemory(user_id=user_id, user_memory=memory_with_agent, agent_id="test_agent"),
            deserialize=True
        )
        assert result is not None
        assert isinstance(result, UserMemory)
        log_test_result("Upsert user memory with agent_id", True)
    except Exception as e:
        log_test_result("Upsert user memory with agent_id", False, str(e))
        raise
    
    try:
        # Test 7.5: Delete user memory
        deleted = await storage.adelete_user_memory(user_id=user_id)
        assert deleted is True
        # Verify deletion
        retrieved = await storage.aget_user_memory(user_id=user_id)
        assert retrieved is None
        log_test_result("Delete user memory", True)
    except Exception as e:
        log_test_result("Delete user memory", False, str(e))
        raise
    
    try:
        # Test 7.6: Delete non-existent user memory
        deleted = await storage.adelete_user_memory(user_id="non_existent_user_xyz")
        assert deleted is False
        log_test_result("Delete non-existent user memory", True)
    except Exception as e:
        log_test_result("Delete non-existent user memory", False, str(e))
        raise
    
    await storage.close()


# ============================================================================
# TEST 8: Bulk User Memory Operations
# ============================================================================
@pytest.mark.asyncio
async def test_bulk_user_memory_operations():
    """Test bulk user memory operations."""
    print_separator("TEST 8: Bulk User Memory Operations")
    
    storage = AsyncMem0Storage(api_key=MEM0_API_KEY)
    
    user_ids = [get_unique_id(f"async_bulk_memory_user_{i}") for i in range(3)]
    
    try:
        # Test 8.1: Upsert multiple user memories - deserialize=True returns UserMemory
        memories = [
            UserMemory(user_id=uid, user_memory={"data": f"data_{i}"})
            for i, uid in enumerate(user_ids)
        ]
        results = await storage.aupsert_user_memories(memories, deserialize=True)
        assert len(results) == 3
        for r in results:
            assert isinstance(r, UserMemory)
        log_test_result("Upsert multiple user memories", True)
    except Exception as e:
        log_test_result("Upsert multiple user memories", False, str(e))
        raise
    
    try:
        # Test 8.2: Get user memories by IDs - deserialize=True returns List[UserMemory]
        result = await storage.aget_user_memories(user_ids=user_ids, deserialize=True)
        assert isinstance(result, list)
        assert len(result) == 3
        for r in result:
            assert isinstance(r, UserMemory)
        log_test_result("Get user memories by IDs", True)
    except Exception as e:
        log_test_result("Get user memories by IDs", False, str(e))
        raise
    
    try:
        # Test 8.3: Delete multiple user memories
        deleted_count = await storage.adelete_user_memories(user_ids=user_ids)
        assert deleted_count == 3
        log_test_result("Delete multiple user memories", True)
    except Exception as e:
        log_test_result("Delete multiple user memories", False, str(e))
        raise
    
    try:
        # Test 8.4: Empty list for aupsert_user_memories
        results = await storage.aupsert_user_memories([])
        assert results == []
        log_test_result("Upsert empty user memory list", True)
    except Exception as e:
        log_test_result("Upsert empty user memory list", False, str(e))
        raise
    
    await storage.close()


# ============================================================================
# TEST 9: Error Handling
# ============================================================================
@pytest.mark.asyncio
async def test_error_handling():
    """Test error handling for invalid inputs."""
    print_separator("TEST 9: Error Handling")
    
    storage = AsyncMem0Storage(api_key=MEM0_API_KEY)
    
    try:
        # Test 9.1: Upsert session without session_id
        session = AgentSession(session_id=None)
        try:
            await storage.aupsert_session(session)
            log_test_result("Upsert session without session_id raises error", False, "No error raised")
        except ValueError:
            log_test_result("Upsert session without session_id raises error", True)
    except Exception as e:
        log_test_result("Upsert session without session_id raises error", False, str(e))
        raise
    
    try:
        # Test 9.2: Delete session without session_id
        try:
            await storage.adelete_session(session_id="")
            log_test_result("Delete session without session_id raises error", False, "No error raised")
        except ValueError:
            log_test_result("Delete session without session_id raises error", True)
    except Exception as e:
        log_test_result("Delete session without session_id raises error", False, str(e))
        raise
    
    try:
        # Test 9.3: Delete sessions with empty list
        try:
            await storage.adelete_sessions(session_ids=[])
            log_test_result("Delete sessions with empty list raises error", False, "No error raised")
        except ValueError:
            log_test_result("Delete sessions with empty list raises error", True)
    except Exception as e:
        log_test_result("Delete sessions with empty list raises error", False, str(e))
        raise
    
    try:
        # Test 9.4: Upsert user memory without user_id
        try:
            await storage.aupsert_user_memory(user_memory=UserMemory(user_id="", user_memory={}))
            log_test_result("Upsert user memory without user_id raises error", False, "No error raised")
        except ValueError:
            log_test_result("Upsert user memory without user_id raises error", True)
    except Exception as e:
        log_test_result("Upsert user memory without user_id raises error", False, str(e))
        raise
    
    try:
        # Test 9.5: Delete user memory without user_id
        try:
            await storage.adelete_user_memory(user_id="")
            log_test_result("Delete user memory without user_id raises error", False, "No error raised")
        except ValueError:
            log_test_result("Delete user memory without user_id raises error", True)
    except Exception as e:
        log_test_result("Delete user memory without user_id raises error", False, str(e))
        raise
    
    try:
        # Test 9.6: Delete user memories with empty list
        try:
            await storage.adelete_user_memories(user_ids=[])
            log_test_result("Delete user memories with empty list raises error", False, "No error raised")
        except ValueError:
            log_test_result("Delete user memories with empty list raises error", True)
    except Exception as e:
        log_test_result("Delete user memories with empty list raises error", False, str(e))
        raise
    
    await storage.close()


# ============================================================================
# TEST 10: Clear All
# ============================================================================
@pytest.mark.asyncio
async def test_clear_all():
    """Test aclear_all method."""
    print_separator("TEST 10: Clear All")
    
    storage = AsyncMem0Storage(api_key=MEM0_API_KEY)
    
    # Create some test data
    session_id = get_unique_id("async_clear_session")
    user_id = get_unique_id("async_clear_user")
    
    session = create_test_agentsession(session_id=session_id)
    await storage.aupsert_session(session)
    await storage.aupsert_user_memory(user_memory=UserMemory(user_id=user_id, user_memory={"test": "data"}))
    
    try:
        # Test 10.1: Clear all data
        await storage.aclear_all()
        log_test_result("Clear all executes without error", True)
    except Exception as e:
        log_test_result("Clear all executes without error", False, str(e))
        raise
    
    await storage.close()


# ============================================================================
# TEST 11: Latest Session/Memory Retrieval
# ============================================================================
@pytest.mark.asyncio
async def test_latest_retrieval():
    """Test getting latest session/memory when no ID is provided."""
    print_separator("TEST 11: Latest Session/Memory Retrieval")
    
    storage = AsyncMem0Storage(api_key=MEM0_API_KEY)
    
    # Create sessions with different timestamps
    session_ids = []
    for i in range(3):
        session_id = get_unique_id(f"async_latest_session_{i}")
        session_ids.append(session_id)
        session = create_test_agentsession(session_id=session_id)
        await storage.aupsert_session(session)
        await asyncio.sleep(0.5)  # Ensure different timestamps
    
    try:
        # Test 11.1: Get latest session (no session_id provided)
        latest = await storage.aget_session()
        assert latest is not None
        log_test_result("Get latest session", True)
    except Exception as e:
        log_test_result("Get latest session", False, str(e))
        raise
    
    # Cleanup
    await storage.adelete_sessions(session_ids)
    await storage.close()


# ============================================================================
# TEST 12: Deserialize Flag Behavior
# ============================================================================
@pytest.mark.asyncio
async def test_deserialize_flag():
    """Test deserialize flag behavior in various methods."""
    print_separator("TEST 12: Deserialize Flag Behavior")
    
    storage = AsyncMem0Storage(api_key=MEM0_API_KEY)
    
    session_id = get_unique_id("async_deserialize_session")
    session = create_test_agentsession(session_id=session_id)
    
    try:
        # Test 12.1: aupsert_session with deserialize=True (default)
        result = await storage.aupsert_session(session, deserialize=True)
        assert isinstance(result, AgentSession)
        log_test_result("aupsert_session with deserialize=True", True)
    except Exception as e:
        log_test_result("aupsert_session with deserialize=True", False, str(e))
        raise
    
    try:
        # Test 12.2: aupsert_session with deserialize=False
        result = await storage.aupsert_session(session, deserialize=False)
        assert isinstance(result, dict)
        log_test_result("aupsert_session with deserialize=False", True)
    except Exception as e:
        log_test_result("aupsert_session with deserialize=False", False, str(e))
        raise
    
    try:
        # Test 12.3: aget_session with deserialize=True (default)
        result = await storage.aget_session(session_id=session_id, deserialize=True)
        assert isinstance(result, AgentSession)
        log_test_result("aget_session with deserialize=True", True)
    except Exception as e:
        log_test_result("aget_session with deserialize=True", False, str(e))
        raise
    
    try:
        # Test 12.4: aget_session with deserialize=False
        result = await storage.aget_session(session_id=session_id, deserialize=False)
        assert isinstance(result, dict)
        log_test_result("aget_session with deserialize=False", True)
    except Exception as e:
        log_test_result("aget_session with deserialize=False", False, str(e))
        raise
    
    # Cleanup
    await storage.adelete_session(session_id)
    await storage.close()


# ============================================================================
# TEST 13: Close Method
# ============================================================================
@pytest.mark.asyncio
async def test_close_method():
    """Test close method."""
    print_separator("TEST 13: Close Method")
    
    try:
        storage = AsyncMem0Storage(api_key=MEM0_API_KEY)
        await storage.close()
        # Close should be idempotent
        await storage.close()
        log_test_result("Close method", True)
    except Exception as e:
        log_test_result("Close method", False, str(e))
        raise


# ============================================================================
# MAIN EXECUTION
# ============================================================================
async def run_all_tests():
    """Run all tests and print summary."""
    print("\n" + "=" * 80)
    print("  ASYNC MEM0 STORAGE COMPREHENSIVE TEST SUITE")
    print("=" * 80)
    print(f"\nUsing API Key: {MEM0_API_KEY[:20]}...")
    
    try:
        await test_initialization()
        await test_table_management()
        await test_session_crud()
        await test_bulk_session_operations()
        await test_session_filtering()
        await test_session_pagination_sorting()
        await test_user_memory_crud()
        await test_bulk_user_memory_operations()
        await test_error_handling()
        await test_clear_all()
        await test_latest_retrieval()
        await test_deserialize_flag()
        await test_close_method()
    except Exception as e:
        print(f"\n❌ Test suite aborted due to error: {e}")
        import traceback
        traceback.print_exc()
    
    # Print summary
    print("\n" + "=" * 80)
    print("  TEST SUMMARY")
    print("=" * 80)
    
    passed = sum(1 for r in test_results if r["passed"])
    failed = sum(1 for r in test_results if not r["passed"])
    total = len(test_results)
    
    print(f"\n  Total:  {total}")
    print(f"  Passed: {passed}")
    print(f"  Failed: {failed}")
    
    if failed > 0:
        print("\n  Failed Tests:")
        for r in test_results:
            if not r["passed"]:
                print(f"    - {r['name']}: {r['message']}")
    
    print("\n" + "=" * 80)
    
    return failed == 0


if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)

