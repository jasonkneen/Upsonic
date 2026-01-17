"""
Test: Full Session Memory with Storage Providers

Success criteria:
- Agent remembers previous conversations across runs
- Messages are properly stored in session
- num_last_messages correctly limits chat history
- feed_tool_call_results correctly filters tool messages
- Session data persists and can be retrieved
"""

import pytest
import os
import tempfile
import uuid
from upsonic import Agent, Task
from upsonic.storage import Memory, SqliteStorage, InMemoryStorage
from upsonic.session.agent import AgentSession
from upsonic.session.base import SessionType
from upsonic.messages import ModelRequest, ModelResponse

pytestmark = pytest.mark.timeout(120)


@pytest.fixture
def test_user_id():
    return f"test_user_{uuid.uuid4().hex[:8]}"


@pytest.fixture
def test_session_id():
    return f"test_session_{uuid.uuid4().hex[:8]}"


@pytest.fixture
def sqlite_storage():
    """Create a temporary SQLite storage."""
    db_file = tempfile.mktemp(suffix=".db")
    storage = SqliteStorage(db_file=db_file)
    yield storage
    # Cleanup
    if os.path.exists(db_file):
        os.remove(db_file)


@pytest.fixture
def inmemory_storage():
    """Create an in-memory storage."""
    return InMemoryStorage()


# =============================================================================
# TEST: Basic Full Session Memory
# =============================================================================

@pytest.mark.asyncio
async def test_full_session_memory_basic(inmemory_storage, test_user_id, test_session_id):
    """Test that full session memory stores and retrieves conversation history."""
    memory = Memory(
        storage=inmemory_storage,
        session_id=test_session_id,
        user_id=test_user_id,
        full_session_memory=True,
        debug=False
    )
    
    agent = Agent(model="openai/gpt-4o-mini", memory=memory)
    
    # First conversation turn
    task1 = Task(description="My name is Alice and I love Python programming")
    result1 = await agent.do_async(task1)
    assert result1 is not None
    
    # Second conversation turn - should remember context
    task2 = Task(description="What is my name?")
    result2 = await agent.do_async(task2)
    assert "alice" in str(result2).lower(), f"Expected 'alice' in response, got: {result2}"
    
    # Verify session was stored with messages
    session = inmemory_storage.get_session(
        session_id=test_session_id,
        session_type=SessionType.AGENT,
        deserialize=True
    )
    assert session is not None, "Session should be stored"
    assert session.messages is not None, "Session should have messages"
    assert len(session.messages) >= 4, f"Should have at least 4 messages, got {len(session.messages)}"


@pytest.mark.asyncio
async def test_full_session_memory_message_accumulation(inmemory_storage, test_user_id, test_session_id):
    """Test that messages accumulate correctly across multiple runs."""
    memory = Memory(
        storage=inmemory_storage,
        session_id=test_session_id,
        user_id=test_user_id,
        full_session_memory=True,
        debug=False
    )
    
    agent = Agent(model="openai/gpt-4o-mini", memory=memory)
    message_counts = []
    
    # Run 3 tasks and track message accumulation
    for i in range(3):
        task = Task(description=f"This is message number {i+1}")
        await agent.do_async(task)
        
        session = inmemory_storage.get_session(
            session_id=test_session_id,
            session_type=SessionType.AGENT,
            deserialize=True
        )
        if session and session.messages:
            message_counts.append(len(session.messages))
        else:
            message_counts.append(0)
    
    # Verify message count increases with each run
    assert len(message_counts) == 3
    assert message_counts[1] > message_counts[0], \
        f"Message count should increase. Run 1: {message_counts[0]}, Run 2: {message_counts[1]}"
    assert message_counts[2] > message_counts[1], \
        f"Message count should increase. Run 2: {message_counts[1]}, Run 3: {message_counts[2]}"


# =============================================================================
# TEST: num_last_messages Limiting
# =============================================================================

@pytest.mark.asyncio
async def test_num_last_messages_limiting(inmemory_storage, test_user_id, test_session_id):
    """Test that num_last_messages correctly limits chat history."""
    memory = Memory(
        storage=inmemory_storage,
        session_id=test_session_id,
        user_id=test_user_id,
        full_session_memory=True,
        num_last_messages=2,  # Only keep last 2 message turns
        debug=False
    )
    
    agent = Agent(model="openai/gpt-4o-mini", memory=memory)
    
    # Run 5 conversations
    for i in range(5):
        task = Task(description=f"Message {i}: I like number {i}")
        await agent.do_async(task)
    
    # Verify messages are stored in session (raw storage)
    session = inmemory_storage.get_session(
        session_id=test_session_id,
        session_type=SessionType.AGENT,
        deserialize=True
    )
    assert session is not None
    assert session.messages is not None
    # Storage should have ALL messages
    assert len(session.messages) >= 10, f"Storage should have all messages, got {len(session.messages)}"
    
    # Verify prepare_inputs_for_task applies the limit
    prepared = await memory.prepare_inputs_for_task(session_type=SessionType.AGENT)
    limited_messages = prepared["message_history"]
    
    # With num_last_messages=2, we should get at most 4 messages (2 runs * 2 messages per run)
    assert len(limited_messages) <= 4, \
        f"Limited history should have at most 4 messages, got {len(limited_messages)}"


@pytest.mark.asyncio
async def test_num_last_messages_preserves_context(inmemory_storage, test_user_id, test_session_id):
    """Test that limiting still preserves necessary context like system prompt."""
    memory = Memory(
        storage=inmemory_storage,
        session_id=test_session_id,
        user_id=test_user_id,
        full_session_memory=True,
        num_last_messages=2,
        debug=False
    )
    
    agent = Agent(model="openai/gpt-4o-mini", memory=memory)
    
    # First message
    task1 = Task(description="My favorite color is blue")
    await agent.do_async(task1)
    
    # Second message
    task2 = Task(description="My favorite food is pizza")
    await agent.do_async(task2)
    
    # Third message
    task3 = Task(description="My favorite sport is tennis")
    await agent.do_async(task3)
    
    # Ask about old message - should NOT remember (outside window)
    task4 = Task(description="What is my favorite color? Just say the color or 'I don't know'.")
    result4 = await agent.do_async(task4)
    
    # Due to num_last_messages=2, "blue" should be outside the context window
    result4_str = str(result4).lower()
    # Either agent says "I don't know" or doesn't mention "blue"
    assert "don't know" in result4_str or "do not know" in result4_str or "blue" not in result4_str, \
        f"num_last_messages should limit context, got: {result4}"


# =============================================================================
# TEST: feed_tool_call_results Filtering
# =============================================================================

@pytest.mark.asyncio
async def test_feed_tool_call_results_filtering(inmemory_storage, test_user_id, test_session_id):
    """Test that feed_tool_call_results=False filters out tool messages."""
    from upsonic.tools import tool
    
    @tool
    def get_weather(location: str) -> str:
        """Get weather for a location."""
        return f"Weather in {location}: Sunny, 72°F"
    
    memory = Memory(
        storage=inmemory_storage,
        session_id=test_session_id,
        user_id=test_user_id,
        full_session_memory=True,
        feed_tool_call_results=False,
        debug=False
    )
    
    agent = Agent(
        model="openai/gpt-4o-mini",
        memory=memory,
        tools=[get_weather]
    )
    
    # Make a task that triggers tool call
    task = Task(description="What's the weather in New York? Use the get_weather tool.")
    await agent.do_async(task)
    
    # Check that prepare_inputs_for_task filters tool messages
    prepared = await memory.prepare_inputs_for_task(session_type=SessionType.AGENT)
    message_history = prepared["message_history"]
    
    # Count tool messages
    tool_count = 0
    for msg in message_history:
        if isinstance(msg, ModelRequest):
            if hasattr(msg, 'parts'):
                for part in msg.parts:
                    if hasattr(part, 'part_kind') and part.part_kind == 'tool-return':
                        tool_count += 1
                        break
        elif isinstance(msg, ModelResponse):
            if hasattr(msg, 'parts'):
                for part in msg.parts:
                    if hasattr(part, 'part_kind') and part.part_kind == 'tool-call':
                        tool_count += 1
                        break
    
    assert tool_count == 0, f"Tool messages should be filtered when feed_tool_call_results=False, found {tool_count}"


@pytest.mark.asyncio
async def test_feed_tool_call_results_included(inmemory_storage, test_user_id, test_session_id):
    """Test that feed_tool_call_results=True includes tool messages."""
    from upsonic.tools import tool
    
    @tool
    def get_weather(location: str) -> str:
        """Get weather for a location."""
        return f"Weather in {location}: Sunny, 72°F"
    
    memory = Memory(
        storage=inmemory_storage,
        session_id=test_session_id,
        user_id=test_user_id,
        full_session_memory=True,
        feed_tool_call_results=True,  # Include tool messages
        debug=False
    )
    
    agent = Agent(
        model="openai/gpt-4o-mini",
        memory=memory,
        tools=[get_weather]
    )
    
    # Make a task that triggers tool call
    task = Task(description="What's the weather in London? Use the get_weather tool.")
    await agent.do_async(task)
    
    # Check that prepare_inputs_for_task includes tool messages
    prepared = await memory.prepare_inputs_for_task(session_type=SessionType.AGENT)
    message_history = prepared["message_history"]
    
    # Count tool messages
    tool_count = 0
    for msg in message_history:
        if isinstance(msg, ModelRequest):
            if hasattr(msg, 'parts'):
                for part in msg.parts:
                    if hasattr(part, 'part_kind') and part.part_kind == 'tool-return':
                        tool_count += 1
                        break
        elif isinstance(msg, ModelResponse):
            if hasattr(msg, 'parts'):
                for part in msg.parts:
                    if hasattr(part, 'part_kind') and part.part_kind == 'tool-call':
                        tool_count += 1
                        break
    
    assert tool_count > 0, f"Tool messages should be included when feed_tool_call_results=True, found {tool_count}"


# =============================================================================
# TEST: Session Persistence with SQLite
# =============================================================================

@pytest.mark.asyncio
async def test_session_persistence_sqlite(sqlite_storage, test_user_id, test_session_id):
    """Test that sessions persist across Memory instances with SQLite."""
    # First instance - create and populate session
    memory1 = Memory(
        storage=sqlite_storage,
        session_id=test_session_id,
        user_id=test_user_id,
        full_session_memory=True,
        debug=False
    )
    
    agent1 = Agent(model="openai/gpt-4o-mini", memory=memory1)
    
    task1 = Task(description="Remember that my secret code is DELTA-789.")
    await agent1.do_async(task1)
    
    # Create second Memory instance with same session_id
    memory2 = Memory(
        storage=sqlite_storage,
        session_id=test_session_id,  # Same session ID
        user_id=test_user_id,
        full_session_memory=True,
        debug=False
    )
    
    agent2 = Agent(model="openai/gpt-4o-mini", memory=memory2)
    
    # Should remember the secret code
    task2 = Task(description="What is my secret code?")
    result2 = await agent2.do_async(task2)
    
    result2_str = str(result2).upper()
    code_remembered = "DELTA" in result2_str or "789" in result2_str
    assert code_remembered, f"Session should persist - 'DELTA' or '789' not found in: {result2}"


# =============================================================================
# TEST: Run Output Attributes
# =============================================================================

@pytest.mark.asyncio
async def test_run_output_attributes_stored(inmemory_storage, test_user_id, test_session_id):
    """Test that AgentRunOutput attributes are stored correctly."""
    memory = Memory(
        storage=inmemory_storage,
        session_id=test_session_id,
        user_id=test_user_id,
        full_session_memory=True,
        debug=False
    )
    
    agent = Agent(model="openai/gpt-4o-mini", memory=memory)
    
    # Run a task
    task = Task(description="Say hello")
    result = await agent.do_async(task)
    assert result is not None
    
    # Check session for stored run
    session = inmemory_storage.get_session(
        session_id=test_session_id,
        session_type=SessionType.AGENT,
        deserialize=True
    )
    assert session is not None
    assert session.runs is not None
    assert len(session.runs) >= 1
    
    # Get the latest run
    latest_run_data = list(session.runs.values())[-1]
    
    if hasattr(latest_run_data, 'output') and latest_run_data.output:
        run_output = latest_run_data.output
        
        from upsonic.run.agent.output import AgentRunOutput
        if isinstance(run_output, AgentRunOutput):
            # Verify run has messages
            if run_output.messages:
                assert isinstance(run_output.messages, list), "Run messages should be a list"
                assert len(run_output.messages) >= 2, \
                    f"Expected at least 2 messages in run, got {len(run_output.messages)}"


# =============================================================================
# TEST: CRUD Operations on Session
# =============================================================================

@pytest.mark.asyncio
async def test_session_crud_operations(inmemory_storage, test_user_id, test_session_id):
    """Test CRUD operations on session."""
    memory = Memory(
        storage=inmemory_storage,
        session_id=test_session_id,
        user_id=test_user_id,
        full_session_memory=True,
        debug=False
    )
    
    agent = Agent(model="openai/gpt-4o-mini", memory=memory)
    
    # CREATE - Run a task to create session
    task = Task(description="Hello world")
    await agent.do_async(task)
    
    # READ - Get session
    session = await memory.get_session_async()
    assert session is not None, "Session should exist after task"
    
    # UPDATE - Set metadata
    await memory.set_metadata_async({"test_key": "test_value"})
    metadata = await memory.get_metadata_async()
    assert metadata is not None
    assert metadata.get("test_key") == "test_value"
    
    # DELETE - Delete session
    deleted = await memory.delete_session_async()
    assert deleted is True
    
    # Verify deleted
    session_after = await memory.get_session_async()
    assert session_after is None, "Session should be None after delete"
