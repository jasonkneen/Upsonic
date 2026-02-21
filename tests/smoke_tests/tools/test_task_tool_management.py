"""
Test Task class tool management with separated ToolManagers.

Success criteria:
- Task has its own ToolManager (separate from Agent's)
- Task.get_tool_defs() returns task-level tools only
- Agent.get_tool_defs() returns agent-level tools only
- Tools are properly registered/removed via task.tool_manager
- tool_manager attribute lifecycle (None -> initialized via _setup_task_tools)
- Separation: no cross-contamination between agent and task tool managers
- Backward compat: task.remove_tools(tools, agent) still works
"""

import pytest
from upsonic import Agent, Task
from upsonic.tools import tool, ToolKit
from upsonic.tools.builtin_tools import WebSearchTool, CodeExecutionTool

pytestmark = pytest.mark.timeout(120)

MODEL = "anthropic/claude-sonnet-4-5"


@tool
def add_numbers(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b


@tool
def multiply_numbers(a: int, b: int) -> int:
    """Multiply two numbers together."""
    return a * b


@tool
def greet(name: str) -> str:
    """Greet someone by name."""
    return f"Hello, {name}!"


@tool
def subtract_numbers(a: int, b: int) -> int:
    """Subtract b from a."""
    return a - b


class MathToolKit(ToolKit):
    """A toolkit for mathematical operations."""
    
    @tool
    def divide(self, a: int, b: int) -> float:
        """Divide a by b."""
        if b == 0:
            raise ValueError("Cannot divide by zero")
        return a / b
    
    @tool
    def modulo(self, a: int, b: int) -> int:
        """Return remainder of a divided by b."""
        return a % b


# ============================================================
# Task ToolManager Attribute Tests
# ============================================================

@pytest.mark.asyncio
async def test_task_tool_manager_initially_none():
    """Task.tool_manager should be None before setup."""
    task = Task(description="test", tools=[add_numbers])
    assert task.tool_manager is None, "tool_manager should be None before _setup_task_tools"
    assert task._tool_manager is None, "_tool_manager should be None before _setup_task_tools"


@pytest.mark.asyncio
async def test_task_get_tool_defs_before_setup():
    """Task.get_tool_defs() should return empty list before ToolManager is initialized."""
    task = Task(description="test", tools=[add_numbers])
    defs = task.get_tool_defs()
    assert isinstance(defs, list), "get_tool_defs should return a list"
    assert len(defs) == 0, "get_tool_defs should return empty list before setup"


@pytest.mark.asyncio
async def test_task_ensure_tool_manager():
    """_ensure_tool_manager should lazily create the ToolManager."""
    task = Task(description="test")
    assert task.tool_manager is None

    from upsonic.tools import ToolManager
    tm = task._ensure_tool_manager()
    assert tm is not None, "_ensure_tool_manager should return a ToolManager"
    assert isinstance(tm, ToolManager), "Should be a ToolManager instance"
    assert task.tool_manager is tm, "tool_manager property should match"

    tm2 = task._ensure_tool_manager()
    assert tm2 is tm, "Calling _ensure_tool_manager again should return the same instance"


@pytest.mark.asyncio
async def test_task_tool_manager_setter():
    """Task.tool_manager setter should work."""
    task = Task(description="test")
    from upsonic.tools import ToolManager
    tm = ToolManager()
    task.tool_manager = tm
    assert task.tool_manager is tm
    assert task._tool_manager is tm

    task.tool_manager = None
    assert task.tool_manager is None


# ============================================================
# Task Tool Registration via _setup_task_tools
# ============================================================

@pytest.mark.asyncio
async def test_setup_task_tools_creates_tool_manager():
    """_setup_task_tools should create a ToolManager on the task."""
    agent = Agent(model=MODEL, name="Test Agent")
    task = Task(description="test", tools=[add_numbers])

    assert task.tool_manager is None
    agent._setup_task_tools(task)
    assert task.tool_manager is not None, "tool_manager should be created after _setup_task_tools"


@pytest.mark.asyncio
async def test_setup_task_tools_registers_function_tools():
    """Function tools should be registered in task.tool_manager after setup."""
    agent = Agent(model=MODEL, name="Test Agent")
    task = Task(description="test", tools=[add_numbers, multiply_numbers])

    agent._setup_task_tools(task)

    assert "add_numbers" in task.registered_task_tools
    assert "multiply_numbers" in task.registered_task_tools
    assert len(task.registered_task_tools) == 2

    defs = task.get_tool_defs()
    def_names = [d.name for d in defs]
    assert "add_numbers" in def_names
    assert "multiply_numbers" in def_names


@pytest.mark.asyncio
async def test_setup_task_tools_registers_toolkit():
    """ToolKit tools should be registered in task.tool_manager after setup."""
    agent = Agent(model=MODEL, name="Test Agent")
    math_kit = MathToolKit()
    task = Task(description="test", tools=[math_kit])

    agent._setup_task_tools(task)

    assert "divide" in task.registered_task_tools
    assert "modulo" in task.registered_task_tools

    defs = task.get_tool_defs()
    def_names = [d.name for d in defs]
    assert "divide" in def_names
    assert "modulo" in def_names


@pytest.mark.asyncio
async def test_setup_task_tools_separates_builtin_tools():
    """Builtin tools should go to task_builtin_tools, not registered_task_tools."""
    agent = Agent(model=MODEL, name="Test Agent")
    web_search = WebSearchTool()
    code_exec = CodeExecutionTool()

    task = Task(description="test", tools=[add_numbers, web_search, code_exec])

    agent._setup_task_tools(task)

    assert len(task.task_builtin_tools) == 2
    builtin_ids = {t.unique_id for t in task.task_builtin_tools}
    assert "web_search" in builtin_ids
    assert "code_execution" in builtin_ids

    assert "add_numbers" in task.registered_task_tools
    assert len(task.registered_task_tools) == 1


# ============================================================
# Task get_tool_defs
# ============================================================

@pytest.mark.asyncio
async def test_task_get_tool_defs_returns_only_task_tools():
    """get_tool_defs should return only tools from the task's ToolManager."""
    agent = Agent(model=MODEL, name="Test Agent", tools=[greet])
    task = Task(description="test", tools=[add_numbers, multiply_numbers])

    agent._setup_task_tools(task)

    agent_defs = agent.get_tool_defs()
    agent_names = {d.name for d in agent_defs}
    assert "greet" in agent_names
    assert "add_numbers" not in agent_names
    assert "multiply_numbers" not in agent_names

    task_defs = task.get_tool_defs()
    task_names = {d.name for d in task_defs}
    assert "add_numbers" in task_names
    assert "multiply_numbers" in task_names
    assert "greet" not in task_names


@pytest.mark.asyncio
async def test_task_get_tool_defs_has_correct_schema():
    """Tool definitions should have name, description, and parameter schema."""
    agent = Agent(model=MODEL, name="Test Agent")
    task = Task(description="test", tools=[add_numbers])

    agent._setup_task_tools(task)

    defs = task.get_tool_defs()
    assert len(defs) == 1
    td = defs[0]
    assert td.name == "add_numbers"
    assert td.description is not None and len(td.description) > 0
    assert td.parameters_json_schema is not None
    assert "properties" in td.parameters_json_schema


# ============================================================
# Separation: Agent vs Task ToolManager
# ============================================================

@pytest.mark.asyncio
async def test_agent_and_task_tool_managers_are_separate():
    """Agent and Task should have different ToolManager instances."""
    agent = Agent(model=MODEL, name="Test Agent", tools=[greet])
    task = Task(description="test", tools=[add_numbers])

    agent._setup_task_tools(task)

    assert agent.tool_manager is not task.tool_manager, \
        "Agent and Task should have separate ToolManager instances"


@pytest.mark.asyncio
async def test_no_cross_contamination_agent_to_task():
    """Agent tools should never appear in task's ToolManager."""
    agent = Agent(model=MODEL, name="Test Agent", tools=[greet, subtract_numbers])
    task = Task(description="test", tools=[add_numbers])

    agent._setup_task_tools(task)

    task_tool_names = {d.name for d in task.get_tool_defs()}
    assert "greet" not in task_tool_names
    assert "subtract_numbers" not in task_tool_names
    assert "add_numbers" in task_tool_names


@pytest.mark.asyncio
async def test_no_cross_contamination_task_to_agent():
    """Task tools should never appear in agent's ToolManager."""
    agent = Agent(model=MODEL, name="Test Agent", tools=[greet])
    task = Task(description="test", tools=[add_numbers, multiply_numbers])

    agent._setup_task_tools(task)

    agent_tool_names = {d.name for d in agent.get_tool_defs()}
    assert "add_numbers" not in agent_tool_names
    assert "multiply_numbers" not in agent_tool_names
    assert "greet" in agent_tool_names


@pytest.mark.asyncio
async def test_combined_tool_definitions():
    """_get_combined_tool_definitions should return both agent and task tools."""
    agent = Agent(model=MODEL, name="Test Agent", tools=[greet])
    task = Task(description="test", tools=[add_numbers])

    agent.current_task = task
    agent._setup_task_tools(task)

    combined = agent._get_combined_tool_definitions()
    combined_names = {d.name for d in combined}
    assert "greet" in combined_names
    assert "add_numbers" in combined_names


@pytest.mark.asyncio
async def test_resolve_tool_manager_routing():
    """_resolve_tool_manager should route to the correct ToolManager."""
    agent = Agent(model=MODEL, name="Test Agent", tools=[greet])
    task = Task(description="test", tools=[add_numbers])

    agent.current_task = task
    agent._setup_task_tools(task)

    greet_mgr = agent._resolve_tool_manager("greet")
    assert greet_mgr is agent.tool_manager

    add_mgr = agent._resolve_tool_manager("add_numbers")
    assert add_mgr is task.tool_manager

    with pytest.raises(ValueError, match="not found"):
        agent._resolve_tool_manager("nonexistent_tool")


# ============================================================
# Task Tool Removal
# ============================================================

@pytest.mark.asyncio
async def test_task_remove_tools_by_name():
    """Removing task tools by name should update registered_task_tools and tool_manager."""
    agent = Agent(model=MODEL, name="Test Agent")
    task = Task(description="test", tools=[add_numbers, multiply_numbers])

    agent._setup_task_tools(task)
    assert "add_numbers" in task.registered_task_tools
    assert "multiply_numbers" in task.registered_task_tools

    task.remove_tools("add_numbers")

    assert "add_numbers" not in task.registered_task_tools
    assert "multiply_numbers" in task.registered_task_tools

    defs = task.get_tool_defs()
    def_names = [d.name for d in defs]
    assert "add_numbers" not in def_names
    assert "multiply_numbers" in def_names


@pytest.mark.asyncio
async def test_task_remove_tools_backward_compat():
    """task.remove_tools(tools, agent) should still work (agent param deprecated)."""
    agent = Agent(model=MODEL, name="Test Agent")
    task = Task(description="test", tools=[add_numbers, multiply_numbers])

    agent._setup_task_tools(task)

    task.remove_tools("add_numbers", agent)
    assert "add_numbers" not in task.registered_task_tools
    assert "multiply_numbers" in task.registered_task_tools


@pytest.mark.asyncio
async def test_task_remove_toolkit_by_object():
    """Removing a ToolKit by object should remove all its methods."""
    agent = Agent(model=MODEL, name="Test Agent")
    math_kit = MathToolKit()
    task = Task(description="test", tools=[math_kit, add_numbers])

    agent._setup_task_tools(task)
    assert "divide" in task.registered_task_tools
    assert "modulo" in task.registered_task_tools
    assert "add_numbers" in task.registered_task_tools

    task.remove_tools(math_kit)

    assert "divide" not in task.registered_task_tools
    assert "modulo" not in task.registered_task_tools
    assert "add_numbers" in task.registered_task_tools
    assert math_kit not in task.tools


@pytest.mark.asyncio
async def test_task_remove_builtin_tools():
    """Removing builtin tools from task should work without requiring agent."""
    agent = Agent(model=MODEL, name="Test Agent")
    web_search = WebSearchTool()
    task = Task(description="test", tools=[web_search, add_numbers])

    agent._setup_task_tools(task)
    assert len(task.task_builtin_tools) == 1
    assert "add_numbers" in task.registered_task_tools

    task.remove_tools([web_search])
    assert len(task.task_builtin_tools) == 0
    assert web_search not in task.tools
    assert "add_numbers" in task.registered_task_tools


@pytest.mark.asyncio
async def test_task_remove_individual_toolkit_method():
    """Removing one ToolKit method by name should keep the other."""
    agent = Agent(model=MODEL, name="Test Agent")
    math_kit = MathToolKit()
    task = Task(description="test", tools=[math_kit])

    agent._setup_task_tools(task)
    assert "divide" in task.registered_task_tools
    assert "modulo" in task.registered_task_tools

    task.remove_tools("divide")

    assert "divide" not in task.registered_task_tools
    assert "modulo" in task.registered_task_tools
    assert math_kit in task.tools


# ============================================================
# Task Tool Manager Processor State
# ============================================================

@pytest.mark.asyncio
async def test_task_tool_manager_processor_has_tools():
    """Task's ToolProcessor should have the task tools registered."""
    agent = Agent(model=MODEL, name="Test Agent")
    task = Task(description="test", tools=[add_numbers])

    agent._setup_task_tools(task)

    processor = task.tool_manager.processor
    assert "add_numbers" in processor.registered_tools
    assert len(processor.registered_tools) >= 1


@pytest.mark.asyncio
async def test_task_tool_manager_processor_isolated_from_agent():
    """Task's ToolProcessor should not contain agent tools."""
    agent = Agent(model=MODEL, name="Test Agent", tools=[greet])
    task = Task(description="test", tools=[add_numbers])

    agent._setup_task_tools(task)

    agent_processor = agent.tool_manager.processor
    task_processor = task.tool_manager.processor

    assert "greet" in agent_processor.registered_tools
    assert "greet" not in task_processor.registered_tools

    assert "add_numbers" in task_processor.registered_tools
    assert "add_numbers" not in agent_processor.registered_tools


@pytest.mark.asyncio
async def test_task_tool_manager_wrapped_tools():
    """Task's ToolManager should have wrapped_tools for task tools."""
    agent = Agent(model=MODEL, name="Test Agent", tools=[greet])
    task = Task(description="test", tools=[add_numbers])

    agent._setup_task_tools(task)

    assert "add_numbers" in task.tool_manager.wrapped_tools
    assert "greet" not in task.tool_manager.wrapped_tools

    assert "greet" in agent.tool_manager.wrapped_tools
    assert "add_numbers" not in agent.tool_manager.wrapped_tools


# ============================================================
# Multiple Tasks Isolation
# ============================================================

@pytest.mark.asyncio
async def test_multiple_tasks_have_independent_tool_managers():
    """Different tasks should get independent ToolManagers."""
    agent = Agent(model=MODEL, name="Test Agent")
    task1 = Task(description="test 1", tools=[add_numbers])
    task2 = Task(description="test 2", tools=[multiply_numbers])

    agent._setup_task_tools(task1)
    agent._setup_task_tools(task2)

    assert task1.tool_manager is not task2.tool_manager
    assert task1.tool_manager is not agent.tool_manager
    assert task2.tool_manager is not agent.tool_manager

    task1_names = {d.name for d in task1.get_tool_defs()}
    task2_names = {d.name for d in task2.get_tool_defs()}

    assert "add_numbers" in task1_names
    assert "multiply_numbers" not in task1_names

    assert "multiply_numbers" in task2_names
    assert "add_numbers" not in task2_names


# ============================================================
# Task Tool Add (before execution)
# ============================================================

@pytest.mark.asyncio
async def test_task_add_tools_before_setup():
    """Adding tools to task before setup should only add to task.tools, not register."""
    task = Task(description="test", tools=[add_numbers])
    task.add_tools(multiply_numbers)

    assert add_numbers in task.tools
    assert multiply_numbers in task.tools
    assert len(task.tools) == 2
    assert len(task.registered_task_tools) == 0
    assert task.tool_manager is None


@pytest.mark.asyncio
async def test_task_add_tools_then_setup():
    """Tools added via add_tools should be registered when _setup_task_tools is called."""
    agent = Agent(model=MODEL, name="Test Agent")
    task = Task(description="test", tools=[add_numbers])
    task.add_tools([multiply_numbers, greet])

    agent._setup_task_tools(task)

    assert "add_numbers" in task.registered_task_tools
    assert "multiply_numbers" in task.registered_task_tools
    assert "greet" in task.registered_task_tools

    defs = task.get_tool_defs()
    assert len(defs) == 3
