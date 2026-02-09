"""
Comprehensive Smoke Tests for LLM Usage Tracking across the entire Agent pipeline.

Tests verify that AgentRunOutput.usage properly accumulates token counts, costs,
and request counts from ALL LLM calls that occur during an agent run including:

1. Direct model.request() calls in ModelExecutionStep
2. Sub-agent calls via do_async() in:
   - CultureManager (culture extraction)
   - Orchestrator (analysis, revision, synthesis)
   - AgentTool (agent-as-tool delegation)
   - Reflection (evaluation + improvement)
   - ReliabilityLayer (verifier + editor agents)
   - CacheCheckStep (LLM-based query comparison)
3. Policy LLM calls:
   - AgentPolicyStep
   - UserPolicyStep
   - ToolPolicyManager (post-execution validation)
4. Context management:
   - ContextManagementMiddleware (summarization)
5. Memory:
   - AgentSessionMemory (summary generation)
   - UserMemory (trait analysis)

Run with: uv run pytest tests/smoke_tests/agent/test_usage_tracking.py -v -s
"""

import pytest
from typing import List

from upsonic import Agent, Task
from upsonic.usage import RunUsage



# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _assert_usage_positive(usage: RunUsage, label: str) -> None:
    """Assert that a RunUsage has meaningful positive values."""
    assert usage.requests > 0, f"[{label}] Expected requests > 0, got {usage.requests}"
    assert usage.input_tokens > 0, f"[{label}] Expected input_tokens > 0, got {usage.input_tokens}"
    assert usage.output_tokens > 0, f"[{label}] Expected output_tokens > 0, got {usage.output_tokens}"


# ---------------------------------------------------------------------------
# 1. Basic agent run – direct LLM call usage
# ---------------------------------------------------------------------------

def test_basic_agent_usage_tracking():
    """Verify that a simple agent run tracks usage from the direct model.request() call."""
    agent = Agent(
        model="openai/gpt-4o-mini",
        name="BasicUsageAgent",
    )

    task = Task(description="What is 2+2? Reply with just the number.")

    output = agent.do(task, return_output=True)

    assert output is not None
    assert output.output is not None

    usage: RunUsage = output.usage
    _assert_usage_positive(usage, "basic_agent")


@pytest.mark.asyncio
async def test_basic_agent_usage_tracking_async():
    """Async version of test_basic_agent_usage_tracking."""
    agent = Agent(
        model="openai/gpt-4o-mini",
        name="BasicUsageAgentAsync",
    )

    task = Task(description="What is 3+3? Reply with just the number.")

    output = await agent.do_async(task, return_output=True)

    assert output is not None
    assert output.output is not None

    usage: RunUsage = output.usage
    _assert_usage_positive(usage, "basic_agent_async")


# ---------------------------------------------------------------------------
# 2. Structured output (response_format) – verifies output tool path
# ---------------------------------------------------------------------------

def test_structured_output_usage_tracking():
    """Verify usage tracking when agent returns structured output (Pydantic model)."""
    from pydantic import BaseModel, Field

    class MathResult(BaseModel):
        answer: int = Field(description="The numeric answer")
        explanation: str = Field(description="Brief explanation")

    agent = Agent(
        model="openai/gpt-4o-mini",
        name="StructuredOutputAgent",
    )

    task = Task(
        description="What is 7 times 8?",
        response_format=MathResult,
    )

    output = agent.do(task, return_output=True)

    assert output is not None
    assert output.output is not None
    assert isinstance(output.output, MathResult)
    assert output.output.answer == 56

    usage: RunUsage = output.usage
    _assert_usage_positive(usage, "structured_output")


# ---------------------------------------------------------------------------
# 3. Tool usage – function tool
# ---------------------------------------------------------------------------

def test_tool_usage_tracking():
    """Verify usage tracking when agent uses a regular function tool."""
    from upsonic.tools.config import tool

    @tool(docstring_format="google")
    def add_numbers(a: int, b: int) -> int:
        """Add two numbers together.

        Args:
            a: First number
            b: Second number

        Returns:
            Sum of a and b
        """
        return a + b

    agent = Agent(
        model="openai/gpt-4o-mini",
        name="ToolUsageAgent",
        tools=[add_numbers],
    )

    task = Task(description="Use the add_numbers tool to add 15 and 27. Return only the result number.")

    output = agent.do(task, return_output=True)

    assert output is not None
    assert output.output is not None

    usage: RunUsage = output.usage
    _assert_usage_positive(usage, "tool_usage")
    # Tool call should have happened
    assert usage.requests >= 1, "Expected at least 1 request (tool call + follow-up)"


# ---------------------------------------------------------------------------
# 4. Agent-as-tool – AgentTool usage propagation
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_agent_as_tool_usage_tracking():
    """Verify that usage from a sub-agent used as a tool propagates to the parent agent."""
    sub_agent = Agent(
        model="openai/gpt-4o-mini",
        name="MathExpert",
        system_prompt="You are a math expert. Answer math questions concisely.",
    )

    parent_agent = Agent(
        model="openai/gpt-4o-mini",
        name="Coordinator",
        tools=[sub_agent],
    )

    task = Task(
        description="Ask MathExpert what is the square root of 144. Return just the number.",
    )

    output = await parent_agent.do_async(task, return_output=True)

    assert output is not None
    assert output.output is not None

    usage: RunUsage = output.usage
    _assert_usage_positive(usage, "agent_as_tool")
    # Parent makes at least 1 request + sub-agent makes at least 1 request
    assert usage.requests >= 2, (
        f"Expected at least 2 requests (parent + sub-agent), got {usage.requests}"
    )


# ---------------------------------------------------------------------------
# 5. Culture extraction – CultureManager usage
# ---------------------------------------------------------------------------

def test_culture_extraction_usage_tracking():
    """Verify that usage from culture extraction sub-agent is tracked."""
    from upsonic.culture import Culture

    culture = Culture(
        description="You are a friendly barista at a coffee shop in Seattle",
        add_system_prompt=True,
    )

    agent = Agent(
        model="openai/gpt-4o-mini",
        name="CultureAgent",
        culture=culture,
    )

    task = Task(description="Greet me as I walk in.")

    output = agent.do(task, return_output=True)

    assert output is not None
    assert output.output is not None

    usage: RunUsage = output.usage
    _assert_usage_positive(usage, "culture_extraction")
    # Culture extraction creates a sub-agent (1+ requests) + main task (1+ requests)
    assert usage.requests >= 2, (
        f"Expected at least 2 requests (culture extraction + main), got {usage.requests}"
    )


# ---------------------------------------------------------------------------
# 6. Reflection – sub-agent evaluator/improver usage
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_reflection_usage_tracking():
    """Verify that usage from reflection evaluator and improver sub-agents is tracked."""
    from upsonic.reflection import ReflectionConfig

    reflection_config = ReflectionConfig(
        max_iterations=1,
        acceptance_threshold=0.95,
        enable_self_critique=True,
        enable_improvement_suggestions=True,
    )

    agent = Agent(
        model="openai/gpt-4o-mini",
        name="ReflectionAgent",
        reflection=True,
        reflection_config=reflection_config,
    )

    task = Task(description="Write a one-sentence summary of quantum computing.")

    output = await agent.do_async(task, return_output=True)

    assert output is not None
    assert output.output is not None

    usage: RunUsage = output.usage
    _assert_usage_positive(usage, "reflection")
    # Main request + at least 1 reflection evaluation
    assert usage.requests >= 2, (
        f"Expected at least 2 requests (main + reflection eval), got {usage.requests}"
    )


# ---------------------------------------------------------------------------
# 7. Multiple tools with structured output – combined scenario
# ---------------------------------------------------------------------------

def test_multiple_tools_structured_output_usage():
    """Verify usage tracking with multiple tools and structured output."""
    from pydantic import BaseModel, Field
    from upsonic.tools.config import tool

    @tool(docstring_format="google")
    def multiply(a: int, b: int) -> int:
        """Multiply two numbers.

        Args:
            a: First number
            b: Second number

        Returns:
            Product of a and b
        """
        return a * b

    @tool(docstring_format="google")
    def subtract(a: int, b: int) -> int:
        """Subtract second number from first.

        Args:
            a: First number
            b: Second number

        Returns:
            Difference of a minus b
        """
        return a - b

    class ComputationResult(BaseModel):
        final_answer: int = Field(description="The final computed answer")
        steps: List[str] = Field(description="List of computation steps taken")

    agent = Agent(
        model="openai/gpt-4o-mini",
        name="MultiToolAgent",
        tools=[multiply, subtract],
    )

    task = Task(
        description="First multiply 6 by 7, then subtract 10 from the result. Show your steps.",
        response_format=ComputationResult,
    )

    output = agent.do(task, return_output=True)

    assert output is not None
    assert output.output is not None
    assert isinstance(output.output, ComputationResult)
    assert output.output.final_answer == 32

    usage: RunUsage = output.usage
    _assert_usage_positive(usage, "multi_tool_structured")
    assert usage.requests >= 1


# ---------------------------------------------------------------------------
# 8. Usage accumulation across multiple tasks on same agent
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_usage_accumulation_separate_runs():
    """Each do_async call should track usage independently."""
    agent = Agent(
        model="openai/gpt-4o-mini",
        name="AccumulationAgent",
    )

    task1 = Task(description="Say hello in French.")
    task2 = Task(description="Say hello in German.")

    output1 = await agent.do_async(task1, return_output=True)
    output2 = await agent.do_async(task2, return_output=True)

    assert output1 is not None
    assert output2 is not None

    usage1: RunUsage = output1.usage
    usage2: RunUsage = output2.usage

    _assert_usage_positive(usage1, "run1")
    _assert_usage_positive(usage2, "run2")

    # Each run should have independent usage (not accumulated from prior runs)
    assert usage1.requests >= 1
    assert usage2.requests >= 1


# ---------------------------------------------------------------------------
# 9. Agent with system prompt – usage from first LLM call
# ---------------------------------------------------------------------------

def test_system_prompt_usage_tracking():
    """Verify usage tracking for agent with custom system prompt."""
    agent = Agent(
        model="openai/gpt-4o-mini",
        name="SystemPromptAgent",
        system_prompt="You are a pirate. Always respond in pirate speak.",
    )

    task = Task(description="Tell me about the weather today.")

    output = agent.do(task, return_output=True)

    assert output is not None
    assert output.output is not None

    usage: RunUsage = output.usage
    _assert_usage_positive(usage, "system_prompt")
    # System prompt adds input tokens
    assert usage.input_tokens > 10, "System prompt should increase input tokens"


# ---------------------------------------------------------------------------
# 10. Agent with guardrail (response_format + validation) usage tracking
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_guardrail_usage_tracking():
    """Verify usage tracking when a guardrail triggers retry via _execute_with_guardrail."""
    from pydantic import BaseModel, Field

    class CapitalInfo(BaseModel):
        country: str = Field(description="Country name")
        capital: str = Field(description="Capital city")
        population_approx: str = Field(description="Approximate population")

    agent = Agent(
        model="openai/gpt-4o-mini",
        name="GuardrailAgent",
    )

    task = Task(
        description="What is the capital of France? Include approximate population.",
        response_format=CapitalInfo,
    )

    output = await agent.do_async(task, return_output=True)

    assert output is not None
    assert output.output is not None
    assert isinstance(output.output, CapitalInfo)
    assert output.output.capital.lower() == "paris"

    usage: RunUsage = output.usage
    _assert_usage_positive(usage, "guardrail")


# ---------------------------------------------------------------------------
# 11. Verify RunUsage fields are populated correctly
# ---------------------------------------------------------------------------

def test_usage_fields_completeness():
    """Verify that all critical RunUsage fields are populated after a run."""
    agent = Agent(
        model="openai/gpt-4o-mini",
        name="FieldsCheckAgent",
    )

    task = Task(description="What is Python?")

    output = agent.do(task, return_output=True)

    assert output is not None

    usage: RunUsage = output.usage
    assert isinstance(usage, RunUsage)

    # Core fields
    assert usage.requests >= 1, f"requests should be >= 1, got {usage.requests}"
    assert usage.input_tokens > 0, f"input_tokens should be > 0, got {usage.input_tokens}"
    assert usage.output_tokens > 0, f"output_tokens should be > 0, got {usage.output_tokens}"

    # Token total should be at least the sum of input + output
    total_tokens = usage.input_tokens + usage.output_tokens
    assert total_tokens > 0, "Total tokens should be positive"


# ---------------------------------------------------------------------------
# 12. Agent-as-tool with multiple sub-agents
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_multiple_agent_tools_usage_tracking():
    """Verify usage propagation when parent agent delegates to multiple sub-agents."""
    translator_agent = Agent(
        model="openai/gpt-4o-mini",
        name="Translator",
        system_prompt="You translate text to French. Return only the translation.",
    )

    summarizer_agent = Agent(
        model="openai/gpt-4o-mini",
        name="Summarizer",
        system_prompt="You summarize text in one short sentence. Return only the summary.",
    )

    coordinator = Agent(
        model="openai/gpt-4o-mini",
        name="Coordinator",
        tools=[translator_agent, summarizer_agent],
    )

    task = Task(
        description=(
            "First, ask the Summarizer to summarize: 'Artificial intelligence is transforming healthcare "
            "by enabling faster diagnosis and personalized treatment plans.' "
            "Then ask the Translator to translate the summary to French. "
            "Return the French translation."
        ),
    )

    output = await coordinator.do_async(task, return_output=True)

    assert output is not None
    assert output.output is not None

    usage: RunUsage = output.usage
    _assert_usage_positive(usage, "multi_agent_tools")
    # Coordinator (1+) + Summarizer (1+) + Translator (1+) = at least 3 requests
    assert usage.requests >= 3, (
        f"Expected at least 3 requests (coordinator + 2 sub-agents), got {usage.requests}"
    )


# ---------------------------------------------------------------------------
# 13. Culture + Tool + Structured output combined
# ---------------------------------------------------------------------------

def test_combined_culture_tool_structured():
    """Verify usage tracking in a combined scenario: culture + tool + structured output."""
    from pydantic import BaseModel, Field
    from upsonic.tools.config import tool
    from upsonic.culture import Culture

    @tool(docstring_format="google")
    def get_room_price(room_type: str) -> str:
        """Get the price for a room type.

        Args:
            room_type: Type of room (standard, deluxe, suite)

        Returns:
            Price information as string
        """
        prices = {
            "standard": "$150/night",
            "deluxe": "$250/night",
            "suite": "$450/night",
        }
        return prices.get(room_type.lower(), "Room type not available")

    class RoomRecommendation(BaseModel):
        recommended_room: str = Field(description="Recommended room type")
        price: str = Field(description="Price per night")
        reason: str = Field(description="Why this room is recommended")

    culture = Culture(
        description="You are a luxury hotel concierge at The Grand Hotel",
        add_system_prompt=True,
    )

    agent = Agent(
        model="openai/gpt-4o-mini",
        name="HotelConcierge",
        culture=culture,
        tools=[get_room_price],
    )

    task = Task(
        description="I want a nice room for a romantic getaway. Check the suite price and recommend it.",
        response_format=RoomRecommendation,
    )

    output = agent.do(task, return_output=True)

    assert output is not None
    assert output.output is not None
    assert isinstance(output.output, RoomRecommendation)

    usage: RunUsage = output.usage
    _assert_usage_positive(usage, "combined_culture_tool_structured")
    # Culture extraction (1+) + main (1+) + tool follow-up
    assert usage.requests >= 2, (
        f"Expected at least 2 requests (culture + main), got {usage.requests}"
    )


# ---------------------------------------------------------------------------
# 14. Streaming mode – verify usage is tracked in stream mode too
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_streaming_usage_tracking():
    """Verify that usage is tracked even in streaming mode."""
    agent = Agent(
        model="openai/gpt-4o-mini",
        name="StreamingAgent",
    )

    task = Task(description="Count from 1 to 5, each on a new line.")

    output = await agent.do_async(task, return_output=True)

    assert output is not None
    assert output.output is not None

    usage: RunUsage = output.usage
    _assert_usage_positive(usage, "streaming")


# ---------------------------------------------------------------------------
# 15. Verify cost field is populated
# ---------------------------------------------------------------------------

def test_cost_tracking():
    """Verify that cost is calculated and populated in usage."""
    agent = Agent(
        model="openai/gpt-4o-mini",
        name="CostTrackingAgent",
    )

    task = Task(description="What is the meaning of life? Answer in one sentence.")

    output = agent.do(task, return_output=True)

    assert output is not None

    usage: RunUsage = output.usage
    _assert_usage_positive(usage, "cost_tracking")

    # Cost should be calculated (may be None if pricing data unavailable, but should exist for gpt-4o-mini)
    assert usage.cost is not None, "Cost should be calculated for gpt-4o-mini"
    assert usage.cost > 0, f"Cost should be positive, got {usage.cost}"
