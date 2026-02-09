"""
Unit tests for LLM usage tracking across the Agent pipeline.

Tests verify usage accumulation and propagation logic without making real API calls.
All tests use mocks; no API key required.

Covers:
- RunUsage / RequestUsage incr and aggregation
- AgentRunOutput.update_usage_from_response, set_usage_cost, _ensure_usage
- CultureManager _last_llm_usage accumulation and drain_accumulated_usage
- Orchestrator _propagate_sub_agent_usage
- AgentTool _accumulated_usage and drain_accumulated_usage
- Agent._drain_agent_tool_usage
- SystemPromptManager draining culture usage into agent run output
"""

from __future__ import annotations

import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock

from upsonic.usage import RunUsage, RequestUsage
from upsonic.run.agent.output import AgentRunOutput


# ---------------------------------------------------------------------------
# RunUsage and RequestUsage
# ---------------------------------------------------------------------------

class TestRunUsageRequestUsage:
    """Unit tests for RunUsage and RequestUsage aggregation."""

    def test_run_usage_incr_request_usage(self) -> None:
        """RunUsage.incr(RequestUsage) increments tokens and requests."""
        run_usage = RunUsage()
        req_usage = RequestUsage(input_tokens=100, output_tokens=50)

        run_usage.incr(req_usage)

        assert run_usage.requests == 1
        assert run_usage.input_tokens == 100
        assert run_usage.output_tokens == 50

    def test_run_usage_incr_request_usage_twice(self) -> None:
        """RunUsage.incr(RequestUsage) twice accumulates correctly."""
        run_usage = RunUsage()
        req1 = RequestUsage(input_tokens=10, output_tokens=5)
        req2 = RequestUsage(input_tokens=20, output_tokens=15)

        run_usage.incr(req1)
        run_usage.incr(req2)

        assert run_usage.requests == 2
        assert run_usage.input_tokens == 30
        assert run_usage.output_tokens == 20

    def test_run_usage_incr_run_usage(self) -> None:
        """RunUsage.incr(RunUsage) merges requests, tokens, and cost."""
        run_usage = RunUsage()
        other = RunUsage(
            requests=2,
            input_tokens=200,
            output_tokens=80,
            cost=0.001,
        )

        run_usage.incr(other)

        assert run_usage.requests == 2
        assert run_usage.input_tokens == 200
        assert run_usage.output_tokens == 80
        assert run_usage.cost == 0.001

    def test_run_usage_incr_run_usage_cost_accumulates(self) -> None:
        """RunUsage.incr(RunUsage) accumulates cost when both have cost."""
        run_usage = RunUsage(cost=0.001)
        other = RunUsage(cost=0.002, requests=1, input_tokens=1, output_tokens=1)

        run_usage.incr(other)

        assert run_usage.cost == 0.003


# ---------------------------------------------------------------------------
# AgentRunOutput usage methods
# ---------------------------------------------------------------------------

class TestAgentRunOutputUsage:
    """Unit tests for AgentRunOutput usage tracking methods."""

    def test_ensure_usage_creates_run_usage(self) -> None:
        """_ensure_usage() creates RunUsage when usage is None."""
        output = AgentRunOutput(
            run_id="r1",
            agent_id="a1",
            agent_name="Test",
            session_id="s1",
            user_id="u1",
        )
        output.usage = None

        usage = output._ensure_usage()

        assert usage is not None
        assert isinstance(usage, RunUsage)
        assert output.usage is usage

    def test_update_usage_from_response_request_usage(self) -> None:
        """update_usage_from_response(RequestUsage) increments usage."""
        output = AgentRunOutput(
            run_id="r1",
            agent_id="a1",
            agent_name="Test",
            session_id="s1",
            user_id="u1",
        )

        req_usage = RequestUsage(input_tokens=50, output_tokens=25)
        output.update_usage_from_response(req_usage)

        assert output.usage.requests == 1
        assert output.usage.input_tokens == 50
        assert output.usage.output_tokens == 25

    def test_update_usage_from_response_dict(self) -> None:
        """update_usage_from_response(dict) converts and increments."""
        output = AgentRunOutput(
            run_id="r1",
            agent_id="a1",
            agent_name="Test",
            session_id="s1",
            user_id="u1",
        )

        output.update_usage_from_response({
            "input_tokens": 30,
            "output_tokens": 10,
        })

        assert output.usage.requests == 1
        assert output.usage.input_tokens == 30
        assert output.usage.output_tokens == 10

    def test_set_usage_cost_initial(self) -> None:
        """set_usage_cost sets cost when current cost is None."""
        output = AgentRunOutput(
            run_id="r1",
            agent_id="a1",
            agent_name="Test",
            session_id="s1",
            user_id="u1",
        )

        output.set_usage_cost(0.005)

        assert output.usage.cost == 0.005

    def test_set_usage_cost_accumulates(self) -> None:
        """set_usage_cost adds to existing cost."""
        output = AgentRunOutput(
            run_id="r1",
            agent_id="a1",
            agent_name="Test",
            session_id="s1",
            user_id="u1",
        )
        output._ensure_usage().cost = 0.001

        output.set_usage_cost(0.002)

        assert output.usage.cost == 0.003


# ---------------------------------------------------------------------------
# CultureManager usage accumulation and drain
# ---------------------------------------------------------------------------

class TestCultureManagerUsageTracking:
    """Unit tests for CultureManager usage accumulation and drain."""

    def test_drain_accumulated_usage_returns_none_when_empty(self) -> None:
        """drain_accumulated_usage returns None when no usage was accumulated."""
        from upsonic.culture.manager import CultureManager

        manager = CultureManager(model="openai/gpt-4o")
        manager._last_llm_usage = None

        result = manager.drain_accumulated_usage()

        assert result is None

    def test_drain_accumulated_usage_returns_and_resets(self) -> None:
        """drain_accumulated_usage returns stored usage and resets internal state."""
        from upsonic.culture.manager import CultureManager

        manager = CultureManager(model="openai/gpt-4o")
        stored = RunUsage(requests=1, input_tokens=100, output_tokens=50, cost=0.0001)
        manager._last_llm_usage = stored

        result = manager.drain_accumulated_usage()

        assert result is stored
        assert manager._last_llm_usage is None

    def test_drain_accumulated_usage_second_call_returns_none(self) -> None:
        """Second drain after first returns None."""
        from upsonic.culture.manager import CultureManager

        manager = CultureManager(model="openai/gpt-4o")
        manager._last_llm_usage = RunUsage(requests=1)

        first = manager.drain_accumulated_usage()
        second = manager.drain_accumulated_usage()

        assert first is not None
        assert second is None


# ---------------------------------------------------------------------------
# Orchestrator _propagate_sub_agent_usage
# ---------------------------------------------------------------------------

class TestOrchestratorUsagePropagation:
    """Unit tests for Orchestrator sub-agent usage propagation."""

    def test_propagate_sub_agent_usage_no_agent_instance(self) -> None:
        """_propagate_sub_agent_usage does nothing when agent_instance is None."""
        from upsonic.tools.orchestration import Orchestrator

        orch = Orchestrator(agent_instance=None, task=None, wrapped_tools={})
        sub_output = Mock()
        sub_output.usage = RunUsage(requests=1, input_tokens=10, output_tokens=5)

        orch._propagate_sub_agent_usage(sub_output)

        # No exception; no parent to update
        assert True

    def test_propagate_sub_agent_usage_no_run_output(self) -> None:
        """_propagate_sub_agent_usage does nothing when parent has no _agent_run_output."""
        from upsonic.tools.orchestration import Orchestrator

        parent = Mock()
        del parent._agent_run_output
        orch = Orchestrator(agent_instance=parent, task=None, wrapped_tools={})
        sub_output = Mock()
        sub_output.usage = RunUsage(requests=1, input_tokens=10, output_tokens=5)

        orch._propagate_sub_agent_usage(sub_output)

        assert not hasattr(parent, "_agent_run_output") or getattr(parent, "_agent_run_output", None) is None

    def test_propagate_sub_agent_usage_increments_parent_usage(self) -> None:
        """_propagate_sub_agent_usage increments parent agent's run output usage."""
        from upsonic.tools.orchestration import Orchestrator

        parent_usage = RunUsage()
        parent_output = Mock()
        parent_output.usage = parent_usage
        parent = Mock(_agent_run_output=parent_output)

        orch = Orchestrator(agent_instance=parent, task=None, wrapped_tools={})
        sub_output = Mock()
        sub_output.usage = RunUsage(requests=1, input_tokens=100, output_tokens=50, cost=0.001)

        orch._propagate_sub_agent_usage(sub_output)

        assert parent_usage.requests == 1
        assert parent_usage.input_tokens == 100
        assert parent_usage.output_tokens == 50
        assert parent_usage.cost == 0.001

    def test_propagate_sub_agent_usage_ignores_none_usage(self) -> None:
        """_propagate_sub_agent_usage does nothing when sub output has no usage."""
        from upsonic.tools.orchestration import Orchestrator

        parent_usage = RunUsage()
        parent_output = Mock()
        parent_output.usage = parent_usage
        parent = Mock(_agent_run_output=parent_output)

        orch = Orchestrator(agent_instance=parent, task=None, wrapped_tools={})
        sub_output = Mock(spec=[])  # no usage attr
        sub_output.usage = None

        orch._propagate_sub_agent_usage(sub_output)

        assert parent_usage.requests == 0
        assert parent_usage.input_tokens == 0


# ---------------------------------------------------------------------------
# AgentTool usage accumulation and drain
# ---------------------------------------------------------------------------

class TestAgentToolUsageTracking:
    """Unit tests for AgentTool usage accumulation and drain."""

    def test_agent_tool_has_accumulated_usage_attr(self) -> None:
        """AgentTool initializes _accumulated_usage as None."""
        from upsonic.tools.wrappers import AgentTool

        mock_agent = Mock()
        mock_agent.name = "Helper"
        tool = AgentTool(mock_agent)

        assert hasattr(tool, "_accumulated_usage")
        assert tool._accumulated_usage is None

    def test_agent_tool_drain_accumulated_usage_returns_none_when_empty(self) -> None:
        """drain_accumulated_usage returns None when no usage was accumulated."""
        from upsonic.tools.wrappers import AgentTool

        mock_agent = Mock()
        mock_agent.name = "Helper"
        tool = AgentTool(mock_agent)
        tool._accumulated_usage = None

        result = tool.drain_accumulated_usage()

        assert result is None

    def test_agent_tool_drain_accumulated_usage_returns_and_resets(self) -> None:
        """drain_accumulated_usage returns stored usage and resets."""
        from upsonic.tools.wrappers import AgentTool

        mock_agent = Mock()
        mock_agent.name = "Helper"
        tool = AgentTool(mock_agent)
        stored = RunUsage(requests=1, input_tokens=50, output_tokens=20)
        tool._accumulated_usage = stored

        result = tool.drain_accumulated_usage()

        assert result is stored
        assert tool._accumulated_usage is None


# ---------------------------------------------------------------------------
# Agent._drain_agent_tool_usage
# ---------------------------------------------------------------------------

class TestAgentDrainAgentToolUsage:
    """Unit tests for Agent._drain_agent_tool_usage."""

    @patch("upsonic.models.infer_model")
    def test_drain_agent_tool_usage_non_agent_tool_ignored(self, mock_infer_model: Mock) -> None:
        """_drain_agent_tool_usage does nothing when registered tool is not AgentTool."""
        from upsonic.agent.agent import Agent

        mock_infer_model.return_value = MagicMock(model_name="openai/gpt-4o-mini")
        agent = Agent(model="openai/gpt-4o-mini", name="Test")
        run_output = AgentRunOutput(
            run_id="r1",
            agent_id="a1",
            agent_name="Test",
            session_id="s1",
            user_id="u1",
        )
        run_output.usage = RunUsage()
        agent._agent_run_output = run_output
        agent.tool_manager = Mock()
        agent.tool_manager.processor = Mock()
        agent.tool_manager.processor.registered_tools = {
            "some_tool": Mock(),  # Not an AgentTool
        }

        agent._drain_agent_tool_usage("some_tool")

        assert agent._agent_run_output.usage.requests == 0
        assert agent._agent_run_output.usage.input_tokens == 0

    @patch("upsonic.models.infer_model")
    def test_drain_agent_tool_usage_agent_tool_drains_into_run_output(self, mock_infer_model: Mock) -> None:
        """_drain_agent_tool_usage drains AgentTool usage into run output."""
        from upsonic.agent.agent import Agent
        from upsonic.tools.wrappers import AgentTool

        mock_infer_model.return_value = MagicMock(model_name="openai/gpt-4o-mini")
        agent = Agent(model="openai/gpt-4o-mini", name="Test")
        run_output = AgentRunOutput(
            run_id="r1",
            agent_id="a1",
            agent_name="Test",
            session_id="s1",
            user_id="u1",
        )
        run_output.usage = RunUsage()
        agent._agent_run_output = run_output

        mock_sub_agent = Mock()
        mock_sub_agent.name = "Sub"
        agent_tool = AgentTool(mock_sub_agent)
        agent_tool._accumulated_usage = RunUsage(
            requests=1,
            input_tokens=60,
            output_tokens=30,
            cost=0.0005,
        )

        agent.tool_manager = Mock()
        agent.tool_manager.processor = Mock()
        agent.tool_manager.processor.registered_tools = {
            "ask_sub": agent_tool,
        }

        agent._drain_agent_tool_usage("ask_sub")

        assert agent._agent_run_output.usage.requests == 1
        assert agent._agent_run_output.usage.input_tokens == 60
        assert agent._agent_run_output.usage.output_tokens == 30
        assert agent._agent_run_output.usage.cost == 0.0005
        assert agent_tool._accumulated_usage is None

    @patch("upsonic.models.infer_model")
    def test_drain_agent_tool_usage_unknown_tool_name(self, mock_infer_model: Mock) -> None:
        """_drain_agent_tool_usage does nothing for unknown tool name."""
        from upsonic.agent.agent import Agent

        mock_infer_model.return_value = MagicMock(model_name="openai/gpt-4o-mini")
        agent = Agent(model="openai/gpt-4o-mini", name="Test")
        run_output = AgentRunOutput(
            run_id="r1",
            agent_id="a1",
            agent_name="Test",
            session_id="s1",
            user_id="u1",
        )
        run_output.usage = RunUsage()
        agent._agent_run_output = run_output
        agent.tool_manager = Mock()
        agent.tool_manager.processor = Mock()
        agent.tool_manager.processor.registered_tools = {}

        agent._drain_agent_tool_usage("nonexistent_tool")

        assert agent._agent_run_output.usage.requests == 0


# ---------------------------------------------------------------------------
# SystemPromptManager culture usage drain (integration of culture into output)
# ---------------------------------------------------------------------------

class TestSystemPromptManagerCultureUsageDrain:
    """Unit tests for SystemPromptManager draining culture usage into agent run output."""

    @pytest.mark.asyncio
    async def test_aprepare_drains_culture_usage_into_agent_run_output(self) -> None:
        """When culture is prepared, drain_accumulated_usage is called and merged into agent._agent_run_output.usage."""
        from upsonic.agent.context_managers.system_prompt_manager import SystemPromptManager

        mock_agent = Mock()
        mock_agent._culture_manager = Mock()
        mock_agent._culture_manager.enabled = True
        mock_agent._culture_manager.prepared = False
        mock_agent._culture_manager.drain_accumulated_usage = Mock(
            return_value=RunUsage(requests=1, input_tokens=40, output_tokens=20, cost=0.0002)
        )
        run_output = AgentRunOutput(
            run_id="r1",
            agent_id="a1",
            agent_name="Test",
            session_id="s1",
            user_id="u1",
        )
        run_output.usage = RunUsage()
        mock_agent._agent_run_output = run_output

        mock_task = Mock()
        sp_manager = SystemPromptManager(mock_agent, mock_task)

        with patch.object(mock_agent._culture_manager, "aprepare", new_callable=AsyncMock), \
             patch.object(sp_manager, "_build_system_prompt", return_value=""):
            await sp_manager.aprepare(memory_handler=None)

        mock_agent._culture_manager.drain_accumulated_usage.assert_called_once()
        assert mock_agent._agent_run_output.usage.requests == 1
        assert mock_agent._agent_run_output.usage.input_tokens == 40
        assert mock_agent._agent_run_output.usage.output_tokens == 20
        assert mock_agent._agent_run_output.usage.cost == 0.0002

    @pytest.mark.asyncio
    async def test_aprepare_no_drain_when_culture_usage_none(self) -> None:
        """When drain returns None, agent run output usage is unchanged (no merge)."""
        from upsonic.agent.context_managers.system_prompt_manager import SystemPromptManager

        mock_agent = Mock()
        mock_agent._culture_manager = Mock()
        mock_agent._culture_manager.enabled = True
        mock_agent._culture_manager.prepared = False
        mock_agent._culture_manager.drain_accumulated_usage = Mock(return_value=None)
        run_output = AgentRunOutput(
            run_id="r1",
            agent_id="a1",
            agent_name="Test",
            session_id="s1",
            user_id="u1",
        )
        run_output.usage = RunUsage()
        mock_agent._agent_run_output = run_output

        mock_task = Mock()
        sp_manager = SystemPromptManager(mock_agent, mock_task)

        with patch.object(mock_agent._culture_manager, "aprepare", new_callable=AsyncMock), \
             patch.object(sp_manager, "_build_system_prompt", return_value=""):
            await sp_manager.aprepare(memory_handler=None)

        mock_agent._culture_manager.drain_accumulated_usage.assert_called_once()
        assert mock_agent._agent_run_output.usage.requests == 0
        assert mock_agent._agent_run_output.usage.input_tokens == 0


# ---------------------------------------------------------------------------
# RunUsage aggregation from multiple sources (sanity)
# ---------------------------------------------------------------------------

class TestRunUsageAggregation:
    """Sanity tests for RunUsage aggregation from multiple sources."""

    def test_aggregate_direct_request_plus_sub_agent_run_usage(self) -> None:
        """Simulate one direct RequestUsage + one sub-agent RunUsage."""
        run_usage = RunUsage()

        run_usage.incr(RequestUsage(input_tokens=100, output_tokens=50))
        run_usage.incr(RunUsage(requests=1, input_tokens=200, output_tokens=80, cost=0.001))

        assert run_usage.requests == 2
        assert run_usage.input_tokens == 300
        assert run_usage.output_tokens == 130
        assert run_usage.cost == 0.001

    def test_agent_run_output_multiple_updates(self) -> None:
        """AgentRunOutput can receive multiple update_usage_from_response and set_usage_cost."""
        output = AgentRunOutput(
            run_id="r1",
            agent_id="a1",
            agent_name="Test",
            session_id="s1",
            user_id="u1",
        )

        output.update_usage_from_response(RequestUsage(input_tokens=10, output_tokens=5))
        output.set_usage_cost(0.0001)
        output.update_usage_from_response(RequestUsage(input_tokens=20, output_tokens=10))
        output.set_usage_cost(0.0002)
        output.usage.incr(RunUsage(requests=1, input_tokens=30, output_tokens=15, cost=0.0003))

        assert output.usage.requests == 3
        assert output.usage.input_tokens == 60
        assert output.usage.output_tokens == 30
        assert output.usage.cost == pytest.approx(0.0006)
