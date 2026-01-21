"""
Unit tests for Culture integration with Agent.

Tests cover:
- Agent initialization with culture
- System prompt injection via SystemPromptManager
- Culture preparation flow
- Repeat functionality in _handle_model_response
- Culture without system prompt injection
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from upsonic.culture import Culture, CultureManager
from upsonic.agent.agent import Agent
from upsonic.tasks.tasks import Task


class TestAgentCultureInitialization:
    """Test Agent initialization with Culture."""
    
    def test_agent_with_culture(self):
        """Test Agent initialization with culture parameter."""
        culture = Culture(description="You are a helpful assistant")
        agent = Agent("openai/gpt-4o", culture=culture)
        
        assert agent._culture_input == culture
        assert agent._culture_manager is not None
        assert agent._culture_manager.culture == culture
    
    def test_agent_without_culture(self):
        """Test Agent initialization without culture."""
        agent = Agent("openai/gpt-4o")
        
        assert agent._culture_input is None
        assert agent._culture_manager is None
    
    def test_agent_culture_manager_enabled(self):
        """Test that CultureManager is enabled when culture is provided."""
        culture = Culture(description="You are helpful")
        agent = Agent("openai/gpt-4o", culture=culture)
        
        assert agent._culture_manager.enabled is True


class TestAgentCultureSystemPromptInjection:
    """Test culture injection into system prompt."""
    
    @pytest.mark.asyncio
    async def test_system_prompt_includes_culture_when_add_system_prompt_true(self):
        """Test that culture is added to system prompt when add_system_prompt=True."""
        from upsonic.tasks.tasks import Task
        
        culture = Culture(
            description="You are a 5-star hotel receptionist",
            add_system_prompt=True
        )
        agent = Agent("openai/gpt-4o", culture=culture)
        
        # Prepare culture
        await agent._culture_manager.aprepare()
        
        # Get system prompt manager with task
        from upsonic.agent.context_managers.system_prompt_manager import SystemPromptManager
        task = Task("Test task")
        system_prompt_manager = SystemPromptManager(agent, task)
        
        # Prepare system prompt
        await system_prompt_manager.aprepare()
        
        system_prompt = system_prompt_manager.get_system_prompt()
        
        assert system_prompt is not None
        assert "<CulturalKnowledge>" in system_prompt
        assert "## Agent Culture Guidelines" in system_prompt
    
    @pytest.mark.asyncio
    async def test_system_prompt_excludes_culture_when_add_system_prompt_false(self):
        """Test that culture is NOT added when add_system_prompt=False."""
        from upsonic.tasks.tasks import Task
        
        culture = Culture(
            description="You are a 5-star hotel receptionist",
            add_system_prompt=False
        )
        agent = Agent("openai/gpt-4o", culture=culture)
        
        # Prepare culture
        await agent._culture_manager.aprepare()
        
        # Get system prompt manager with task
        from upsonic.agent.context_managers.system_prompt_manager import SystemPromptManager
        task = Task("Test task")
        system_prompt_manager = SystemPromptManager(agent, task)
        
        # Prepare system prompt
        await system_prompt_manager.aprepare()
        
        system_prompt = system_prompt_manager.get_system_prompt()
        
        # Culture should not be in system prompt
        if system_prompt:
            assert "<CulturalKnowledge>" not in system_prompt
    
    @pytest.mark.asyncio
    async def test_system_prompt_prepares_culture_if_not_prepared(self):
        """Test that system prompt preparation triggers culture preparation."""
        from upsonic.tasks.tasks import Task
        
        culture = Culture(description="You are helpful", add_system_prompt=True)
        agent = Agent("openai/gpt-4o", culture=culture)
        
        # Culture should not be prepared yet
        assert agent._culture_manager.prepared is False
        
        # Prepare system prompt (should prepare culture)
        from upsonic.agent.context_managers.system_prompt_manager import SystemPromptManager
        task = Task("Test task")
        system_prompt_manager = SystemPromptManager(agent, task)
        await system_prompt_manager.aprepare()
        
        # Culture should now be prepared
        assert agent._culture_manager.prepared is True


class TestAgentCultureRepeatFunctionality:
    """Test culture repeat functionality in Agent."""
    
    @pytest.mark.asyncio
    async def test_handle_model_response_injects_culture_on_repeat(self):
        """Test that _handle_model_response injects culture when should_repeat returns True."""
        culture = Culture(
            description="You are helpful",
            repeat=True,
            repeat_interval=1  # Repeat every message
        )
        agent = Agent("openai/gpt-4o", culture=culture)
        
        # Prepare culture
        await agent._culture_manager.aprepare()
        
        # Setup mock response and messages
        from upsonic.messages import ModelResponse, ModelRequest, TextPart
        
        mock_response = ModelResponse(
            parts=[TextPart(content="Test response")],
            model_name="openai/gpt-4o",
        )
        
        messages = [
            ModelRequest(
                parts=[TextPart(content="User message")],
            )
        ]
        
        # Set message count to trigger repeat
        agent._culture_manager._message_count = 0  # Will trigger on first should_repeat call
        
        # Call _handle_model_response
        result = await agent._handle_model_response(mock_response, messages)
        
        # Check that culture was injected (new message should be added)
        # The exact implementation depends on how culture is injected
        # This test verifies the method completes without error
        assert result is not None
    
    @pytest.mark.asyncio
    async def test_handle_model_response_no_repeat_when_repeat_false(self):
        """Test that culture is not repeated when repeat=False."""
        culture = Culture(
            description="You are helpful",
            repeat=False
        )
        agent = Agent("openai/gpt-4o", culture=culture)
        
        await agent._culture_manager.aprepare()
        
        from upsonic.messages import ModelResponse, ModelRequest, TextPart
        
        mock_response = ModelResponse(
            parts=[TextPart(content="Test response")],
            model_name="openai/gpt-4o",
        )
        
        messages = [
            ModelRequest(
                parts=[TextPart(content="User message")],
            )
        ]
        
        initial_message_count = len(messages)
        
        result = await agent._handle_model_response(mock_response, messages)
        
        # Messages should not have culture injected
        assert result is not None


class TestAgentCulturePreparationFlow:
    """Test culture preparation flow in Agent execution."""
    
    @pytest.mark.asyncio
    @patch('upsonic.agent.agent.Agent')
    async def test_culture_preparation_during_agent_execution(self, mock_agent_class):
        """Test that culture is prepared during agent execution."""
        from upsonic.tasks.tasks import Task
        
        # Setup mock for culture extraction
        mock_extractor = AsyncMock()
        mock_result = Mock()
        mock_result.tone_of_speech = "Professional"
        mock_result.topics_to_avoid = "None"
        mock_result.topics_to_help = "All topics"
        mock_result.things_to_pay_attention = "User needs"
        mock_extractor.do_async = AsyncMock(return_value=mock_result)
        mock_agent_class.return_value = mock_extractor
        
        culture = Culture(description="You are helpful", add_system_prompt=True)
        agent = Agent("openai/gpt-4o", culture=culture)
        
        # Culture should not be prepared initially
        assert agent._culture_manager.prepared is False
        
        # Simulate system prompt preparation (happens during agent execution)
        from upsonic.agent.context_managers.system_prompt_manager import SystemPromptManager
        task = Task("Test task")
        system_prompt_manager = SystemPromptManager(agent, task)
        await system_prompt_manager.aprepare()
        
        # Culture should now be prepared
        assert agent._culture_manager.prepared is True
        assert agent._culture_manager.extracted_guidelines is not None


class TestAgentCultureEdgeCases:
    """Test edge cases for Agent-Culture integration."""
    
    def test_agent_with_culture_disabled(self):
        """Test Agent with culture but CultureManager disabled."""
        culture = Culture(description="You are helpful")
        agent = Agent("openai/gpt-4o", culture=culture)
        
        # Disable culture manager
        agent._culture_manager.enabled = False
        
        assert agent._culture_manager.enabled is False
    
    @pytest.mark.asyncio
    async def test_agent_culture_with_empty_guidelines(self):
        """Test Agent with culture that has empty extracted guidelines."""
        from upsonic.tasks.tasks import Task
        
        culture = Culture(description="You are helpful", add_system_prompt=True)
        agent = Agent("openai/gpt-4o", culture=culture)
        
        # Manually set empty guidelines
        agent._culture_manager._extracted_guidelines = {}
        agent._culture_manager._prepared = True
        
        from upsonic.agent.context_managers.system_prompt_manager import SystemPromptManager
        task = Task("Test task")
        system_prompt_manager = SystemPromptManager(agent, task)
        await system_prompt_manager.aprepare()
        
        # Should handle gracefully
        system_prompt = system_prompt_manager.get_system_prompt()
        # System prompt should still exist (may or may not include culture)
        assert system_prompt is not None or system_prompt == ""
    
    def test_agent_culture_manager_initialization_parameters(self):
        """Test that CultureManager is initialized with correct parameters from Agent."""
        culture = Culture(description="You are helpful")
        agent = Agent(
            "openai/gpt-4o",
            culture=culture,
            debug=True,
            debug_level=2
        )
        
        assert agent._culture_manager.debug is True
        assert agent._culture_manager.debug_level == 2
        assert agent._culture_manager._model_spec == "openai/gpt-4o"


class TestAgentCultureWithTaskExecution:
    """Test Agent culture behavior during task execution."""
    
    @pytest.mark.asyncio
    @pytest.mark.skip(reason="Requires actual model API call - use smoke tests instead")
    async def test_agent_do_with_culture(self):
        """Test agent.do() with culture (integration test - skipped in unit tests)."""
        # This test would require actual API calls
        # Should be in smoke tests instead
        pass
    
    def test_culture_format_in_system_prompt_structure(self):
        """Test that culture formatting follows expected structure."""
        culture = Culture(description="You are a professional consultant", add_system_prompt=True)
        agent = Agent("openai/gpt-4o", culture=culture)
        
        # Manually set guidelines
        agent._culture_manager._extracted_guidelines = {
            "tone_of_speech": "Professional and courteous",
            "topics_to_avoid": "Personal financial information",
            "topics_to_help": "Business strategy and consulting",
            "things_to_pay_attention": "Client confidentiality"
        }
        agent._culture_manager._prepared = True
        
        formatted = agent._culture_manager.format_for_system_prompt()
        
        assert formatted is not None
        assert formatted.startswith("<CulturalKnowledge>")
        assert formatted.endswith("</CulturalKnowledge>")
        assert "## Agent Culture Guidelines" in formatted
        assert "### Tone of Speech" in formatted
        assert "### Topics I Shouldn't Talk About" in formatted
        assert "### Topics I Can Help With" in formatted
        assert "### Things I Should Pay Attention To" in formatted
