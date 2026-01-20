import pytest
import uuid
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from upsonic.culture.manager import CultureManager
from upsonic.culture.cultural_knowledge import CulturalKnowledge


class TestCultureManagerInitialization:
    def test_init_defaults(self):
        cm = CultureManager()
        assert cm._model_spec is None
        assert cm.enabled is True
        assert cm.agent_id is None
        assert cm.team_id is None
        assert cm.debug is False
        assert cm.debug_level == 1
        assert cm._cultural_knowledge is None
        assert cm._pending_string_input is None
        assert cm._is_instance_input is False
        assert cm._stored_knowledge == []
        assert cm._knowledge_updated is False
        assert cm._prepared is False

    def test_init_with_all_params(self):
        cm = CultureManager(
            model="openai/gpt-4o",
            enabled=False,
            agent_id="agent_123",
            team_id="team_456",
            debug=True,
            debug_level=2
        )
        assert cm._model_spec == "openai/gpt-4o"
        assert cm.enabled is False
        assert cm.agent_id == "agent_123"
        assert cm.team_id == "team_456"
        assert cm.debug is True
        assert cm.debug_level == 2


class TestCultureManagerProperties:
    def test_cultural_knowledge_property(self):
        cm = CultureManager()
        assert cm.cultural_knowledge is None
        knowledge = CulturalKnowledge(name="Test")
        cm._cultural_knowledge = knowledge
        assert cm.cultural_knowledge == knowledge

    def test_stored_knowledge_property_getter(self):
        cm = CultureManager()
        assert cm.stored_knowledge == []
        knowledge = CulturalKnowledge(name="Test")
        cm._stored_knowledge = [knowledge]
        assert cm.stored_knowledge == [knowledge]

    def test_stored_knowledge_property_setter(self):
        cm = CultureManager()
        knowledge = CulturalKnowledge(name="Test")
        cm.stored_knowledge = [knowledge]
        assert cm._stored_knowledge == [knowledge]

    def test_knowledge_updated_property(self):
        cm = CultureManager()
        assert cm.knowledge_updated is False
        cm._knowledge_updated = True
        assert cm.knowledge_updated is True

    def test_has_pending_input_no_input(self):
        cm = CultureManager()
        assert cm.has_pending_input is False

    def test_has_pending_input_with_string_not_prepared(self):
        cm = CultureManager()
        cm._pending_string_input = "test string"
        cm._prepared = False
        assert cm.has_pending_input is True

    def test_has_pending_input_with_string_prepared(self):
        cm = CultureManager()
        cm._pending_string_input = "test string"
        cm._prepared = True
        assert cm.has_pending_input is False

    def test_has_pending_input_with_instance(self):
        cm = CultureManager()
        knowledge = CulturalKnowledge(name="Test")
        cm._cultural_knowledge = knowledge
        cm._pending_string_input = None
        cm._prepared = False
        assert cm.has_pending_input is False


class TestCultureManagerSetCulturalKnowledge:
    def test_set_cultural_knowledge_with_string(self):
        cm = CultureManager()
        cm.set_cultural_knowledge("Be helpful")
        assert cm._pending_string_input == "Be helpful"
        assert cm._is_instance_input is False
        assert cm._cultural_knowledge is not None
        assert cm._cultural_knowledge.name == "User Cultural Guidelines"
        assert cm._cultural_knowledge.content == "Be helpful"
        assert cm._knowledge_updated is True

    def test_set_cultural_knowledge_with_instance(self):
        cm = CultureManager()
        knowledge = CulturalKnowledge(name="Test", content="Content")
        cm.set_cultural_knowledge(knowledge)
        assert cm._cultural_knowledge == knowledge
        assert cm._pending_string_input is None
        assert cm._is_instance_input is True
        assert cm._knowledge_updated is True

    def test_set_cultural_knowledge_with_instance_no_id(self):
        cm = CultureManager()
        knowledge = CulturalKnowledge(name="Test")
        assert knowledge.id is None
        cm.set_cultural_knowledge(knowledge)
        assert knowledge.id is not None
        assert isinstance(knowledge.id, str)

    def test_set_cultural_knowledge_with_instance_has_id(self):
        cm = CultureManager()
        original_id = str(uuid.uuid4())
        knowledge = CulturalKnowledge(id=original_id, name="Test")
        cm.set_cultural_knowledge(knowledge)
        assert knowledge.id == original_id


class TestCultureManagerGetCombinedKnowledge:
    def test_get_combined_knowledge_empty(self):
        cm = CultureManager()
        result = cm.get_combined_knowledge()
        assert result == []

    def test_get_combined_knowledge_user_only(self):
        cm = CultureManager()
        user_knowledge = CulturalKnowledge(id="user-1", name="User Knowledge")
        cm._cultural_knowledge = user_knowledge
        result = cm.get_combined_knowledge()
        assert len(result) == 1
        assert result[0] == user_knowledge

    def test_get_combined_knowledge_stored_only(self):
        cm = CultureManager()
        stored_knowledge = CulturalKnowledge(id="stored-1", name="Stored Knowledge")
        cm._stored_knowledge = [stored_knowledge]
        result = cm.get_combined_knowledge()
        assert len(result) == 1
        assert result[0] == stored_knowledge

    def test_get_combined_knowledge_user_and_stored(self):
        cm = CultureManager()
        user_knowledge = CulturalKnowledge(id="user-1", name="User Knowledge")
        stored_knowledge = CulturalKnowledge(id="stored-1", name="Stored Knowledge")
        cm._cultural_knowledge = user_knowledge
        cm._stored_knowledge = [stored_knowledge]
        result = cm.get_combined_knowledge()
        assert len(result) == 2
        assert result[0] == user_knowledge
        assert result[1] == stored_knowledge

    def test_get_combined_knowledge_avoids_duplicates(self):
        cm = CultureManager()
        knowledge = CulturalKnowledge(id="same-id", name="Same Knowledge")
        cm._cultural_knowledge = knowledge
        cm._stored_knowledge = [knowledge]
        result = cm.get_combined_knowledge()
        assert len(result) == 1
        assert result[0] == knowledge

    def test_get_combined_knowledge_multiple_stored(self):
        cm = CultureManager()
        user_knowledge = CulturalKnowledge(id="user-1", name="User")
        stored1 = CulturalKnowledge(id="stored-1", name="Stored 1")
        stored2 = CulturalKnowledge(id="stored-2", name="Stored 2")
        cm._cultural_knowledge = user_knowledge
        cm._stored_knowledge = [stored1, stored2]
        result = cm.get_combined_knowledge()
        assert len(result) == 3
        assert result[0] == user_knowledge
        assert result[1] == stored1
        assert result[2] == stored2


class TestCultureManagerFormatForSystemPrompt:
    def test_format_for_system_prompt_empty(self):
        cm = CultureManager()
        result = cm.format_for_system_prompt()
        assert result is None

    def test_format_for_system_prompt_with_knowledge(self):
        cm = CultureManager()
        knowledge = CulturalKnowledge(
            name="Test Knowledge",
            summary="Test summary",
            content="Test content",
            categories=["test"]
        )
        cm._cultural_knowledge = knowledge
        result = cm.format_for_system_prompt()
        assert result is not None
        assert "<CulturalKnowledge>" in result
        assert "</CulturalKnowledge>" in result
        assert "Test Knowledge" in result
        assert "Test summary" in result
        assert "Test content" in result
        assert "test" in result
        assert "Important:" in result
        assert "not required to use all" in result

    def test_format_for_system_prompt_with_multiple_knowledge(self):
        cm = CultureManager()
        knowledge1 = CulturalKnowledge(id="1", name="Knowledge 1", content="Content 1")
        knowledge2 = CulturalKnowledge(id="2", name="Knowledge 2", content="Content 2")
        cm._cultural_knowledge = knowledge1
        cm._stored_knowledge = [knowledge2]
        result = cm.format_for_system_prompt()
        assert result is not None
        assert "Knowledge 1" in result
        assert "Knowledge 2" in result
        assert "Content 1" in result
        assert "Content 2" in result

    def test_format_for_system_prompt_respects_max_length(self):
        cm = CultureManager()
        knowledge = CulturalKnowledge(
            name="Test",
            content="x" * 5000
        )
        cm._cultural_knowledge = knowledge
        result = cm.format_for_system_prompt(max_length=50)
        if result is None:
            assert True
        else:
            assert len(result) < 200

    def test_format_for_system_prompt_includes_notes(self):
        cm = CultureManager()
        knowledge = CulturalKnowledge(
            name="Test",
            notes=["Note 1", "Note 2", "Note 3", "Note 4"]
        )
        cm._cultural_knowledge = knowledge
        result = cm.format_for_system_prompt()
        assert result is not None
        assert "Note 1" in result
        assert "Note 2" in result
        assert "Note 3" in result
        assert "Note 4" not in result

    def test_format_for_system_prompt_missing_fields(self):
        cm = CultureManager()
        knowledge = CulturalKnowledge(name="Test")
        cm._cultural_knowledge = knowledge
        result = cm.format_for_system_prompt()
        assert result is not None
        assert "Test" in result


class TestCultureManagerToDict:
    def test_to_dict_defaults(self):
        cm = CultureManager()
        result = cm.to_dict()
        assert result["enabled"] is True
        assert result["agent_id"] is None
        assert result["team_id"] is None
        assert result["knowledge_updated"] is False
        assert result["prepared"] is False
        assert result["cultural_knowledge"] is None
        assert result["stored_knowledge"] == []

    def test_to_dict_with_knowledge(self):
        cm = CultureManager()
        knowledge = CulturalKnowledge(name="Test", content="Content")
        cm._cultural_knowledge = knowledge
        result = cm.to_dict()
        assert result["cultural_knowledge"] is not None
        assert result["cultural_knowledge"]["name"] == "Test"

    def test_to_dict_with_stored_knowledge(self):
        cm = CultureManager()
        knowledge = CulturalKnowledge(name="Test")
        cm._stored_knowledge = [knowledge]
        result = cm.to_dict()
        assert len(result["stored_knowledge"]) == 1
        assert result["stored_knowledge"][0]["name"] == "Test"

    def test_to_dict_with_all_fields(self):
        cm = CultureManager(
            enabled=False,
            agent_id="agent_123",
            team_id="team_456"
        )
        cm._knowledge_updated = True
        cm._prepared = True
        knowledge = CulturalKnowledge(name="Test")
        cm._cultural_knowledge = knowledge
        cm._stored_knowledge = [knowledge]
        result = cm.to_dict()
        assert result["enabled"] is False
        assert result["agent_id"] == "agent_123"
        assert result["team_id"] == "team_456"
        assert result["knowledge_updated"] is True
        assert result["prepared"] is True
        assert result["cultural_knowledge"] is not None
        assert len(result["stored_knowledge"]) == 1


class TestCultureManagerFromDict:
    def test_from_dict_minimal(self):
        data = {
            "enabled": True,
            "agent_id": None,
            "team_id": None,
            "knowledge_updated": False,
            "prepared": False,
            "cultural_knowledge": None,
            "stored_knowledge": []
        }
        cm = CultureManager.from_dict(data, model="openai/gpt-4o")
        assert cm.enabled is True
        assert cm.agent_id is None
        assert cm.team_id is None
        assert cm._knowledge_updated is False
        assert cm._prepared is False
        assert cm._cultural_knowledge is None
        assert cm._stored_knowledge == []

    def test_from_dict_with_knowledge(self):
        data = {
            "enabled": True,
            "agent_id": None,
            "team_id": None,
            "knowledge_updated": True,
            "prepared": False,
            "cultural_knowledge": {
                "name": "Test",
                "content": "Content"
            },
            "stored_knowledge": []
        }
        cm = CultureManager.from_dict(data, model="openai/gpt-4o")
        assert cm._cultural_knowledge is not None
        assert cm._cultural_knowledge.name == "Test"
        assert cm._cultural_knowledge.content == "Content"

    def test_from_dict_with_stored_knowledge(self):
        data = {
            "enabled": True,
            "agent_id": None,
            "team_id": None,
            "knowledge_updated": False,
            "prepared": False,
            "cultural_knowledge": None,
            "stored_knowledge": [
                {"name": "Stored 1"},
                {"name": "Stored 2"}
            ]
        }
        cm = CultureManager.from_dict(data, model="openai/gpt-4o")
        assert len(cm._stored_knowledge) == 2
        assert cm._stored_knowledge[0].name == "Stored 1"
        assert cm._stored_knowledge[1].name == "Stored 2"

    def test_from_dict_with_all_fields(self):
        data = {
            "enabled": False,
            "agent_id": "agent_123",
            "team_id": "team_456",
            "knowledge_updated": True,
            "prepared": True,
            "cultural_knowledge": {"name": "Test"},
            "stored_knowledge": [{"name": "Stored"}]
        }
        cm = CultureManager.from_dict(data, model="openai/gpt-4o")
        assert cm.enabled is False
        assert cm.agent_id == "agent_123"
        assert cm.team_id == "team_456"
        assert cm._knowledge_updated is True
        assert cm._prepared is True
        assert cm._cultural_knowledge is not None
        assert len(cm._stored_knowledge) == 1


class TestCultureManagerAprepare:
    @pytest.mark.asyncio
    async def test_aprepare_already_prepared(self):
        cm = CultureManager()
        cm._prepared = True
        await cm.aprepare()
        assert cm._prepared is True

    @pytest.mark.asyncio
    async def test_aprepare_with_instance_input(self):
        cm = CultureManager()
        knowledge = CulturalKnowledge(name="Test")
        cm._cultural_knowledge = knowledge
        cm._is_instance_input = True
        cm._pending_string_input = None
        await cm.aprepare()
        assert cm._prepared is True

    @pytest.mark.asyncio
    async def test_aprepare_with_string_input_no_model(self):
        cm = CultureManager(model=None)
        cm.set_cultural_knowledge("Be helpful")
        await cm.aprepare()
        assert cm._prepared is True
        assert cm._cultural_knowledge is not None

    @pytest.mark.asyncio
    @patch('upsonic.agent.agent.Agent')
    async def test_aprepare_with_string_input_with_model(self, mock_agent_class):
        mock_agent = AsyncMock()
        mock_result = Mock()
        mock_result.name = "Extracted Knowledge"
        mock_result.summary = "Summary"
        mock_result.content = "Content"
        mock_result.categories = ["test"]
        mock_result.notes = ["note"]
        mock_agent.do_async = AsyncMock(return_value=mock_result)
        mock_agent_class.return_value = mock_agent

        cm = CultureManager(model="openai/gpt-4o")
        cm.set_cultural_knowledge("Be helpful")
        await cm.aprepare()
        assert cm._prepared is True
        assert cm._cultural_knowledge is not None
        assert cm._cultural_knowledge.name == "Extracted Knowledge"

    @pytest.mark.asyncio
    @patch('upsonic.agent.agent.Agent')
    async def test_aprepare_with_string_input_extraction_fails(self, mock_agent_class):
        mock_agent = AsyncMock()
        mock_agent.do_async = AsyncMock(return_value=None)
        mock_agent_class.return_value = mock_agent

        cm = CultureManager(model="openai/gpt-4o", debug=True)
        cm.set_cultural_knowledge("Be helpful")
        original_knowledge = cm._cultural_knowledge
        await cm.aprepare()
        assert cm._prepared is True
        assert cm._cultural_knowledge == original_knowledge

    @pytest.mark.asyncio
    @patch('upsonic.agent.agent.Agent')
    async def test_aprepare_with_string_input_exception(self, mock_agent_class):
        mock_agent = AsyncMock()
        mock_agent.do_async = AsyncMock(side_effect=Exception("Test error"))
        mock_agent_class.return_value = mock_agent

        cm = CultureManager(model="openai/gpt-4o", debug=True)
        cm.set_cultural_knowledge("Be helpful")
        original_knowledge = cm._cultural_knowledge
        await cm.aprepare()
        assert cm._prepared is True
        assert cm._cultural_knowledge == original_knowledge


class TestCultureManagerACreateCulturalKnowledge:
    @pytest.mark.asyncio
    @patch('upsonic.agent.agent.Agent')
    async def test_acreate_cultural_knowledge(self, mock_agent_class):
        mock_agent = AsyncMock()
        mock_result = Mock()
        mock_result.name = "Created Knowledge"
        mock_result.summary = "Summary"
        mock_result.content = "Content"
        mock_result.categories = ["test"]
        mock_result.notes = []
        mock_agent.do_async = AsyncMock(return_value=mock_result)
        mock_agent_class.return_value = mock_agent

        cm = CultureManager(model="openai/gpt-4o")
        result = await cm.acreate_cultural_knowledge("Be helpful")
        assert result is not None
        assert result.name == "Created Knowledge"
        assert cm._pending_string_input == "Be helpful"
        assert cm._is_instance_input is False

    @pytest.mark.asyncio
    @patch('upsonic.agent.agent.Agent')
    async def test_acreate_cultural_knowledge_with_existing(self, mock_agent_class):
        mock_agent = AsyncMock()
        mock_result = Mock()
        mock_result.name = "Created Knowledge"
        mock_result.summary = "Summary"
        mock_result.content = "Content"
        mock_result.categories = ["test"]
        mock_result.notes = []
        mock_agent.do_async = AsyncMock(return_value=mock_result)
        mock_agent_class.return_value = mock_agent

        cm = CultureManager(model="openai/gpt-4o")
        existing = CulturalKnowledge(name="Existing")
        result = await cm.acreate_cultural_knowledge("Be helpful", existing_knowledge=[existing])
        assert result is not None
        assert cm._stored_knowledge == [existing]


class TestCultureManagerRoundTrip:
    def test_to_dict_from_dict_roundtrip(self):
        cm1 = CultureManager(
            model="openai/gpt-4o",
            enabled=False,
            agent_id="agent_123",
            team_id="team_456"
        )
        knowledge = CulturalKnowledge(name="Test", content="Content")
        cm1._cultural_knowledge = knowledge
        cm1._stored_knowledge = [knowledge]
        cm1._knowledge_updated = True
        cm1._prepared = True

        data = cm1.to_dict()
        cm2 = CultureManager.from_dict(data, model="openai/gpt-4o")

        assert cm2.enabled == cm1.enabled
        assert cm2.agent_id == cm1.agent_id
        assert cm2.team_id == cm1.team_id
        assert cm2._knowledge_updated == cm1._knowledge_updated
        assert cm2._prepared == cm1._prepared
        assert cm2._cultural_knowledge.name == cm1._cultural_knowledge.name
        assert len(cm2._stored_knowledge) == len(cm1._stored_knowledge)
