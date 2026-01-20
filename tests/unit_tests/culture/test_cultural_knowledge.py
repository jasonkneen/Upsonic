import pytest
import uuid
from datetime import datetime, timezone
from upsonic.culture.cultural_knowledge import CulturalKnowledge, _now_epoch_s, _to_epoch_s, _epoch_to_rfc3339_z


class TestCulturalKnowledgeInitialization:
    def test_init_minimal(self):
        knowledge = CulturalKnowledge(name="Test")
        assert knowledge.name == "Test"
        assert knowledge.id is None
        assert knowledge.content is None
        assert knowledge.summary is None
        assert knowledge.categories is None
        assert knowledge.notes is None
        assert knowledge.metadata is None
        assert knowledge.input is None
        assert knowledge.agent_id is None
        assert knowledge.team_id is None
        assert knowledge.created_at is not None
        assert knowledge.updated_at is not None
        assert knowledge.created_at == knowledge.updated_at

    def test_init_with_all_fields(self):
        knowledge_id = str(uuid.uuid4())
        knowledge = CulturalKnowledge(
            id=knowledge_id,
            name="Code Review Standards",
            summary="Focus on security",
            content="Check for vulnerabilities",
            categories=["engineering", "security"],
            notes=["Always be constructive"],
            metadata={"version": "1.0"},
            input="User input string",
            agent_id="agent_123",
            team_id="team_456"
        )
        assert knowledge.id == knowledge_id
        assert knowledge.name == "Code Review Standards"
        assert knowledge.summary == "Focus on security"
        assert knowledge.content == "Check for vulnerabilities"
        assert knowledge.categories == ["engineering", "security"]
        assert knowledge.notes == ["Always be constructive"]
        assert knowledge.metadata == {"version": "1.0"}
        assert knowledge.input == "User input string"
        assert knowledge.agent_id == "agent_123"
        assert knowledge.team_id == "team_456"

    def test_init_empty_name_raises_error(self):
        with pytest.raises(ValueError, match="name must be a non-empty string"):
            CulturalKnowledge(name="")

    def test_init_whitespace_name_raises_error(self):
        with pytest.raises(ValueError, match="name must be a non-empty string"):
            CulturalKnowledge(name="   ")

    def test_init_auto_generates_timestamps(self):
        before = _now_epoch_s()
        knowledge = CulturalKnowledge(name="Test")
        after = _now_epoch_s()
        assert before <= knowledge.created_at <= after
        assert before <= knowledge.updated_at <= after

    def test_init_with_custom_timestamps(self):
        created = 1000000000
        updated = 2000000000
        knowledge = CulturalKnowledge(
            name="Test",
            created_at=created,
            updated_at=updated
        )
        assert knowledge.created_at == created
        assert knowledge.updated_at == updated

    def test_init_with_datetime_timestamps(self):
        dt = datetime.now(timezone.utc)
        knowledge = CulturalKnowledge(
            name="Test",
            created_at=dt
        )
        assert knowledge.created_at == int(dt.timestamp())
        assert knowledge.updated_at == knowledge.created_at

    def test_init_with_rfc3339_string_timestamp(self):
        dt = datetime.now(timezone.utc)
        rfc3339 = dt.isoformat().replace("+00:00", "Z")
        knowledge = CulturalKnowledge(
            name="Test",
            created_at=rfc3339
        )
        assert abs(knowledge.created_at - int(dt.timestamp())) < 2


class TestCulturalKnowledgeBumpUpdatedAt:
    def test_bump_updated_at(self):
        knowledge = CulturalKnowledge(name="Test")
        original_updated = knowledge.updated_at
        import time
        time.sleep(1)
        knowledge.bump_updated_at()
        assert knowledge.updated_at >= original_updated
        assert knowledge.created_at <= knowledge.updated_at


class TestCulturalKnowledgePreview:
    def test_preview_basic(self):
        knowledge = CulturalKnowledge(
            name="Test",
            summary="Short summary",
            content="Some content"
        )
        preview = knowledge.preview()
        assert preview["id"] is None
        assert preview["name"] == "Test"
        assert preview["summary"] == "Short summary"
        assert preview["content"] == "Some content"

    def test_preview_with_categories(self):
        knowledge = CulturalKnowledge(
            name="Test",
            categories=["engineering", "security"]
        )
        preview = knowledge.preview()
        assert preview["categories"] == ["engineering", "security"]

    def test_preview_truncates_long_summary(self):
        long_summary = "a" * 150
        knowledge = CulturalKnowledge(
            name="Test",
            summary=long_summary
        )
        preview = knowledge.preview()
        assert len(preview["summary"]) == 103
        assert preview["summary"].endswith("...")

    def test_preview_truncates_long_content(self):
        long_content = "b" * 150
        knowledge = CulturalKnowledge(
            name="Test",
            content=long_content
        )
        preview = knowledge.preview()
        assert len(preview["content"]) == 103
        assert preview["content"].endswith("...")

    def test_preview_truncates_long_notes(self):
        knowledge = CulturalKnowledge(
            name="Test",
            notes=["a" * 150, "b" * 150]
        )
        preview = knowledge.preview()
        assert len(preview["notes"][0]) == 103
        assert preview["notes"][0].endswith("...")
        assert len(preview["notes"][1]) == 103
        assert preview["notes"][1].endswith("...")


class TestCulturalKnowledgeToDict:
    def test_to_dict_minimal(self):
        knowledge = CulturalKnowledge(name="Test")
        result = knowledge.to_dict()
        assert result["name"] == "Test"
        assert "id" not in result
        assert "content" not in result
        assert "summary" not in result

    def test_to_dict_with_all_fields(self):
        knowledge = CulturalKnowledge(
            id="test-id",
            name="Test",
            summary="Summary",
            content="Content",
            categories=["cat1"],
            notes=["note1"],
            metadata={"key": "value"},
            input="input",
            agent_id="agent1",
            team_id="team1"
        )
        result = knowledge.to_dict()
        assert result["id"] == "test-id"
        assert result["name"] == "Test"
        assert result["summary"] == "Summary"
        assert result["content"] == "Content"
        assert result["categories"] == ["cat1"]
        assert result["notes"] == ["note1"]
        assert result["metadata"] == {"key": "value"}
        assert result["input"] == "input"
        assert result["agent_id"] == "agent1"
        assert result["team_id"] == "team1"
        assert "created_at" in result
        assert "updated_at" in result

    def test_to_dict_timestamps_rfc3339(self):
        knowledge = CulturalKnowledge(name="Test")
        result = knowledge.to_dict()
        assert isinstance(result["created_at"], str)
        assert result["created_at"].endswith("Z")
        assert isinstance(result["updated_at"], str)
        assert result["updated_at"].endswith("Z")

    def test_to_dict_excludes_none(self):
        knowledge = CulturalKnowledge(name="Test")
        result = knowledge.to_dict()
        assert None not in result.values()


class TestCulturalKnowledgeFromDict:
    def test_from_dict_minimal(self):
        data = {"name": "Test"}
        knowledge = CulturalKnowledge.from_dict(data)
        assert knowledge.name == "Test"
        assert knowledge.created_at is not None
        assert knowledge.updated_at is not None

    def test_from_dict_with_all_fields(self):
        data = {
            "id": "test-id",
            "name": "Test",
            "summary": "Summary",
            "content": "Content",
            "categories": ["cat1"],
            "notes": ["note1"],
            "metadata": {"key": "value"},
            "input": "input",
            "agent_id": "agent1",
            "team_id": "team1",
            "created_at": "2024-01-01T00:00:00Z",
            "updated_at": "2024-01-02T00:00:00Z"
        }
        knowledge = CulturalKnowledge.from_dict(data)
        assert knowledge.id == "test-id"
        assert knowledge.name == "Test"
        assert knowledge.summary == "Summary"
        assert knowledge.content == "Content"
        assert knowledge.categories == ["cat1"]
        assert knowledge.notes == ["note1"]
        assert knowledge.metadata == {"key": "value"}
        assert knowledge.input == "input"
        assert knowledge.agent_id == "agent1"
        assert knowledge.team_id == "team1"

    def test_from_dict_with_epoch_timestamp(self):
        epoch = 1000000000
        data = {"name": "Test", "created_at": epoch}
        knowledge = CulturalKnowledge.from_dict(data)
        assert knowledge.created_at == epoch

    def test_from_dict_with_datetime_object(self):
        dt = datetime.now(timezone.utc)
        data = {"name": "Test", "created_at": dt}
        knowledge = CulturalKnowledge.from_dict(data)
        assert knowledge.created_at == int(dt.timestamp())

    def test_from_dict_with_rfc3339_string(self):
        data = {"name": "Test", "created_at": "2024-01-01T00:00:00Z"}
        knowledge = CulturalKnowledge.from_dict(data)
        assert knowledge.created_at is not None

    def test_from_dict_handles_missing_timestamps(self):
        data = {"name": "Test"}
        knowledge = CulturalKnowledge.from_dict(data)
        assert knowledge.created_at is not None
        assert knowledge.updated_at is not None


class TestCulturalKnowledgeRepr:
    def test_repr(self):
        knowledge = CulturalKnowledge(
            id="test-id",
            name="Test Name",
            categories=["cat1", "cat2"]
        )
        repr_str = repr(knowledge)
        assert "test-id" in repr_str
        assert "Test Name" in repr_str
        assert "cat1" in repr_str
        assert "cat2" in repr_str


class TestCulturalKnowledgeEdgeCases:
    def test_none_name_allowed(self):
        knowledge = CulturalKnowledge(name=None)
        assert knowledge.name is None

    def test_empty_categories_list(self):
        knowledge = CulturalKnowledge(name="Test", categories=[])
        assert knowledge.categories == []

    def test_empty_notes_list(self):
        knowledge = CulturalKnowledge(name="Test", notes=[])
        assert knowledge.notes == []

    def test_empty_metadata_dict(self):
        knowledge = CulturalKnowledge(name="Test", metadata={})
        assert knowledge.metadata == {}

    def test_zero_timestamp(self):
        knowledge = CulturalKnowledge(name="Test", created_at=0)
        assert knowledge.created_at == 0
