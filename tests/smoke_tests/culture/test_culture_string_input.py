"""
Smoke Test: Cultural Knowledge with STRING Input

Scenario 1: User provides cultural_knowledge as a STRING
- The string is processed and converted to structured CulturalKnowledge
- Combined with any stored cultural knowledge from storage
- Injected into the system prompt

Tests all CultureManager attributes for STRING input scenario.
"""

import uuid
from upsonic import Agent, Task
from upsonic.storage.memory import Memory
from upsonic.storage.sqlite import SqliteStorage
from upsonic.culture import CulturalKnowledge, CultureManager


def test_culture_string_input():
    """Test cultural knowledge with STRING input and storage."""
    # Setup storage and memory with culture_memory_enabled
    storage = SqliteStorage(db_file="test_culture_string.db")

    memory = Memory(
        storage=storage,
        session_id="culture_string_session_001",
        user_id="user_123",
        full_session_memory=True,
        culture_memory_enabled=True,
        model="openai/gpt-4o"
    )

    # Create agent with STRING cultural_knowledge
    agent = Agent(
        model="openai/gpt-4o",
        memory=memory,
        cultural_knowledge="I want my agent to act like a friendly hotel concierge who is always welcoming and helpful"
    )

    print("=" * 70)
    print("SCENARIO 1: Cultural Knowledge with STRING Input")
    print("=" * 70)

    cm = agent._culture_manager

    # ============================================================================
    # TEST ALL CULTUREMANAGER ATTRIBUTES
    # ============================================================================

    print("\n--- Testing CultureManager Attributes (STRING Input) ---\n")

    # 1. _model_spec
    print(f"1. _model_spec: {cm._model_spec}")
    assert cm._model_spec == "openai/gpt-4o", "_model_spec should be set to model"

    # 2. enabled
    print(f"2. enabled: {cm.enabled}")
    assert cm.enabled == True, "enabled should be True by default"

    # 3. agent_id
    print(f"3. agent_id: {cm.agent_id}")
    assert cm.agent_id is None, "agent_id should be None (not set)"

    # 4. team_id
    print(f"4. team_id: {cm.team_id}")
    assert cm.team_id is None, "team_id should be None (not set)"

    # 5. debug
    print(f"5. debug: {cm.debug}")
    assert cm.debug == False, "debug should be False by default"

    # 6. debug_level
    print(f"6. debug_level: {cm.debug_level}")
    assert cm.debug_level == 1, "debug_level should be 1 by default"

    # 7. _cultural_knowledge (should have basic fallback)
    print(f"7. _cultural_knowledge: {cm._cultural_knowledge}")
    assert cm._cultural_knowledge is not None, "_cultural_knowledge should be set"
    assert cm._cultural_knowledge.name == "User Cultural Guidelines", \
        "_cultural_knowledge name should be 'User Cultural Guidelines' for string input"

    # 8. _pending_string_input
    print(f"8. _pending_string_input: {cm._pending_string_input[:50]}...")
    assert cm._pending_string_input is not None, "_pending_string_input should be set for string input"
    assert "hotel concierge" in cm._pending_string_input, \
        "_pending_string_input should contain original string"

    # 9. _is_instance_input
    print(f"9. _is_instance_input: {cm._is_instance_input}")
    assert cm._is_instance_input == False, "_is_instance_input should be False for string input"

    # 10. _stored_knowledge
    print(f"10. _stored_knowledge: {cm._stored_knowledge}")
    assert cm._stored_knowledge == [], "_stored_knowledge should be empty initially"

    # 11. _knowledge_updated
    print(f"11. _knowledge_updated: {cm._knowledge_updated}")
    assert cm._knowledge_updated == True, "_knowledge_updated should be True after set_cultural_knowledge"

    # 12. _prepared
    print(f"12. _prepared: {cm._prepared}")
    assert cm._prepared == False, "_prepared should be False before aprepare() is called"

    # ============================================================================
    # TEST PROPERTIES
    # ============================================================================

    print("\n--- Testing CultureManager Properties ---\n")

    # 13. cultural_knowledge property
    print(f"13. cultural_knowledge property: {cm.cultural_knowledge.name}")
    assert cm.cultural_knowledge is not None, "cultural_knowledge property should return value"

    # 14. stored_knowledge property (getter)
    print(f"14. stored_knowledge property: {cm.stored_knowledge}")
    assert cm.stored_knowledge == [], "stored_knowledge should be empty"

    # 15. stored_knowledge property (setter)
    test_stored = [CulturalKnowledge(id=str(uuid.uuid4()), name="Test", content="Test content")]
    cm.stored_knowledge = test_stored
    print(f"15. stored_knowledge after setter: {len(cm.stored_knowledge)} items")
    assert len(cm.stored_knowledge) == 1, "stored_knowledge setter should work"
    cm.stored_knowledge = []  # Reset

    # 16. knowledge_updated property
    print(f"16. knowledge_updated property: {cm.knowledge_updated}")
    assert cm.knowledge_updated == True, "knowledge_updated property should return True"

    # 17. has_pending_input property
    print(f"17. has_pending_input property: {cm.has_pending_input}")
    assert cm.has_pending_input == True, "has_pending_input should be True for string input before aprepare"

    # ============================================================================
    # TEST METHODS
    # ============================================================================

    print("\n--- Testing CultureManager Methods ---\n")

    # 18. get_combined_knowledge()
    combined = cm.get_combined_knowledge()
    print(f"18. get_combined_knowledge(): {len(combined)} items")
    assert len(combined) >= 1, "get_combined_knowledge should return at least user knowledge"

    # 19. format_for_system_prompt()
    formatted = cm.format_for_system_prompt()
    print(f"19. format_for_system_prompt(): {len(formatted)} chars")
    assert formatted is not None, "format_for_system_prompt should return string"
    assert "<CulturalKnowledge>" in formatted, "formatted output should have CulturalKnowledge tags"

    # 20. to_dict()
    cm_dict = cm.to_dict()
    print(f"20. to_dict() keys: {list(cm_dict.keys())}")
    assert "enabled" in cm_dict, "to_dict should contain 'enabled'"
    assert "cultural_knowledge" in cm_dict, "to_dict should contain 'cultural_knowledge'"
    assert "stored_knowledge" in cm_dict, "to_dict should contain 'stored_knowledge'"
    assert "knowledge_updated" in cm_dict, "to_dict should contain 'knowledge_updated'"
    assert "prepared" in cm_dict, "to_dict should contain 'prepared'"

    # 21. from_dict()
    restored = CultureManager.from_dict(cm_dict, model="openai/gpt-4o")
    print(f"21. from_dict() restored: enabled={restored.enabled}")
    assert restored.enabled == cm.enabled, "from_dict should restore enabled"
    assert restored._knowledge_updated == cm._knowledge_updated, "from_dict should restore knowledge_updated"

    # ============================================================================
    # RUN TASK
    # ============================================================================

    print("\n--- Running Task with Cultural Knowledge ---\n")

    task = Task("Greet me as a guest arriving at your hotel")
    result = agent.do(task)

    print(f"📝 Task: 'Greet me as a guest arriving at your hotel'")
    print(f"📤 Response: {result}")

    assert result is not None, "Result should not be None"

    # Check _prepared after task execution
    print(f"\n✓ _prepared after task: {cm._prepared}")

print("\n" + "=" * 70)
print("✅ SCENARIO 1 PASSED: All CultureManager attributes tested for STRING input")
print("=" * 70)
