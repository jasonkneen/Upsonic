"""
Smoke Test: Cultural Knowledge with STRING Input (NO STORAGE)

Scenario: User provides cultural_knowledge as a STRING without any storage backend
- The string is processed and converted to structured CulturalKnowledge
- No stored cultural knowledge (no storage)
- Injected into the system prompt

Tests that cultural knowledge works properly without storage.
"""

from upsonic import Agent, Task
from upsonic.culture import CulturalKnowledge, CultureManager


def test_culture_string_no_storage():
    """Test cultural knowledge with STRING input without storage."""
    print("=" * 70)
    print("SCENARIO: Cultural Knowledge with STRING Input (NO STORAGE)")
    print("=" * 70)

    # Create agent with STRING cultural_knowledge (NO MEMORY/STORAGE)
    agent = Agent(
        model="openai/gpt-4o",
        cultural_knowledge="I want my agent to act like a friendly hotel concierge who is always welcoming and helpful"
    )

    cm = agent._culture_manager

    # ============================================================================
    # TEST CULTUREMANAGER ATTRIBUTES
    # ============================================================================

    print("\n--- Testing CultureManager Attributes (STRING Input, NO STORAGE) ---\n")

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

    # 5. _cultural_knowledge (should have basic fallback)
    print(f"5. _cultural_knowledge: {cm._cultural_knowledge}")
    assert cm._cultural_knowledge is not None, "_cultural_knowledge should be set"
    assert cm._cultural_knowledge.name == "User Cultural Guidelines", \
        "_cultural_knowledge name should be 'User Cultural Guidelines' for string input"

    # 6. _pending_string_input
    print(f"6. _pending_string_input: {cm._pending_string_input[:50]}...")
    assert cm._pending_string_input is not None, "_pending_string_input should be set for string input"
    assert "hotel concierge" in cm._pending_string_input, \
        "_pending_string_input should contain original string"

    # 7. _is_instance_input
    print(f"7. _is_instance_input: {cm._is_instance_input}")
    assert cm._is_instance_input == False, "_is_instance_input should be False for string input"

    # 8. _stored_knowledge (should be empty - no storage)
    print(f"8. _stored_knowledge: {cm._stored_knowledge}")
    assert cm._stored_knowledge == [], "_stored_knowledge should be empty (no storage)"

    # 9. _knowledge_updated
    print(f"9. _knowledge_updated: {cm._knowledge_updated}")
    assert cm._knowledge_updated == True, "_knowledge_updated should be True after set_cultural_knowledge"

    # 10. _prepared
    print(f"10. _prepared: {cm._prepared}")
    assert cm._prepared == False, "_prepared should be False before aprepare() is called"

    # ============================================================================
    # TEST PROPERTIES
    # ============================================================================

    print("\n--- Testing CultureManager Properties ---\n")

    # 11. cultural_knowledge property
    print(f"11. cultural_knowledge property: {cm.cultural_knowledge.name}")
    assert cm.cultural_knowledge is not None, "cultural_knowledge property should return value"

    # 12. stored_knowledge property (should be empty)
    print(f"12. stored_knowledge property: {cm.stored_knowledge}")
    assert cm.stored_knowledge == [], "stored_knowledge should be empty (no storage)"

    # 13. knowledge_updated property
    print(f"13. knowledge_updated property: {cm.knowledge_updated}")
    assert cm.knowledge_updated == True, "knowledge_updated property should return True"

    # 14. has_pending_input property
    print(f"14. has_pending_input property: {cm.has_pending_input}")
    assert cm.has_pending_input == True, "has_pending_input should be True for string input before aprepare"

    # ============================================================================
    # TEST METHODS
    # ============================================================================

    print("\n--- Testing CultureManager Methods ---\n")

    # 15. get_combined_knowledge() - should return only user knowledge (no stored)
    combined = cm.get_combined_knowledge()
    print(f"15. get_combined_knowledge(): {len(combined)} items")
    assert len(combined) == 1, "get_combined_knowledge should return only user knowledge (no storage)"
    assert combined[0].name == "User Cultural Guidelines", "Should return user knowledge"

    # 16. format_for_system_prompt()
    formatted = cm.format_for_system_prompt()
    print(f"16. format_for_system_prompt(): {len(formatted)} chars")
    assert formatted is not None, "format_for_system_prompt should return string"
    assert "<CulturalKnowledge>" in formatted, "formatted output should have CulturalKnowledge tags"

    # 17. to_dict()
    cm_dict = cm.to_dict()
    print(f"17. to_dict() keys: {list(cm_dict.keys())}")
    assert "enabled" in cm_dict, "to_dict should contain 'enabled'"
    assert "cultural_knowledge" in cm_dict, "to_dict should contain 'cultural_knowledge'"
    assert "stored_knowledge" in cm_dict, "to_dict should contain 'stored_knowledge'"

    # 18. from_dict()
    restored = CultureManager.from_dict(cm_dict, model="openai/gpt-4o")
    print(f"18. from_dict() restored: enabled={restored.enabled}")
    assert restored.enabled == cm.enabled, "from_dict should restore enabled"

    # ============================================================================
    # RUN TASK (NO STORAGE)
    # ============================================================================

    print("\n--- Running Task with Cultural Knowledge (NO STORAGE) ---\n")

    task = Task("Greet me as a guest arriving at your hotel")
    result = agent.do(task)

    print(f"📝 Task: 'Greet me as a guest arriving at your hotel'")
    print(f"📤 Response: {result}")

    assert result is not None, "Result should not be None"
    assert len(str(result)) > 0, "Result should not be empty"

    # Check _prepared after task execution
    print(f"\n✓ _prepared after task: {cm._prepared}")
    assert cm._prepared == True, "_prepared should be True after task execution"

    # Verify no storage was used
    print(f"\n✓ stored_knowledge after task: {len(cm.stored_knowledge)} items")
    assert len(cm.stored_knowledge) == 0, "stored_knowledge should remain empty (no storage)"

    print("\n" + "=" * 70)
    print("✅ PASSED: Cultural Knowledge with STRING Input (NO STORAGE)")
    print("=" * 70)
