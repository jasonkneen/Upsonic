"""
Smoke Test: Cultural Knowledge with CulturalKnowledge INSTANCE (NO STORAGE)

Scenario: User provides cultural_knowledge as a CulturalKnowledge instance without any storage backend
- The instance is used directly (no processing needed)
- No stored cultural knowledge (no storage)
- Injected into the system prompt

Tests that cultural knowledge works properly without storage.
"""

import uuid
from upsonic import Agent, Task
from upsonic.culture import CulturalKnowledge, CultureManager


def test_culture_instance_no_storage():
    """Test cultural knowledge with INSTANCE input without storage."""
    print("=" * 70)
    print("SCENARIO: Cultural Knowledge with INSTANCE Input (NO STORAGE)")
    print("=" * 70)

    # Create a CulturalKnowledge instance with specific code review standards
    code_review_standards = CulturalKnowledge(
        name="Code Review Standards",
        summary="Focus on maintainability, security, and performance",
        categories=["engineering", "code-review"],
        content=(
            "- Check for security vulnerabilities first\n"
            "- Verify error handling is comprehensive\n"
            "- Ensure code is self-documenting\n"
            "- Suggest performance optimizations where relevant\n"
            "- Follow the DRY principle\n"
            "- Prefer composition over inheritance"
        ),
        notes=["Always be constructive", "Focus on teaching"],
    )

    # Create agent with CulturalKnowledge INSTANCE (NO MEMORY/STORAGE)
    agent = Agent(
        model="openai/gpt-4o",
        cultural_knowledge=code_review_standards
    )

    cm = agent._culture_manager

    # ============================================================================
    # TEST CULTUREMANAGER ATTRIBUTES
    # ============================================================================

    print("\n--- Testing CultureManager Attributes (INSTANCE Input, NO STORAGE) ---\n")

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

    # 5. _cultural_knowledge (should be the provided instance)
    print(f"5. _cultural_knowledge.name: {cm._cultural_knowledge.name}")
    assert cm._cultural_knowledge is not None, "_cultural_knowledge should be set"
    assert cm._cultural_knowledge.name == "Code Review Standards", \
        "_cultural_knowledge name should be 'Code Review Standards'"
    assert cm._cultural_knowledge.summary == "Focus on maintainability, security, and performance", \
        "_cultural_knowledge summary should match"
    assert "engineering" in cm._cultural_knowledge.categories, \
        "_cultural_knowledge categories should include 'engineering'"

    # 6. _pending_string_input (should be None for instance input)
    print(f"6. _pending_string_input: {cm._pending_string_input}")
    assert cm._pending_string_input is None, "_pending_string_input should be None for instance input"

    # 7. _is_instance_input (should be True)
    print(f"7. _is_instance_input: {cm._is_instance_input}")
    assert cm._is_instance_input == True, "_is_instance_input should be True for instance input"

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
    print(f"11. cultural_knowledge.name: {cm.cultural_knowledge.name}")
    assert cm.cultural_knowledge.name == "Code Review Standards", \
        "cultural_knowledge property should return correct name"

    # 12. stored_knowledge property (should be empty)
    print(f"12. stored_knowledge: {cm.stored_knowledge}")
    assert cm.stored_knowledge == [], "stored_knowledge should be empty (no storage)"

    # 13. knowledge_updated property
    print(f"13. knowledge_updated: {cm.knowledge_updated}")
    assert cm.knowledge_updated == True, "knowledge_updated should be True"

    # 14. has_pending_input property (should be False for instance)
    print(f"14. has_pending_input: {cm.has_pending_input}")
    assert cm.has_pending_input == False, "has_pending_input should be False for instance input"

    # ============================================================================
    # TEST METHODS
    # ============================================================================

    print("\n--- Testing CultureManager Methods ---\n")

    # 15. get_combined_knowledge() - should return only user instance (no stored)
    combined = cm.get_combined_knowledge()
    print(f"15. get_combined_knowledge(): {len(combined)} items")
    assert len(combined) == 1, "get_combined_knowledge should return only user knowledge (no storage)"
    assert combined[0].name == "Code Review Standards", "First item should be user knowledge"

    # 16. format_for_system_prompt()
    formatted = cm.format_for_system_prompt()
    print(f"16. format_for_system_prompt(): {len(formatted)} chars")
    assert formatted is not None, "format_for_system_prompt should return string"
    assert "<CulturalKnowledge>" in formatted, "formatted should have CulturalKnowledge tags"
    assert "Code Review Standards" in formatted, "formatted should contain user knowledge"

    # 17. to_dict()
    cm_dict = cm.to_dict()
    print(f"17. to_dict() keys: {list(cm_dict.keys())}")
    assert "enabled" in cm_dict, "to_dict should contain 'enabled'"
    assert "cultural_knowledge" in cm_dict, "to_dict should contain 'cultural_knowledge'"
    assert "stored_knowledge" in cm_dict, "to_dict should contain 'stored_knowledge'"

    # Verify cultural_knowledge is serialized properly
    assert cm_dict["cultural_knowledge"]["name"] == "Code Review Standards", \
        "to_dict should serialize cultural_knowledge correctly"

    # 18. from_dict()
    cm_dict = cm.to_dict()
    restored = CultureManager.from_dict(cm_dict, model="openai/gpt-4o")
    print(f"18. from_dict() restored cultural_knowledge.name: {restored.cultural_knowledge.name}")
    assert restored.enabled == cm.enabled, "from_dict should restore enabled"
    assert restored.cultural_knowledge.name == "Code Review Standards", \
        "from_dict should restore cultural_knowledge"

    # ============================================================================
    # RUN TASK (NO STORAGE)
    # ============================================================================

    print("\n--- Running Task with Cultural Knowledge (NO STORAGE) ---\n")

    task = Task("""
    Review this Python code and provide feedback:

    def process_data(data):
    result = []
    for item in data:
        result.append(item * 2)
    return result
    """)
    result = agent.do(task)

    print(f"📝 Task: Review Python code")
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
    print("✅ PASSED: Cultural Knowledge with INSTANCE Input (NO STORAGE)")
    print("=" * 70)
