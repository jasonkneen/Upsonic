"""
Smoke Test: Cultural Knowledge with CulturalKnowledge INSTANCE

Scenario 2: User provides cultural_knowledge as a CulturalKnowledge instance
- The instance is used directly (no processing needed)
- Combined with any stored cultural knowledge from storage
- Injected into the system prompt

Tests all CultureManager attributes for INSTANCE input scenario.
"""

import uuid
from upsonic import Agent, Task
from upsonic.storage.memory import Memory
from upsonic.storage.sqlite import SqliteStorage
from upsonic.culture import CulturalKnowledge, CultureManager


def test_culture_instance_input():
    """Test cultural knowledge with INSTANCE input and storage."""
    # Setup storage and memory with culture_memory_enabled
    storage = SqliteStorage(db_file="test_culture_instance.db")

    memory = Memory(
        storage=storage,
        session_id="culture_instance_session_001",
        user_id="user_123",
        full_session_memory=True,
        culture_memory_enabled=True,
        model="openai/gpt-4o"
    )

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

    # Create agent with CulturalKnowledge INSTANCE
    agent = Agent(
        model="openai/gpt-4o",
        memory=memory,
        cultural_knowledge=code_review_standards
    )

    print("=" * 70)
    print("SCENARIO 2: Cultural Knowledge with INSTANCE Input")
    print("=" * 70)

    cm = agent._culture_manager

    # ============================================================================
    # TEST ALL CULTUREMANAGER ATTRIBUTES
    # ============================================================================

    print("\n--- Testing CultureManager Attributes (INSTANCE Input) ---\n")

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

    # 7. _cultural_knowledge (should be the provided instance)
    print(f"7. _cultural_knowledge.name: {cm._cultural_knowledge.name}")
    assert cm._cultural_knowledge is not None, "_cultural_knowledge should be set"
    assert cm._cultural_knowledge.name == "Code Review Standards", \
        "_cultural_knowledge name should be 'Code Review Standards'"
    assert cm._cultural_knowledge.summary == "Focus on maintainability, security, and performance", \
        "_cultural_knowledge summary should match"
    assert "engineering" in cm._cultural_knowledge.categories, \
        "_cultural_knowledge categories should include 'engineering'"

    # 8. _pending_string_input (should be None for instance input)
    print(f"8. _pending_string_input: {cm._pending_string_input}")
    assert cm._pending_string_input is None, "_pending_string_input should be None for instance input"

    # 9. _is_instance_input (should be True)
    print(f"9. _is_instance_input: {cm._is_instance_input}")
    assert cm._is_instance_input == True, "_is_instance_input should be True for instance input"

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
    print(f"13. cultural_knowledge.name: {cm.cultural_knowledge.name}")
    assert cm.cultural_knowledge.name == "Code Review Standards", \
        "cultural_knowledge property should return correct name"

    # 14. stored_knowledge property (getter)
    print(f"14. stored_knowledge: {cm.stored_knowledge}")
    assert cm.stored_knowledge == [], "stored_knowledge should be empty"

    # 15. stored_knowledge property (setter)
    test_stored = [CulturalKnowledge(id=str(uuid.uuid4()), name="Test", content="Test content")]
    cm.stored_knowledge = test_stored
    print(f"15. stored_knowledge after setter: {len(cm.stored_knowledge)} items")
    assert len(cm.stored_knowledge) == 1, "stored_knowledge setter should work"

    # 16. knowledge_updated property
    print(f"16. knowledge_updated: {cm.knowledge_updated}")
    assert cm.knowledge_updated == True, "knowledge_updated should be True"

    # 17. has_pending_input property (should be False for instance)
    print(f"17. has_pending_input: {cm.has_pending_input}")
    assert cm.has_pending_input == False, "has_pending_input should be False for instance input"

    # ============================================================================
    # TEST METHODS
    # ============================================================================

    print("\n--- Testing CultureManager Methods ---\n")

    # 18. get_combined_knowledge() - should combine user instance + stored
    combined = cm.get_combined_knowledge()
    print(f"18. get_combined_knowledge(): {len(combined)} items")
    assert len(combined) == 2, "get_combined_knowledge should return user + stored (2 items)"
    assert combined[0].name == "Code Review Standards", "First item should be user knowledge"
    assert combined[1].name == "Test", "Second item should be stored knowledge"

    # 19. format_for_system_prompt()
    formatted = cm.format_for_system_prompt()
    print(f"19. format_for_system_prompt(): {len(formatted)} chars")
    assert formatted is not None, "format_for_system_prompt should return string"
    assert "<CulturalKnowledge>" in formatted, "formatted should have CulturalKnowledge tags"
    assert "Code Review Standards" in formatted, "formatted should contain user knowledge"
    assert "Test" in formatted, "formatted should contain stored knowledge"

    # 20. format_for_system_prompt(max_length) - test max_length
    short_formatted = cm.format_for_system_prompt(max_length=100)
    print(f"20. format_for_system_prompt(max_length=100): {len(short_formatted) if short_formatted else 0} chars")

    # 21. to_dict()
    cm_dict = cm.to_dict()
    print(f"21. to_dict() keys: {list(cm_dict.keys())}")
    assert "enabled" in cm_dict, "to_dict should contain 'enabled'"
    assert "agent_id" in cm_dict, "to_dict should contain 'agent_id'"
    assert "team_id" in cm_dict, "to_dict should contain 'team_id'"
    assert "cultural_knowledge" in cm_dict, "to_dict should contain 'cultural_knowledge'"
    assert "stored_knowledge" in cm_dict, "to_dict should contain 'stored_knowledge'"
    assert "knowledge_updated" in cm_dict, "to_dict should contain 'knowledge_updated'"
    assert "prepared" in cm_dict, "to_dict should contain 'prepared'"

    # Verify cultural_knowledge is serialized properly
    assert cm_dict["cultural_knowledge"]["name"] == "Code Review Standards", \
        "to_dict should serialize cultural_knowledge correctly"

    # 22. from_dict()
    cm.stored_knowledge = []  # Reset for clean test
    cm_dict = cm.to_dict()
    restored = CultureManager.from_dict(cm_dict, model="openai/gpt-4o")
    print(f"22. from_dict() restored cultural_knowledge.name: {restored.cultural_knowledge.name}")
    assert restored.enabled == cm.enabled, "from_dict should restore enabled"
    assert restored._knowledge_updated == cm._knowledge_updated, "from_dict should restore knowledge_updated"
    assert restored._prepared == cm._prepared, "from_dict should restore prepared"
    assert restored.cultural_knowledge.name == "Code Review Standards", \
        "from_dict should restore cultural_knowledge"

    # ============================================================================
    # RUN TASK
    # ============================================================================

    print("\n--- Running Task with Cultural Knowledge ---\n")

    # Reset stored knowledge for clean task
    cm.stored_knowledge = []

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

    # Check _prepared after task execution
    print(f"\n✓ _prepared after task: {cm._prepared}")

    print("\n" + "=" * 70)
    print("✅ SCENARIO 2 PASSED: All CultureManager attributes tested for INSTANCE input")
    print("=" * 70)
