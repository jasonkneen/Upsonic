"""
Smoke Test: Cultural Knowledge from Storage Only

Scenario 3: No cultural_knowledge provided, but storage has cultural knowledge
- Agent is created without cultural_knowledge parameter
- Cultural knowledge is loaded from storage via Memory/CultureMemory
- Stored knowledge is injected into the system prompt

Tests behavior when no CultureManager is created (user provides nothing).
"""

import uuid
from upsonic import Agent, Task
from upsonic.storage.memory import Memory
from upsonic.storage.sqlite import SqliteStorage
from upsonic.culture import CulturalKnowledge, CultureManager


def test_culture_from_storage():
    """Test cultural knowledge from storage only (no user input)."""
    # Setup storage and memory with culture_memory_enabled
    storage = SqliteStorage(db_file="test_culture_storage.db")

    memory = Memory(
        storage=storage,
        session_id="culture_storage_session_001",
        user_id="user_123",
        full_session_memory=True,
        culture_memory_enabled=True,
        model="openai/gpt-4o"
    )

    print("=" * 70)
    print("SCENARIO 3: Cultural Knowledge from Storage Only")
    print("=" * 70)

    # ============================================================================
    # FIRST, STORE CULTURAL KNOWLEDGE IN DATABASE
    # ============================================================================

    print("\n--- Storing Cultural Knowledge in Database ---\n")

    customer_service_culture = CulturalKnowledge(
    id=str(uuid.uuid4()),
    name="Customer Service Excellence",
    summary="Always prioritize customer satisfaction and clear communication",
    categories=["customer-service", "communication"],
    content=(
        "- Respond promptly and professionally\n"
        "- Use clear, simple language\n"
        "- Acknowledge customer concerns empathetically\n"
        "- Offer solutions, not excuses\n"
        "- Follow up to ensure satisfaction"
    ),
    notes=["Empathy is key", "Always be solution-oriented"],
    )


    # Use sync method since SqliteStorage is sync
    stored = storage.upsert_cultural_knowledge(customer_service_culture)
    print(f"✓ Stored cultural knowledge: {stored.name if stored else 'Failed'}")
    print(f"  ID: {stored.id if stored else 'N/A'}")
    print(f"  Summary: {stored.summary if stored else 'N/A'}")
    print(f"  Categories: {stored.categories if stored else 'N/A'}")
    assert stored is not None, "Cultural knowledge should be stored"

    # ============================================================================
    # CREATE AGENT WITHOUT cultural_knowledge PARAMETER
    # ============================================================================

    print("\n--- Creating Agent WITHOUT cultural_knowledge parameter ---\n")

    agent = Agent(
    model="openai/gpt-4o",
    memory=memory
    # NOTE: No cultural_knowledge parameter!
    )

    # ============================================================================
    # TEST: NO CULTUREMANAGER WHEN NO INPUT PROVIDED
    # ============================================================================

    print("--- Testing: No CultureManager Created ---\n")

    print(f"1. agent._culture_manager: {agent._culture_manager}")
    assert agent._culture_manager is None, "CultureManager should be None when no cultural_knowledge provided"

    print(f"2. agent._cultural_knowledge_input: {getattr(agent, '_cultural_knowledge_input', 'NOT SET')}")
    assert getattr(agent, '_cultural_knowledge_input', None) is None, \
        "_cultural_knowledge_input should be None"

    # ============================================================================
    # TEST: STANDALONE CULTUREMANAGER (for testing attributes)
    # ============================================================================

    print("\n--- Testing Standalone CultureManager (empty) ---\n")

    cm = CultureManager(
    model="openai/gpt-4o",
    enabled=True,
    agent_id="test_agent",
    team_id="test_team",
    debug=True,
    debug_level=2
    )

    # Test all init parameters
    print(f"3. _model_spec: {cm._model_spec}")
    assert cm._model_spec == "openai/gpt-4o", "_model_spec should be set"

    print(f"4. enabled: {cm.enabled}")
    assert cm.enabled == True, "enabled should be True"

    print(f"5. agent_id: {cm.agent_id}")
    assert cm.agent_id == "test_agent", "agent_id should be 'test_agent'"

    print(f"6. team_id: {cm.team_id}")
    assert cm.team_id == "test_team", "team_id should be 'test_team'"

    print(f"7. debug: {cm.debug}")
    assert cm.debug == True, "debug should be True"

    print(f"8. debug_level: {cm.debug_level}")
    assert cm.debug_level == 2, "debug_level should be 2"

    # Test empty state
    print(f"9. _cultural_knowledge: {cm._cultural_knowledge}")
    assert cm._cultural_knowledge is None, "_cultural_knowledge should be None when nothing set"

    print(f"10. _pending_string_input: {cm._pending_string_input}")
    assert cm._pending_string_input is None, "_pending_string_input should be None"

    print(f"11. _is_instance_input: {cm._is_instance_input}")
    assert cm._is_instance_input == False, "_is_instance_input should be False by default"

    print(f"12. _stored_knowledge: {cm._stored_knowledge}")
    assert cm._stored_knowledge == [], "_stored_knowledge should be empty"

    print(f"13. _knowledge_updated: {cm._knowledge_updated}")
    assert cm._knowledge_updated == False, "_knowledge_updated should be False initially"

    print(f"14. _prepared: {cm._prepared}")
    assert cm._prepared == False, "_prepared should be False"

    # ============================================================================
    # TEST: SET STORED KNOWLEDGE ONLY (no user input)
    # ============================================================================

    print("\n--- Testing: Set Stored Knowledge Only ---\n")

    # Simulate what MemoryManager does - set stored knowledge
    cm.stored_knowledge = [customer_service_culture]

    print(f"15. stored_knowledge after setter: {len(cm.stored_knowledge)} items")
    assert len(cm.stored_knowledge) == 1, "stored_knowledge should have 1 item"

    # Properties when only stored knowledge exists
    print(f"16. cultural_knowledge: {cm.cultural_knowledge}")
    assert cm.cultural_knowledge is None, "cultural_knowledge should still be None (no user input)"

    print(f"17. has_pending_input: {cm.has_pending_input}")
    assert cm.has_pending_input == False, "has_pending_input should be False"

    print(f"18. knowledge_updated: {cm.knowledge_updated}")
    assert cm.knowledge_updated == False, "knowledge_updated should be False (no user change)"

    # get_combined_knowledge should return stored only
    combined = cm.get_combined_knowledge()
    print(f"19. get_combined_knowledge(): {len(combined)} items")
    assert len(combined) == 1, "get_combined_knowledge should return 1 stored item"
    assert combined[0].name == "Customer Service Excellence", \
        "Combined should contain stored knowledge"

    # format_for_system_prompt should work with stored only
    formatted = cm.format_for_system_prompt()
    print(f"20. format_for_system_prompt(): {len(formatted)} chars")
    assert formatted is not None, "format_for_system_prompt should return string"
    assert "Customer Service Excellence" in formatted, \
        "formatted should contain stored knowledge"
    assert "<CulturalKnowledge>" in formatted, "formatted should have tags"

    # to_dict should include stored_knowledge
    cm_dict = cm.to_dict()
    print(f"21. to_dict() stored_knowledge count: {len(cm_dict['stored_knowledge'])}")
    assert len(cm_dict["stored_knowledge"]) == 1, "to_dict should include stored_knowledge"

    # from_dict should restore stored_knowledge
    restored = CultureManager.from_dict(cm_dict, model="openai/gpt-4o")
    print(f"22. from_dict() restored stored_knowledge: {len(restored.stored_knowledge)} items")
    assert len(restored.stored_knowledge) == 1, "from_dict should restore stored_knowledge"
    assert restored.stored_knowledge[0].name == "Customer Service Excellence", \
        "from_dict should restore stored_knowledge correctly"

    # ============================================================================
    # VERIFY STORAGE HAS CULTURAL KNOWLEDGE
    # ============================================================================

    print("\n--- Verifying Storage Contains Cultural Knowledge ---\n")


    # Use sync method since SqliteStorage is sync
    stored_list = storage.get_all_cultural_knowledge(limit=10)
    print(f"23. Cultural knowledge entries in storage: {len(stored_list)}")
    assert len(stored_list) > 0, "Should have cultural knowledge in storage"

    for ck in stored_list:
        print(f"    - {ck.name}: {ck.summary}")

    # ============================================================================
    # RUN TASK (stored knowledge should be loaded by Memory)
    # ============================================================================

    print("\n--- Running Task with Storage-Only Cultural Knowledge ---\n")

    task = Task("A customer is upset about a delayed shipment. How should I respond?")
    result = agent.do(task)

    print(f"📝 Task: 'A customer is upset about a delayed shipment. How should I respond?'")
    print(f"📤 Response: {result}")

    assert result is not None, "Result should not be None"

    print("\n" + "=" * 70)
    print("✅ SCENARIO 3 PASSED: All CultureManager attributes tested for STORAGE-ONLY scenario")
    print("=" * 70)
