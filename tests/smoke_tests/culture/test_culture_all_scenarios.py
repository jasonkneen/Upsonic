"""
Smoke Test: All Cultural Knowledge Scenarios

This test covers all five scenarios for cultural knowledge:
1. STRING input - Processed and combined with storage
2. CulturalKnowledge INSTANCE - Used directly, combined with storage
3. No input - Uses stored cultural knowledge only
4. STRING input - NO STORAGE (only user input)
5. CulturalKnowledge INSTANCE - NO STORAGE (only user input)

Tests ALL CultureManager attributes comprehensively.

Usage:
    python tests/smoke_tests/culture/test_culture_all_scenarios.py
"""

import uuid
from upsonic import Agent, Task
from upsonic.storage.memory import Memory
from upsonic.storage.sqlite import SqliteStorage
from upsonic.culture import CulturalKnowledge, CultureManager


def test_scenario_1_string_input():
    """Scenario 1: User provides cultural_knowledge as a STRING."""
    print("\n" + "=" * 70)
    print("SCENARIO 1: Cultural Knowledge with STRING Input")
    print("=" * 70)
    
    storage = SqliteStorage(db_file="test_culture_all.db")
    memory = Memory(
        storage=storage,
        session_id="scenario_1_session",
        user_id="user_123",
        full_session_memory=True,
        culture_memory_enabled=True,
        model="openai/gpt-4o"
    )
    
    # Create agent with STRING cultural_knowledge
    agent = Agent(
        model="openai/gpt-4o",
        memory=memory,
        cultural_knowledge="Act like a friendly hotel concierge who welcomes guests warmly"
    )
    
    cm = agent._culture_manager
    
    # ============================================================================
    # TEST ALL ATTRIBUTES
    # ============================================================================
    
    print("\n--- Testing All CultureManager Attributes (STRING) ---\n")
    
    # Init parameters
    assert cm._model_spec == "openai/gpt-4o", "_model_spec"
    print(f"✓ _model_spec: {cm._model_spec}")
    
    assert cm.enabled == True, "enabled"
    print(f"✓ enabled: {cm.enabled}")
    
    assert cm.agent_id is None, "agent_id"
    print(f"✓ agent_id: {cm.agent_id}")
    
    assert cm.team_id is None, "team_id"
    print(f"✓ team_id: {cm.team_id}")
    
    assert cm.debug == False, "debug"
    print(f"✓ debug: {cm.debug}")
    
    assert cm.debug_level == 1, "debug_level"
    print(f"✓ debug_level: {cm.debug_level}")
    
    # Internal attributes for STRING input
    assert cm._cultural_knowledge is not None, "_cultural_knowledge"
    assert cm._cultural_knowledge.name == "User Cultural Guidelines", "_cultural_knowledge.name"
    print(f"✓ _cultural_knowledge.name: {cm._cultural_knowledge.name}")
    
    assert cm._pending_string_input is not None, "_pending_string_input"
    print(f"✓ _pending_string_input: {cm._pending_string_input[:30]}...")
    
    assert cm._is_instance_input == False, "_is_instance_input"
    print(f"✓ _is_instance_input: {cm._is_instance_input}")
    
    assert cm._stored_knowledge == [], "_stored_knowledge"
    print(f"✓ _stored_knowledge: {len(cm._stored_knowledge)} items")
    
    assert cm._knowledge_updated == True, "_knowledge_updated"
    print(f"✓ _knowledge_updated: {cm._knowledge_updated}")
    
    assert cm._prepared == False, "_prepared"
    print(f"✓ _prepared: {cm._prepared}")
    
    # Properties
    assert cm.cultural_knowledge is not None, "cultural_knowledge property"
    print(f"✓ cultural_knowledge property: {cm.cultural_knowledge.name}")
    
    assert cm.stored_knowledge == [], "stored_knowledge property"
    print(f"✓ stored_knowledge property: {len(cm.stored_knowledge)} items")
    
    assert cm.knowledge_updated == True, "knowledge_updated property"
    print(f"✓ knowledge_updated property: {cm.knowledge_updated}")
    
    assert cm.has_pending_input == True, "has_pending_input property"
    print(f"✓ has_pending_input property: {cm.has_pending_input}")
    
    # Methods
    combined = cm.get_combined_knowledge()
    assert len(combined) >= 1, "get_combined_knowledge"
    print(f"✓ get_combined_knowledge(): {len(combined)} items")
    
    formatted = cm.format_for_system_prompt()
    assert formatted is not None, "format_for_system_prompt"
    assert "<CulturalKnowledge>" in formatted, "format_for_system_prompt tags"
    print(f"✓ format_for_system_prompt(): {len(formatted)} chars")
    
    cm_dict = cm.to_dict()
    assert "enabled" in cm_dict and "cultural_knowledge" in cm_dict, "to_dict"
    print(f"✓ to_dict(): {len(cm_dict)} keys")
    
    restored = CultureManager.from_dict(cm_dict, model="openai/gpt-4o")
    assert restored.enabled == cm.enabled, "from_dict"
    print(f"✓ from_dict(): restored successfully")
    
    # Run task
    print("\n--- Running Task ---")
    result = agent.do(Task("Greet me as I arrive at the hotel"))
    assert result is not None, "Task result"
    print(f"✓ Task executed: {str(result)[:50]}...")
    
    print("\n✅ SCENARIO 1 PASSED: All attributes tested for STRING input")
    return True


def test_scenario_2_instance_input():
    """Scenario 2: User provides cultural_knowledge as CulturalKnowledge instance."""
    print("\n" + "=" * 70)
    print("SCENARIO 2: Cultural Knowledge with INSTANCE Input")
    print("=" * 70)
    
    storage = SqliteStorage(db_file="test_culture_all.db")
    memory = Memory(
        storage=storage,
        session_id="scenario_2_session",
        user_id="user_123",
        full_session_memory=True,
        culture_memory_enabled=True,
        model="openai/gpt-4o"
    )
    
    # Create CulturalKnowledge instance
    code_review = CulturalKnowledge(
        name="Code Review Standards",
        summary="Focus on security, maintainability, and performance",
        categories=["engineering", "code-review"],
        content=(
            "- Check for security vulnerabilities first\n"
            "- Verify error handling is comprehensive\n"
            "- Ensure code follows DRY principle"
        ),
        notes=["Be constructive", "Focus on learning"],
    )
    
    # Create agent with CulturalKnowledge INSTANCE
    agent = Agent(
        model="openai/gpt-4o",
        memory=memory,
        cultural_knowledge=code_review
    )
    
    cm = agent._culture_manager
    
    # ============================================================================
    # TEST ALL ATTRIBUTES
    # ============================================================================
    
    print("\n--- Testing All CultureManager Attributes (INSTANCE) ---\n")
    
    # Init parameters
    assert cm._model_spec == "openai/gpt-4o", "_model_spec"
    print(f"✓ _model_spec: {cm._model_spec}")
    
    assert cm.enabled == True, "enabled"
    print(f"✓ enabled: {cm.enabled}")
    
    # Internal attributes for INSTANCE input
    assert cm._cultural_knowledge is not None, "_cultural_knowledge"
    assert cm._cultural_knowledge.name == "Code Review Standards", "_cultural_knowledge.name"
    print(f"✓ _cultural_knowledge.name: {cm._cultural_knowledge.name}")
    
    assert cm._cultural_knowledge.summary == "Focus on security, maintainability, and performance", \
        "_cultural_knowledge.summary"
    print(f"✓ _cultural_knowledge.summary: {cm._cultural_knowledge.summary}")
    
    assert "engineering" in cm._cultural_knowledge.categories, "_cultural_knowledge.categories"
    print(f"✓ _cultural_knowledge.categories: {cm._cultural_knowledge.categories}")
    
    assert cm._cultural_knowledge.notes is not None, "_cultural_knowledge.notes"
    print(f"✓ _cultural_knowledge.notes: {cm._cultural_knowledge.notes}")
    
    assert cm._pending_string_input is None, "_pending_string_input should be None"
    print(f"✓ _pending_string_input: {cm._pending_string_input}")
    
    assert cm._is_instance_input == True, "_is_instance_input should be True"
    print(f"✓ _is_instance_input: {cm._is_instance_input}")
    
    assert cm._knowledge_updated == True, "_knowledge_updated"
    print(f"✓ _knowledge_updated: {cm._knowledge_updated}")
    
    assert cm._prepared == False, "_prepared"
    print(f"✓ _prepared: {cm._prepared}")
    
    # Properties
    assert cm.has_pending_input == False, "has_pending_input should be False"
    print(f"✓ has_pending_input: {cm.has_pending_input}")
    
    # Test stored_knowledge setter
    stored = [CulturalKnowledge(id=str(uuid.uuid4()), name="Test Stored", content="Test")]
    cm.stored_knowledge = stored
    assert len(cm.stored_knowledge) == 1, "stored_knowledge setter"
    print(f"✓ stored_knowledge setter: {len(cm.stored_knowledge)} items")
    
    # Methods with combined knowledge
    combined = cm.get_combined_knowledge()
    assert len(combined) == 2, "get_combined_knowledge should have 2 items"
    print(f"✓ get_combined_knowledge(): {len(combined)} items (user + stored)")
    
    formatted = cm.format_for_system_prompt()
    assert "Code Review Standards" in formatted, "format should have user knowledge"
    assert "Test Stored" in formatted, "format should have stored knowledge"
    print(f"✓ format_for_system_prompt(): contains both user and stored")
    
    # to_dict / from_dict
    cm.stored_knowledge = []  # Reset for clean serialization
    cm_dict = cm.to_dict()
    assert cm_dict["cultural_knowledge"]["name"] == "Code Review Standards", "to_dict serialization"
    print(f"✓ to_dict(): serialized cultural_knowledge correctly")
    
    restored = CultureManager.from_dict(cm_dict, model="openai/gpt-4o")
    assert restored.cultural_knowledge.name == "Code Review Standards", "from_dict restoration"
    print(f"✓ from_dict(): restored cultural_knowledge correctly")
    
    # Run task
    print("\n--- Running Task ---")
    result = agent.do(Task("Review this code: def add(a, b): return a + b"))
    assert result is not None, "Task result"
    print(f"✓ Task executed: {str(result)[:50]}...")
    
    print("\n✅ SCENARIO 2 PASSED: All attributes tested for INSTANCE input")
    return True


def test_scenario_3_storage_only():
    """Scenario 3: No cultural_knowledge provided, uses storage only."""
    print("\n" + "=" * 70)
    print("SCENARIO 3: Cultural Knowledge from Storage Only")
    print("=" * 70)
    
    storage = SqliteStorage(db_file="test_culture_all.db")
    memory = Memory(
        storage=storage,
        session_id="scenario_3_session",
        user_id="user_123",
        full_session_memory=True,
        culture_memory_enabled=True,
        model="openai/gpt-4o"
    )
    
    # First, store cultural knowledge
    customer_service = CulturalKnowledge(
        id=str(uuid.uuid4()),
        name="Customer Service Guidelines",
        summary="Always be helpful and empathetic",
        categories=["customer-service"],
        content=(
            "- Respond with empathy\n"
            "- Offer clear solutions\n"
            "- Follow up to ensure satisfaction"
        ),
    )
    
    # Use sync method since SqliteStorage is sync
    stored = storage.upsert_cultural_knowledge(customer_service)
    assert stored is not None, "Cultural knowledge should be stored"
    print(f"\n✓ Stored cultural knowledge: {stored.name}")
    
    # ============================================================================
    # TEST: AGENT WITHOUT cultural_knowledge
    # ============================================================================
    
    print("\n--- Testing Agent WITHOUT cultural_knowledge ---\n")
    
    agent = Agent(
        model="openai/gpt-4o",
        memory=memory
        # NOTE: No cultural_knowledge!
    )
    
    assert agent._culture_manager is None, "_culture_manager should be None"
    print(f"✓ agent._culture_manager: {agent._culture_manager}")
    
    # ============================================================================
    # TEST: STANDALONE CULTUREMANAGER WITH ONLY STORED KNOWLEDGE
    # ============================================================================
    
    print("\n--- Testing Standalone CultureManager (storage only) ---\n")
    
    cm = CultureManager(
        model="openai/gpt-4o",
        enabled=True,
        agent_id="test_agent_123",
        team_id="test_team_456",
        debug=True,
        debug_level=2
    )
    
    # Test all init parameters
    assert cm._model_spec == "openai/gpt-4o", "_model_spec"
    print(f"✓ _model_spec: {cm._model_spec}")
    
    assert cm.enabled == True, "enabled"
    print(f"✓ enabled: {cm.enabled}")
    
    assert cm.agent_id == "test_agent_123", "agent_id"
    print(f"✓ agent_id: {cm.agent_id}")
    
    assert cm.team_id == "test_team_456", "team_id"
    print(f"✓ team_id: {cm.team_id}")
    
    assert cm.debug == True, "debug"
    print(f"✓ debug: {cm.debug}")
    
    assert cm.debug_level == 2, "debug_level"
    print(f"✓ debug_level: {cm.debug_level}")
    
    # Empty state
    assert cm._cultural_knowledge is None, "_cultural_knowledge should be None"
    print(f"✓ _cultural_knowledge: {cm._cultural_knowledge}")
    
    assert cm._pending_string_input is None, "_pending_string_input should be None"
    print(f"✓ _pending_string_input: {cm._pending_string_input}")
    
    assert cm._is_instance_input == False, "_is_instance_input"
    print(f"✓ _is_instance_input: {cm._is_instance_input}")
    
    assert cm._stored_knowledge == [], "_stored_knowledge empty initially"
    print(f"✓ _stored_knowledge: {len(cm._stored_knowledge)} items")
    
    assert cm._knowledge_updated == False, "_knowledge_updated should be False"
    print(f"✓ _knowledge_updated: {cm._knowledge_updated}")
    
    assert cm._prepared == False, "_prepared"
    print(f"✓ _prepared: {cm._prepared}")
    
    # Set stored knowledge only
    cm.stored_knowledge = [customer_service]
    
    # Properties with stored only
    assert cm.cultural_knowledge is None, "cultural_knowledge should be None"
    print(f"✓ cultural_knowledge: {cm.cultural_knowledge}")
    
    assert len(cm.stored_knowledge) == 1, "stored_knowledge should have 1 item"
    print(f"✓ stored_knowledge: {len(cm.stored_knowledge)} items")
    
    assert cm.has_pending_input == False, "has_pending_input should be False"
    print(f"✓ has_pending_input: {cm.has_pending_input}")
    
    assert cm.knowledge_updated == False, "knowledge_updated should be False"
    print(f"✓ knowledge_updated: {cm.knowledge_updated}")
    
    # Methods with stored only
    combined = cm.get_combined_knowledge()
    assert len(combined) == 1, "get_combined_knowledge should return stored only"
    assert combined[0].name == "Customer Service Guidelines", "should contain stored"
    print(f"✓ get_combined_knowledge(): {len(combined)} item (stored only)")
    
    formatted = cm.format_for_system_prompt()
    assert formatted is not None, "format should work with stored only"
    assert "Customer Service Guidelines" in formatted, "format should contain stored"
    print(f"✓ format_for_system_prompt(): {len(formatted)} chars (stored only)")
    
    # to_dict / from_dict with stored only
    cm_dict = cm.to_dict()
    assert cm_dict["cultural_knowledge"] is None, "to_dict cultural_knowledge should be None"
    assert len(cm_dict["stored_knowledge"]) == 1, "to_dict should have stored_knowledge"
    print(f"✓ to_dict(): cultural_knowledge=None, stored_knowledge=1")
    
    restored = CultureManager.from_dict(cm_dict, model="openai/gpt-4o")
    assert restored.cultural_knowledge is None, "from_dict cultural_knowledge should be None"
    assert len(restored.stored_knowledge) == 1, "from_dict should restore stored_knowledge"
    print(f"✓ from_dict(): restored stored_knowledge correctly")
    
    # Run task
    print("\n--- Running Task ---")
    result = agent.do(Task("A customer is unhappy. What should I do?"))
    assert result is not None, "Task result"
    print(f"✓ Task executed: {str(result)[:50]}...")
    
    print("\n✅ SCENARIO 3 PASSED: All attributes tested for STORAGE-ONLY")
    return True


def test_scenario_4_string_no_storage():
    """Scenario 4: User provides cultural_knowledge as a STRING without storage."""
    print("\n" + "=" * 70)
    print("SCENARIO 4: Cultural Knowledge with STRING Input (NO STORAGE)")
    print("=" * 70)
    
    # Create agent with STRING cultural_knowledge (NO MEMORY/STORAGE)
    agent = Agent(
        model="openai/gpt-4o",
        cultural_knowledge="Act like a friendly hotel concierge who welcomes guests warmly"
    )
    
    cm = agent._culture_manager
    
    # ============================================================================
    # TEST ALL ATTRIBUTES
    # ============================================================================
    
    print("\n--- Testing All CultureManager Attributes (STRING, NO STORAGE) ---\n")
    
    # Init parameters
    assert cm._model_spec == "openai/gpt-4o", "_model_spec"
    print(f"✓ _model_spec: {cm._model_spec}")
    
    assert cm.enabled == True, "enabled"
    print(f"✓ enabled: {cm.enabled}")
    
    # Internal attributes for STRING input
    assert cm._cultural_knowledge is not None, "_cultural_knowledge"
    assert cm._cultural_knowledge.name == "User Cultural Guidelines", "_cultural_knowledge.name"
    print(f"✓ _cultural_knowledge.name: {cm._cultural_knowledge.name}")
    
    assert cm._pending_string_input is not None, "_pending_string_input"
    print(f"✓ _pending_string_input: {cm._pending_string_input[:30]}...")
    
    assert cm._is_instance_input == False, "_is_instance_input"
    print(f"✓ _is_instance_input: {cm._is_instance_input}")
    
    assert cm._stored_knowledge == [], "_stored_knowledge (no storage)"
    print(f"✓ _stored_knowledge: {len(cm._stored_knowledge)} items")
    
    assert cm._knowledge_updated == True, "_knowledge_updated"
    print(f"✓ _knowledge_updated: {cm._knowledge_updated}")
    
    assert cm._prepared == False, "_prepared"
    print(f"✓ _prepared: {cm._prepared}")
    
    # Properties
    assert cm.cultural_knowledge is not None, "cultural_knowledge property"
    print(f"✓ cultural_knowledge property: {cm.cultural_knowledge.name}")
    
    assert cm.stored_knowledge == [], "stored_knowledge property (no storage)"
    print(f"✓ stored_knowledge property: {len(cm.stored_knowledge)} items")
    
    assert cm.knowledge_updated == True, "knowledge_updated property"
    print(f"✓ knowledge_updated property: {cm.knowledge_updated}")
    
    assert cm.has_pending_input == True, "has_pending_input property"
    print(f"✓ has_pending_input property: {cm.has_pending_input}")
    
    # Methods
    combined = cm.get_combined_knowledge()
    assert len(combined) == 1, "get_combined_knowledge should return only user knowledge (no storage)"
    print(f"✓ get_combined_knowledge(): {len(combined)} items (user only, no storage)")
    
    formatted = cm.format_for_system_prompt()
    assert formatted is not None, "format_for_system_prompt"
    assert "<CulturalKnowledge>" in formatted, "format_for_system_prompt tags"
    print(f"✓ format_for_system_prompt(): {len(formatted)} chars")
    
    cm_dict = cm.to_dict()
    assert "enabled" in cm_dict and "cultural_knowledge" in cm_dict, "to_dict"
    print(f"✓ to_dict(): {len(cm_dict)} keys")
    
    restored = CultureManager.from_dict(cm_dict, model="openai/gpt-4o")
    assert restored.enabled == cm.enabled, "from_dict"
    print(f"✓ from_dict(): restored successfully")
    
    # Run task
    print("\n--- Running Task (NO STORAGE) ---")
    result = agent.do(Task("Greet me as I arrive at the hotel"))
    assert result is not None, "Task result"
    assert len(str(result)) > 0, "Result should not be empty"
    print(f"✓ Task executed: {str(result)[:50]}...")
    
    # Verify no storage was used
    assert len(cm.stored_knowledge) == 0, "stored_knowledge should remain empty (no storage)"
    print(f"✓ Verified: stored_knowledge remains empty (no storage)")
    

    task2 = Task("Greet me as I arrive at the hotel")
    result2 = agent.do(task2)
    assert result2 is not None, "Task result"
    assert len(str(result2)) > 0, "Result should not be empty"
    print(f"✓ Task executed: {str(result2)[:50]}...")
    print(f"✓ Result: {result2}")
    
    print("\n" + "=" * 70)
    print("✅ SCENARIO 4 PASSED: All attributes tested for STRING input (NO STORAGE)")
    print("=" * 70)
    
    return True


def test_scenario_5_instance_no_storage():
    """Scenario 5: User provides cultural_knowledge as CulturalKnowledge instance without storage."""
    print("\n" + "=" * 70)
    print("SCENARIO 5: Cultural Knowledge with INSTANCE Input (NO STORAGE)")
    print("=" * 70)
    
    # Create CulturalKnowledge instance
    code_review = CulturalKnowledge(
        name="Code Review Standards",
        summary="Focus on security, maintainability, and performance",
        categories=["engineering", "code-review"],
        content=(
            "- Check for security vulnerabilities first\n"
            "- Verify error handling is comprehensive\n"
            "- Ensure code follows DRY principle"
        ),
        notes=["Be constructive", "Focus on learning"],
    )
    
    # Create agent with CulturalKnowledge INSTANCE (NO MEMORY/STORAGE)
    agent = Agent(
        model="openai/gpt-4o",
        cultural_knowledge=code_review
    )
    
    cm = agent._culture_manager
    
    # ============================================================================
    # TEST ALL ATTRIBUTES
    # ============================================================================
    
    print("\n--- Testing All CultureManager Attributes (INSTANCE, NO STORAGE) ---\n")
    
    # Init parameters
    assert cm._model_spec == "openai/gpt-4o", "_model_spec"
    print(f"✓ _model_spec: {cm._model_spec}")
    
    assert cm.enabled == True, "enabled"
    print(f"✓ enabled: {cm.enabled}")
    
    # Internal attributes for INSTANCE input
    assert cm._cultural_knowledge is not None, "_cultural_knowledge"
    assert cm._cultural_knowledge.name == "Code Review Standards", "_cultural_knowledge.name"
    print(f"✓ _cultural_knowledge.name: {cm._cultural_knowledge.name}")
    
    assert cm._cultural_knowledge.summary == "Focus on security, maintainability, and performance", \
        "_cultural_knowledge.summary"
    print(f"✓ _cultural_knowledge.summary: {cm._cultural_knowledge.summary}")
    
    assert "engineering" in cm._cultural_knowledge.categories, "_cultural_knowledge.categories"
    print(f"✓ _cultural_knowledge.categories: {cm._cultural_knowledge.categories}")
    
    assert cm._pending_string_input is None, "_pending_string_input should be None"
    print(f"✓ _pending_string_input: {cm._pending_string_input}")
    
    assert cm._is_instance_input == True, "_is_instance_input should be True"
    print(f"✓ _is_instance_input: {cm._is_instance_input}")
    
    assert cm._stored_knowledge == [], "_stored_knowledge (no storage)"
    print(f"✓ _stored_knowledge: {len(cm._stored_knowledge)} items")
    
    assert cm._knowledge_updated == True, "_knowledge_updated"
    print(f"✓ _knowledge_updated: {cm._knowledge_updated}")
    
    assert cm._prepared == False, "_prepared"
    print(f"✓ _prepared: {cm._prepared}")
    
    # Properties
    assert cm.has_pending_input == False, "has_pending_input should be False"
    print(f"✓ has_pending_input: {cm.has_pending_input}")
    
    assert cm.stored_knowledge == [], "stored_knowledge (no storage)"
    print(f"✓ stored_knowledge: {len(cm.stored_knowledge)} items")
    
    # Methods with user knowledge only (no stored)
    combined = cm.get_combined_knowledge()
    assert len(combined) == 1, "get_combined_knowledge should return only user knowledge (no storage)"
    assert combined[0].name == "Code Review Standards", "should contain user knowledge"
    print(f"✓ get_combined_knowledge(): {len(combined)} items (user only, no storage)")
    
    formatted = cm.format_for_system_prompt()
    assert "Code Review Standards" in formatted, "format should have user knowledge"
    print(f"✓ format_for_system_prompt(): contains user knowledge")
    
    # to_dict / from_dict
    cm_dict = cm.to_dict()
    assert cm_dict["cultural_knowledge"]["name"] == "Code Review Standards", "to_dict serialization"
    print(f"✓ to_dict(): serialized cultural_knowledge correctly")
    
    restored = CultureManager.from_dict(cm_dict, model="openai/gpt-4o")
    assert restored.cultural_knowledge.name == "Code Review Standards", "from_dict restoration"
    print(f"✓ from_dict(): restored cultural_knowledge correctly")
    
    # Run task
    print("\n--- Running Task (NO STORAGE) ---")
    result = agent.do(Task("Review this code: def add(a, b): return a + b"))
    assert result is not None, "Task result"
    assert len(str(result)) > 0, "Result should not be empty"
    print(f"✓ Task executed: {str(result)[:50]}...")
    
    # Verify no storage was used
    assert len(cm.stored_knowledge) == 0, "stored_knowledge should remain empty (no storage)"
    print(f"✓ Verified: stored_knowledge remains empty (no storage)")
    
    task2 = Task("Review this code: def add(a, b): return a + b")
    result2 = agent.do(task2)
    assert result2 is not None, "Task result"
    assert len(str(result2)) > 0, "Result should not be empty"
    print(f"✓ Task executed: {str(result2)[:50]}...")
    print(f"✓ Result: {result2}")
    
    print("\n" + "=" * 70)
    print("✅ SCENARIO 5 PASSED: All attributes tested for INSTANCE input (NO STORAGE)")
    print("=" * 70)
    
    return True


def main():
    """Run all cultural knowledge scenarios."""
    print("\n" + "=" * 70)
    print("CULTURAL KNOWLEDGE - ALL SCENARIOS TEST")
    print("Testing ALL CultureManager Attributes")
    print("Scenarios: 1-3 (with storage), 4-5 (no storage)")
    print("=" * 70)
    
    results = []
    
    try:
        results.append(("Scenario 1: STRING input", test_scenario_1_string_input()))
    except Exception as e:
        print(f"\n❌ SCENARIO 1 FAILED: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Scenario 1: STRING input", False))
    
    try:
        results.append(("Scenario 2: INSTANCE input", test_scenario_2_instance_input()))
    except Exception as e:
        print(f"\n❌ SCENARIO 2 FAILED: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Scenario 2: INSTANCE input", False))
    
    try:
        results.append(("Scenario 3: Storage only", test_scenario_3_storage_only()))
    except Exception as e:
        print(f"\n❌ SCENARIO 3 FAILED: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Scenario 3: Storage only", False))
    
    try:
        results.append(("Scenario 4: STRING input (NO STORAGE)", test_scenario_4_string_no_storage()))
    except Exception as e:
        print(f"\n❌ SCENARIO 4 FAILED: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Scenario 4: STRING input (NO STORAGE)", False))
    
    try:
        results.append(("Scenario 5: INSTANCE input (NO STORAGE)", test_scenario_5_instance_no_storage()))
    except Exception as e:
        print(f"\n❌ SCENARIO 5 FAILED: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Scenario 5: INSTANCE input (NO STORAGE)", False))
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY - CultureManager Attributes Tested:")
    print("=" * 70)
    print("""
Attributes tested per scenario:
  - _model_spec
  - enabled
  - agent_id
  - team_id
  - debug
  - debug_level
  - _cultural_knowledge
  - _pending_string_input
  - _is_instance_input
  - _stored_knowledge
  - _knowledge_updated
  - _prepared

Properties tested:
  - cultural_knowledge (getter)
  - stored_knowledge (getter/setter)
  - knowledge_updated (getter)
  - has_pending_input (getter)

Methods tested:
  - set_cultural_knowledge()
  - get_combined_knowledge()
  - format_for_system_prompt()
  - to_dict()
  - from_dict()
""")
    
    print("=" * 70)
    print("RESULTS:")
    print("=" * 70)
    
    all_passed = True
    for name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{status}: {name}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 70)
    if all_passed:
        print("🎉 ALL SCENARIOS PASSED! All CultureManager attributes verified!")
    else:
        print("⚠️ SOME SCENARIOS FAILED")
    print("=" * 70)
    
    return all_passed


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
