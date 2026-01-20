"""Comprehensive test suite for Cultural Knowledge Storage Methods across all providers.

This test suite verifies ALL cultural knowledge storage methods:
- delete_cultural_knowledge / adelete_cultural_knowledge
- get_cultural_knowledge / aget_cultural_knowledge
- get_all_cultural_knowledge / aget_all_cultural_knowledge
- upsert_cultural_knowledge / aupsert_cultural_knowledge

Storage providers tested:
- SqliteStorage (sync)
- AsyncSqliteStorage (async)
- PostgresStorage (sync)
- AsyncPostgresStorage (async)
- MongoStorage (sync)
- AsyncMongoStorage (async)
- RedisStorage (sync)
- JSONStorage (sync)
- InMemoryStorage (sync)
- Mem0Storage (sync) - optional
- AsyncMem0Storage (async) - optional

Docker compose required for:
- PostgreSQL: postgresql://upsonic_test:test_password@localhost:5432/upsonic_test
- MongoDB: mongodb://upsonic_test:test_password@localhost:27017
- Redis: redis://localhost:6379
"""
import asyncio
import os
import sys
import tempfile
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple, Union

from upsonic.culture.cultural_knowledge import CulturalKnowledge
# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None

# Test result tracking
test_results: List[Dict[str, Any]] = []


def log_test_result(test_name: str, passed: bool, message: str = "") -> None:
    """Log test result."""
    status = "✅ PASSED" if passed else "❌ FAILED"
    result = f"{status}: {test_name}"
    if message:
        result += f" - {message}"
    print(result, flush=True)
    test_results.append({"name": test_name, "passed": passed, "message": message})


def print_separator(title: str) -> None:
    """Print test section separator."""
    print("\n" + "=" * 80, flush=True)
    print(f"  {title}", flush=True)
    print("=" * 80 + "\n", flush=True)


def create_test_cultural_knowledge(
    id: Optional[str] = None,
    name: str = "Test Cultural Knowledge",
    content: Optional[str] = None,
    summary: Optional[str] = None,
    categories: Optional[List[str]] = None,
    notes: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    input_str: Optional[str] = None,
    agent_id: Optional[str] = None,
    team_id: Optional[str] = None,
) -> "CulturalKnowledge":
    """Create a test CulturalKnowledge instance with comprehensive data."""
    from upsonic.culture.cultural_knowledge import CulturalKnowledge
    
    return CulturalKnowledge(
        id=id or str(uuid.uuid4()),
        name=name,
        content=content or "This is the main content of the cultural knowledge. It contains important guidelines and principles.",
        summary=summary or "A brief summary of the cultural knowledge.",
        categories=categories or ["engineering", "best-practices", "guidelines"],
        notes=notes or ["Note 1: Important context", "Note 2: Additional information"],
        metadata=metadata or {"source": "test", "version": "1.0", "author": "test_suite"},
        input=input_str or "Original user input that generated this knowledge",
        agent_id=agent_id or "test_agent_001",
        team_id=team_id or "test_team_001",
    )


def verify_cultural_knowledge_attributes(
    original: "CulturalKnowledge",
    retrieved: Union["CulturalKnowledge", Dict[str, Any]],
    test_name: str,
    check_timestamps: bool = False,
) -> bool:
    """Verify that original and retrieved cultural knowledge have matching attributes."""
    from upsonic.culture.cultural_knowledge import CulturalKnowledge
    
    # Convert dict to CulturalKnowledge if needed
    if isinstance(retrieved, dict):
        retrieved = CulturalKnowledge.from_dict(retrieved)
    
    errors = []
    
    # Check all attributes
    if original.id != retrieved.id:
        errors.append(f"id mismatch: {original.id!r} != {retrieved.id!r}")
    
    if original.name != retrieved.name:
        errors.append(f"name mismatch: {original.name!r} != {retrieved.name!r}")
    
    if original.content != retrieved.content:
        errors.append(f"content mismatch: {original.content!r} != {retrieved.content!r}")
    
    if original.summary != retrieved.summary:
        errors.append(f"summary mismatch: {original.summary!r} != {retrieved.summary!r}")
    
    if original.categories != retrieved.categories:
        errors.append(f"categories mismatch: {original.categories!r} != {retrieved.categories!r}")
    
    if original.notes != retrieved.notes:
        errors.append(f"notes mismatch: {original.notes!r} != {retrieved.notes!r}")
    
    if original.metadata != retrieved.metadata:
        errors.append(f"metadata mismatch: {original.metadata!r} != {retrieved.metadata!r}")
    
    if original.input != retrieved.input:
        errors.append(f"input mismatch: {original.input!r} != {retrieved.input!r}")
    
    if original.agent_id != retrieved.agent_id:
        errors.append(f"agent_id mismatch: {original.agent_id!r} != {retrieved.agent_id!r}")
    
    if original.team_id != retrieved.team_id:
        errors.append(f"team_id mismatch: {original.team_id!r} != {retrieved.team_id!r}")
    
    # Timestamps are handled specially (storage may update updated_at)
    if check_timestamps:
        if original.created_at != retrieved.created_at:
            errors.append(f"created_at mismatch: {original.created_at!r} != {retrieved.created_at!r}")
    
    if errors:
        log_test_result(test_name, False, "; ".join(errors))
        return False
    
    log_test_result(test_name, True)
    return True


# =============================================================================
# SQLITE STORAGE TESTS (Sync)
# =============================================================================

def test_sqlite_storage() -> Tuple[int, int]:
    """Test cultural knowledge methods for SqliteStorage."""
    from upsonic.culture.cultural_knowledge import CulturalKnowledge
    from upsonic.storage.sqlite import SqliteStorage
    
    print_separator("SqliteStorage - Cultural Knowledge Tests")
    
    passed = 0
    failed = 0
    db_file = tempfile.mktemp(suffix=".db")
    
    try:
        storage = SqliteStorage(db_file=db_file)
        
        # Test 1: Upsert cultural knowledge
        test_name = "SQLite: upsert_cultural_knowledge - basic insert"
        try:
            ck = create_test_cultural_knowledge(name="SQLite Test Knowledge")
            result = storage.upsert_cultural_knowledge(ck, deserialize=True)
            
            if result is not None and isinstance(result, CulturalKnowledge):
                if verify_cultural_knowledge_attributes(ck, result, test_name):
                    passed += 1
                else:
                    failed += 1
            else:
                log_test_result(test_name, False, f"Expected CulturalKnowledge, got {type(result)}")
                failed += 1
        except Exception as e:
            log_test_result(test_name, False, str(e))
            failed += 1
        
        # Test 2: Get cultural knowledge by ID
        test_name = "SQLite: get_cultural_knowledge - by ID"
        try:
            retrieved = storage.get_cultural_knowledge(ck.id, deserialize=True)
            
            if retrieved is not None and isinstance(retrieved, CulturalKnowledge):
                if verify_cultural_knowledge_attributes(ck, retrieved, test_name):
                    passed += 1
                else:
                    failed += 1
            else:
                log_test_result(test_name, False, f"Expected CulturalKnowledge, got {type(retrieved)}")
                failed += 1
        except Exception as e:
            log_test_result(test_name, False, str(e))
            failed += 1
        
        # Test 3: Get cultural knowledge with deserialize=False
        test_name = "SQLite: get_cultural_knowledge - deserialize=False"
        try:
            retrieved_dict = storage.get_cultural_knowledge(ck.id, deserialize=False)
            
            if isinstance(retrieved_dict, dict):
                if "id" in retrieved_dict and retrieved_dict["id"] == ck.id:
                    log_test_result(test_name, True)
                    passed += 1
                else:
                    log_test_result(test_name, False, f"ID mismatch or missing: {retrieved_dict}")
                    failed += 1
            else:
                log_test_result(test_name, False, f"Expected dict, got {type(retrieved_dict)}")
                failed += 1
        except Exception as e:
            log_test_result(test_name, False, str(e))
            failed += 1
        
        # Test 4: Update cultural knowledge
        test_name = "SQLite: upsert_cultural_knowledge - update existing"
        try:
            ck.content = "Updated content for testing"
            ck.summary = "Updated summary"
            ck.categories = ["updated", "categories"]
            result = storage.upsert_cultural_knowledge(ck, deserialize=True)
            
            if result is not None and result.content == "Updated content for testing":
                log_test_result(test_name, True)
                passed += 1
            else:
                log_test_result(test_name, False, f"Content not updated: {result.content if result else 'None'}")
                failed += 1
        except Exception as e:
            log_test_result(test_name, False, str(e))
            failed += 1
        
        # Test 5: Get all cultural knowledge
        test_name = "SQLite: get_all_cultural_knowledge - basic"
        try:
            # Add more entries
            ck2 = create_test_cultural_knowledge(name="SQLite Test Knowledge 2", agent_id="agent_002")
            ck3 = create_test_cultural_knowledge(name="SQLite Test Knowledge 3", agent_id="agent_003")
            storage.upsert_cultural_knowledge(ck2)
            storage.upsert_cultural_knowledge(ck3)
            
            all_ck = storage.get_all_cultural_knowledge(deserialize=True)
            
            if isinstance(all_ck, list) and len(all_ck) >= 3:
                log_test_result(test_name, True, f"Retrieved {len(all_ck)} entries")
                passed += 1
            else:
                log_test_result(test_name, False, f"Expected list with >= 3 entries, got {type(all_ck)} with {len(all_ck) if isinstance(all_ck, list) else 'N/A'}")
                failed += 1
        except Exception as e:
            log_test_result(test_name, False, str(e))
            failed += 1
        
        # Test 6: Get all with filter by name
        test_name = "SQLite: get_all_cultural_knowledge - filter by name"
        try:
            filtered = storage.get_all_cultural_knowledge(name="SQLite Test Knowledge 2", deserialize=True)
            
            if isinstance(filtered, list) and len(filtered) == 1:
                if filtered[0].name == "SQLite Test Knowledge 2":
                    log_test_result(test_name, True)
                    passed += 1
                else:
                    log_test_result(test_name, False, f"Name mismatch: {filtered[0].name}")
                    failed += 1
            else:
                log_test_result(test_name, False, f"Expected 1 entry, got {len(filtered) if isinstance(filtered, list) else 'N/A'}")
                failed += 1
        except Exception as e:
            log_test_result(test_name, False, str(e))
            failed += 1
        
        # Test 7: Get all with filter by agent_id
        test_name = "SQLite: get_all_cultural_knowledge - filter by agent_id"
        try:
            filtered = storage.get_all_cultural_knowledge(agent_id="agent_002", deserialize=True)
            
            if isinstance(filtered, list) and len(filtered) == 1:
                if filtered[0].agent_id == "agent_002":
                    log_test_result(test_name, True)
                    passed += 1
                else:
                    log_test_result(test_name, False, f"agent_id mismatch: {filtered[0].agent_id}")
                    failed += 1
            else:
                log_test_result(test_name, False, f"Expected 1 entry, got {len(filtered) if isinstance(filtered, list) else 'N/A'}")
                failed += 1
        except Exception as e:
            log_test_result(test_name, False, str(e))
            failed += 1
        
        # Test 8: Get all with deserialize=False
        test_name = "SQLite: get_all_cultural_knowledge - deserialize=False"
        try:
            result = storage.get_all_cultural_knowledge(deserialize=False)
            
            if isinstance(result, tuple) and len(result) == 2:
                rows, count = result
                if isinstance(rows, list) and all(isinstance(r, dict) for r in rows):
                    log_test_result(test_name, True, f"Got {len(rows)} dicts, total count: {count}")
                    passed += 1
                else:
                    log_test_result(test_name, False, f"Expected list of dicts, got {type(rows)}")
                    failed += 1
            else:
                log_test_result(test_name, False, f"Expected tuple (list, int), got {type(result)}")
                failed += 1
        except Exception as e:
            log_test_result(test_name, False, str(e))
            failed += 1
        
        # Test 9: Get all with pagination
        test_name = "SQLite: get_all_cultural_knowledge - pagination"
        try:
            page1 = storage.get_all_cultural_knowledge(limit=2, page=1, deserialize=True)
            
            if isinstance(page1, list) and len(page1) == 2:
                log_test_result(test_name, True, f"Page 1 returned {len(page1)} entries")
                passed += 1
            else:
                log_test_result(test_name, False, f"Expected 2 entries, got {len(page1) if isinstance(page1, list) else 'N/A'}")
                failed += 1
        except Exception as e:
            log_test_result(test_name, False, str(e))
            failed += 1
        
        # Test 10: Get all with sorting
        test_name = "SQLite: get_all_cultural_knowledge - sorting"
        try:
            sorted_asc = storage.get_all_cultural_knowledge(sort_by="name", sort_order="asc", deserialize=True)
            sorted_desc = storage.get_all_cultural_knowledge(sort_by="name", sort_order="desc", deserialize=True)
            
            if isinstance(sorted_asc, list) and isinstance(sorted_desc, list):
                if len(sorted_asc) > 1 and sorted_asc[0].name != sorted_desc[0].name:
                    log_test_result(test_name, True)
                    passed += 1
                else:
                    log_test_result(test_name, False, "Sorting doesn't appear to work")
                    failed += 1
            else:
                log_test_result(test_name, False, "Expected lists")
                failed += 1
        except Exception as e:
            log_test_result(test_name, False, str(e))
            failed += 1
        
        # Test 11: Delete cultural knowledge
        test_name = "SQLite: delete_cultural_knowledge"
        try:
            storage.delete_cultural_knowledge(ck2.id)
            deleted = storage.get_cultural_knowledge(ck2.id)
            
            if deleted is None:
                log_test_result(test_name, True)
                passed += 1
            else:
                log_test_result(test_name, False, "Entry still exists after deletion")
                failed += 1
        except Exception as e:
            log_test_result(test_name, False, str(e))
            failed += 1
        
        # Test 12: Get non-existent cultural knowledge
        test_name = "SQLite: get_cultural_knowledge - non-existent ID"
        try:
            result = storage.get_cultural_knowledge("non_existent_id_12345")
            
            if result is None:
                log_test_result(test_name, True)
                passed += 1
            else:
                log_test_result(test_name, False, f"Expected None, got {type(result)}")
                failed += 1
        except Exception as e:
            log_test_result(test_name, False, str(e))
            failed += 1
        
        # Cleanup
        storage.close()
        
    except Exception as e:
        log_test_result("SQLite: Setup/Teardown", False, str(e))
        failed += 1
    finally:
        if os.path.exists(db_file):
            try:
                os.remove(db_file)
            except Exception:
                pass
    
    return passed, failed


# =============================================================================
# ASYNC SQLITE STORAGE TESTS
# =============================================================================

import pytest

@pytest.mark.asyncio
async def test_async_sqlite_storage() -> Tuple[int, int]:
    """Test cultural knowledge methods for AsyncSqliteStorage."""
    from upsonic.culture.cultural_knowledge import CulturalKnowledge
    from upsonic.storage.sqlite import AsyncSqliteStorage
    
    print_separator("AsyncSqliteStorage - Cultural Knowledge Tests")
    
    passed = 0
    failed = 0
    db_file = tempfile.mktemp(suffix=".db")
    
    try:
        storage = AsyncSqliteStorage(db_file=db_file)
        
        # Test 1: Upsert cultural knowledge
        test_name = "AsyncSQLite: aupsert_cultural_knowledge - basic insert"
        try:
            ck = create_test_cultural_knowledge(name="AsyncSQLite Test Knowledge")
            result = await storage.aupsert_cultural_knowledge(ck, deserialize=True)
            
            if result is not None and isinstance(result, CulturalKnowledge):
                if verify_cultural_knowledge_attributes(ck, result, test_name):
                    passed += 1
                else:
                    failed += 1
            else:
                log_test_result(test_name, False, f"Expected CulturalKnowledge, got {type(result)}")
                failed += 1
        except Exception as e:
            log_test_result(test_name, False, str(e))
            failed += 1
        
        # Test 2: Get cultural knowledge by ID
        test_name = "AsyncSQLite: aget_cultural_knowledge - by ID"
        try:
            retrieved = await storage.aget_cultural_knowledge(ck.id, deserialize=True)
            
            if retrieved is not None and isinstance(retrieved, CulturalKnowledge):
                if verify_cultural_knowledge_attributes(ck, retrieved, test_name):
                    passed += 1
                else:
                    failed += 1
            else:
                log_test_result(test_name, False, f"Expected CulturalKnowledge, got {type(retrieved)}")
                failed += 1
        except Exception as e:
            log_test_result(test_name, False, str(e))
            failed += 1
        
        # Test 3: Get cultural knowledge with deserialize=False
        test_name = "AsyncSQLite: aget_cultural_knowledge - deserialize=False"
        try:
            retrieved_dict = await storage.aget_cultural_knowledge(ck.id, deserialize=False)
            
            if isinstance(retrieved_dict, dict):
                if "id" in retrieved_dict and retrieved_dict["id"] == ck.id:
                    log_test_result(test_name, True)
                    passed += 1
                else:
                    log_test_result(test_name, False, f"ID mismatch or missing")
                    failed += 1
            else:
                log_test_result(test_name, False, f"Expected dict, got {type(retrieved_dict)}")
                failed += 1
        except Exception as e:
            log_test_result(test_name, False, str(e))
            failed += 1
        
        # Test 4: Get all cultural knowledge
        test_name = "AsyncSQLite: aget_all_cultural_knowledge - basic"
        try:
            ck2 = create_test_cultural_knowledge(name="AsyncSQLite Test Knowledge 2")
            await storage.aupsert_cultural_knowledge(ck2)
            
            all_ck = await storage.aget_all_cultural_knowledge(deserialize=True)
            
            if isinstance(all_ck, list) and len(all_ck) >= 2:
                log_test_result(test_name, True, f"Retrieved {len(all_ck)} entries")
                passed += 1
            else:
                log_test_result(test_name, False, f"Expected list with >= 2 entries")
                failed += 1
        except Exception as e:
            log_test_result(test_name, False, str(e))
            failed += 1
        
        # Test 5: Get all with deserialize=False
        test_name = "AsyncSQLite: aget_all_cultural_knowledge - deserialize=False"
        try:
            result = await storage.aget_all_cultural_knowledge(deserialize=False)
            
            if isinstance(result, tuple) and len(result) == 2:
                rows, count = result
                if isinstance(rows, list) and all(isinstance(r, dict) for r in rows):
                    log_test_result(test_name, True, f"Got {len(rows)} dicts, total count: {count}")
                    passed += 1
                else:
                    log_test_result(test_name, False, f"Expected list of dicts")
                    failed += 1
            else:
                log_test_result(test_name, False, f"Expected tuple (list, int)")
                failed += 1
        except Exception as e:
            log_test_result(test_name, False, str(e))
            failed += 1
        
        # Test 6: Delete cultural knowledge
        test_name = "AsyncSQLite: adelete_cultural_knowledge"
        try:
            await storage.adelete_cultural_knowledge(ck2.id)
            deleted = await storage.aget_cultural_knowledge(ck2.id)
            
            if deleted is None:
                log_test_result(test_name, True)
                passed += 1
            else:
                log_test_result(test_name, False, "Entry still exists after deletion")
                failed += 1
        except Exception as e:
            log_test_result(test_name, False, str(e))
            failed += 1
        
        # Cleanup
        await storage.close()
        
    except Exception as e:
        log_test_result("AsyncSQLite: Setup/Teardown", False, str(e))
        failed += 1
    finally:
        if os.path.exists(db_file):
            try:
                os.remove(db_file)
            except Exception:
                pass
    
    return passed, failed


# =============================================================================
# IN-MEMORY STORAGE TESTS
# =============================================================================

def test_in_memory_storage() -> Tuple[int, int]:
    """Test cultural knowledge methods for InMemoryStorage."""
    from upsonic.culture.cultural_knowledge import CulturalKnowledge
    from upsonic.storage.in_memory import InMemoryStorage
    
    print_separator("InMemoryStorage - Cultural Knowledge Tests")
    
    passed = 0
    failed = 0
    
    try:
        storage = InMemoryStorage()
        
        # Test 1: Upsert cultural knowledge
        test_name = "InMemory: upsert_cultural_knowledge - basic insert"
        try:
            ck = create_test_cultural_knowledge(name="InMemory Test Knowledge")
            result = storage.upsert_cultural_knowledge(ck, deserialize=True)
            
            if result is not None and isinstance(result, CulturalKnowledge):
                if verify_cultural_knowledge_attributes(ck, result, test_name):
                    passed += 1
                else:
                    failed += 1
            else:
                log_test_result(test_name, False, f"Expected CulturalKnowledge, got {type(result)}")
                failed += 1
        except Exception as e:
            log_test_result(test_name, False, str(e))
            failed += 1
        
        # Test 2: Get cultural knowledge by ID
        test_name = "InMemory: get_cultural_knowledge - by ID"
        try:
            retrieved = storage.get_cultural_knowledge(ck.id, deserialize=True)
            
            if retrieved is not None and isinstance(retrieved, CulturalKnowledge):
                if verify_cultural_knowledge_attributes(ck, retrieved, test_name):
                    passed += 1
                else:
                    failed += 1
            else:
                log_test_result(test_name, False, f"Expected CulturalKnowledge, got {type(retrieved)}")
                failed += 1
        except Exception as e:
            log_test_result(test_name, False, str(e))
            failed += 1
        
        # Test 3: Get cultural knowledge with deserialize=False
        test_name = "InMemory: get_cultural_knowledge - deserialize=False"
        try:
            retrieved_dict = storage.get_cultural_knowledge(ck.id, deserialize=False)
            
            if isinstance(retrieved_dict, dict):
                if "id" in retrieved_dict and retrieved_dict["id"] == ck.id:
                    log_test_result(test_name, True)
                    passed += 1
                else:
                    log_test_result(test_name, False, f"ID mismatch or missing")
                    failed += 1
            else:
                log_test_result(test_name, False, f"Expected dict, got {type(retrieved_dict)}")
                failed += 1
        except Exception as e:
            log_test_result(test_name, False, str(e))
            failed += 1
        
        # Test 4: Get all cultural knowledge
        test_name = "InMemory: get_all_cultural_knowledge - basic"
        try:
            ck2 = create_test_cultural_knowledge(name="InMemory Test Knowledge 2", agent_id="agent_002")
            ck3 = create_test_cultural_knowledge(name="InMemory Test Knowledge 3", team_id="team_002")
            storage.upsert_cultural_knowledge(ck2)
            storage.upsert_cultural_knowledge(ck3)
            
            all_ck = storage.get_all_cultural_knowledge(deserialize=True)
            
            if isinstance(all_ck, list) and len(all_ck) >= 3:
                log_test_result(test_name, True, f"Retrieved {len(all_ck)} entries")
                passed += 1
            else:
                log_test_result(test_name, False, f"Expected list with >= 3 entries")
                failed += 1
        except Exception as e:
            log_test_result(test_name, False, str(e))
            failed += 1
        
        # Test 5: Get all with filter
        test_name = "InMemory: get_all_cultural_knowledge - filter by agent_id"
        try:
            filtered = storage.get_all_cultural_knowledge(agent_id="agent_002", deserialize=True)
            
            if isinstance(filtered, list) and len(filtered) == 1:
                log_test_result(test_name, True)
                passed += 1
            else:
                log_test_result(test_name, False, f"Expected 1 entry")
                failed += 1
        except Exception as e:
            log_test_result(test_name, False, str(e))
            failed += 1
        
        # Test 6: Get all with deserialize=False
        test_name = "InMemory: get_all_cultural_knowledge - deserialize=False"
        try:
            result = storage.get_all_cultural_knowledge(deserialize=False)
            
            if isinstance(result, tuple) and len(result) == 2:
                rows, count = result
                if isinstance(rows, list) and all(isinstance(r, dict) for r in rows):
                    log_test_result(test_name, True, f"Got {len(rows)} dicts, total: {count}")
                    passed += 1
                else:
                    log_test_result(test_name, False, f"Expected list of dicts")
                    failed += 1
            else:
                log_test_result(test_name, False, f"Expected tuple")
                failed += 1
        except Exception as e:
            log_test_result(test_name, False, str(e))
            failed += 1
        
        # Test 7: Delete cultural knowledge
        test_name = "InMemory: delete_cultural_knowledge"
        try:
            storage.delete_cultural_knowledge(ck2.id)
            deleted = storage.get_cultural_knowledge(ck2.id)
            
            if deleted is None:
                log_test_result(test_name, True)
                passed += 1
            else:
                log_test_result(test_name, False, "Entry still exists")
                failed += 1
        except Exception as e:
            log_test_result(test_name, False, str(e))
            failed += 1
        
        # Cleanup
        storage.close()
        
    except Exception as e:
        log_test_result("InMemory: Setup/Teardown", False, str(e))
        failed += 1
    
    return passed, failed


# =============================================================================
# JSON STORAGE TESTS
# =============================================================================

def test_json_storage() -> Tuple[int, int]:
    """Test cultural knowledge methods for JSONStorage."""
    from upsonic.culture.cultural_knowledge import CulturalKnowledge
    from upsonic.storage.json import JSONStorage
    
    print_separator("JSONStorage - Cultural Knowledge Tests")
    
    passed = 0
    failed = 0
    json_dir = tempfile.mkdtemp()
    json_file = os.path.join(json_dir, "test_storage.json")
    
    try:
        storage = JSONStorage(db_path=json_file)
        
        # Test 1: Upsert cultural knowledge
        test_name = "JSON: upsert_cultural_knowledge - basic insert"
        try:
            ck = create_test_cultural_knowledge(name="JSON Test Knowledge")
            result = storage.upsert_cultural_knowledge(ck, deserialize=True)
            
            if result is not None and isinstance(result, CulturalKnowledge):
                if verify_cultural_knowledge_attributes(ck, result, test_name):
                    passed += 1
                else:
                    failed += 1
            else:
                log_test_result(test_name, False, f"Expected CulturalKnowledge, got {type(result)}")
                failed += 1
        except Exception as e:
            log_test_result(test_name, False, str(e))
            failed += 1
        
        # Test 2: Get cultural knowledge by ID
        test_name = "JSON: get_cultural_knowledge - by ID"
        try:
            retrieved = storage.get_cultural_knowledge(ck.id, deserialize=True)
            
            if retrieved is not None and isinstance(retrieved, CulturalKnowledge):
                if verify_cultural_knowledge_attributes(ck, retrieved, test_name):
                    passed += 1
                else:
                    failed += 1
            else:
                log_test_result(test_name, False, f"Expected CulturalKnowledge, got {type(retrieved)}")
                failed += 1
        except Exception as e:
            log_test_result(test_name, False, str(e))
            failed += 1
        
        # Test 3: Get all cultural knowledge
        test_name = "JSON: get_all_cultural_knowledge - basic"
        try:
            ck2 = create_test_cultural_knowledge(name="JSON Test Knowledge 2")
            storage.upsert_cultural_knowledge(ck2)
            
            all_ck = storage.get_all_cultural_knowledge(deserialize=True)
            
            if isinstance(all_ck, list) and len(all_ck) >= 2:
                log_test_result(test_name, True, f"Retrieved {len(all_ck)} entries")
                passed += 1
            else:
                log_test_result(test_name, False, f"Expected list with >= 2 entries")
                failed += 1
        except Exception as e:
            log_test_result(test_name, False, str(e))
            failed += 1
        
        # Test 4: Delete cultural knowledge
        test_name = "JSON: delete_cultural_knowledge"
        try:
            storage.delete_cultural_knowledge(ck2.id)
            deleted = storage.get_cultural_knowledge(ck2.id)
            
            if deleted is None:
                log_test_result(test_name, True)
                passed += 1
            else:
                log_test_result(test_name, False, "Entry still exists")
                failed += 1
        except Exception as e:
            log_test_result(test_name, False, str(e))
            failed += 1
        
        # Cleanup
        storage.close()
        
    except Exception as e:
        log_test_result("JSON: Setup/Teardown", False, str(e))
        failed += 1
    finally:
        import shutil
        try:
            shutil.rmtree(json_dir)
        except Exception:
            pass
    
    return passed, failed


# =============================================================================
# POSTGRES STORAGE TESTS (Sync)
# =============================================================================

def test_postgres_storage() -> Tuple[int, int]:
    """Test cultural knowledge methods for PostgresStorage."""
    from upsonic.culture.cultural_knowledge import CulturalKnowledge
    
    print_separator("PostgresStorage - Cultural Knowledge Tests")
    
    passed = 0
    failed = 0
    
    # Get connection URL from environment or use docker-compose defaults
    postgres_url = os.getenv(
        "POSTGRES_URL",
        "postgresql://upsonic_test:test_password@localhost:5432/upsonic_test"
    )
    
    try:
        from upsonic.storage.postgres import PostgresStorage
        
        storage = PostgresStorage(db_url=postgres_url)
        
        # Test 1: Upsert cultural knowledge
        test_name = "Postgres: upsert_cultural_knowledge - basic insert"
        try:
            ck = create_test_cultural_knowledge(name="Postgres Test Knowledge")
            result = storage.upsert_cultural_knowledge(ck, deserialize=True)
            
            if result is not None and isinstance(result, CulturalKnowledge):
                if verify_cultural_knowledge_attributes(ck, result, test_name):
                    passed += 1
                else:
                    failed += 1
            else:
                log_test_result(test_name, False, f"Expected CulturalKnowledge, got {type(result)}")
                failed += 1
        except Exception as e:
            log_test_result(test_name, False, str(e))
            failed += 1
        
        # Test 2: Get cultural knowledge by ID
        test_name = "Postgres: get_cultural_knowledge - by ID"
        try:
            retrieved = storage.get_cultural_knowledge(ck.id, deserialize=True)
            
            if retrieved is not None and isinstance(retrieved, CulturalKnowledge):
                if verify_cultural_knowledge_attributes(ck, retrieved, test_name):
                    passed += 1
                else:
                    failed += 1
            else:
                log_test_result(test_name, False, f"Expected CulturalKnowledge, got {type(retrieved)}")
                failed += 1
        except Exception as e:
            log_test_result(test_name, False, str(e))
            failed += 1
        
        # Test 3: Get all cultural knowledge
        test_name = "Postgres: get_all_cultural_knowledge - basic"
        try:
            ck2 = create_test_cultural_knowledge(name="Postgres Test Knowledge 2", agent_id="pg_agent_002")
            storage.upsert_cultural_knowledge(ck2)
            
            all_ck = storage.get_all_cultural_knowledge(deserialize=True)
            
            if isinstance(all_ck, list) and len(all_ck) >= 2:
                log_test_result(test_name, True, f"Retrieved {len(all_ck)} entries")
                passed += 1
            else:
                log_test_result(test_name, False, f"Expected list with >= 2 entries")
                failed += 1
        except Exception as e:
            log_test_result(test_name, False, str(e))
            failed += 1
        
        # Test 4: Get all with filter
        test_name = "Postgres: get_all_cultural_knowledge - filter by agent_id"
        try:
            filtered = storage.get_all_cultural_knowledge(agent_id="pg_agent_002", deserialize=True)
            
            if isinstance(filtered, list) and len(filtered) == 1:
                log_test_result(test_name, True)
                passed += 1
            else:
                log_test_result(test_name, False, f"Expected 1 entry")
                failed += 1
        except Exception as e:
            log_test_result(test_name, False, str(e))
            failed += 1
        
        # Test 5: Delete cultural knowledge
        test_name = "Postgres: delete_cultural_knowledge"
        try:
            storage.delete_cultural_knowledge(ck.id)
            storage.delete_cultural_knowledge(ck2.id)
            deleted = storage.get_cultural_knowledge(ck.id)
            
            if deleted is None:
                log_test_result(test_name, True)
                passed += 1
            else:
                log_test_result(test_name, False, "Entry still exists")
                failed += 1
        except Exception as e:
            log_test_result(test_name, False, str(e))
            failed += 1
        
        # Cleanup
        storage.close()
        
    except ImportError as e:
        log_test_result("Postgres: Import", False, f"PostgresStorage not available: {e}")
        failed += 1
    except Exception as e:
        error_msg = str(e)
        if "connection" in error_msg.lower() or "refused" in error_msg.lower():
            log_test_result("Postgres: Connection", False, f"PostgreSQL not available (is docker-compose running?): {e}")
        else:
            log_test_result("Postgres: Setup/Teardown", False, str(e))
        failed += 1
    
    return passed, failed


# =============================================================================
# ASYNC POSTGRES STORAGE TESTS
# =============================================================================

@pytest.mark.asyncio
async def test_async_postgres_storage() -> Tuple[int, int]:
    """Test cultural knowledge methods for AsyncPostgresStorage."""
    from upsonic.culture.cultural_knowledge import CulturalKnowledge
    
    print_separator("AsyncPostgresStorage - Cultural Knowledge Tests")
    
    passed = 0
    failed = 0
    
    postgres_url = os.getenv(
        "POSTGRES_URL",
        "postgresql+asyncpg://upsonic_test:test_password@localhost:5432/upsonic_test"
    )
    
    try:
        from upsonic.storage.postgres import AsyncPostgresStorage
        
        storage = AsyncPostgresStorage(db_url=postgres_url)
        
        # Test 1: Upsert cultural knowledge
        test_name = "AsyncPostgres: aupsert_cultural_knowledge - basic insert"
        try:
            ck = create_test_cultural_knowledge(name="AsyncPostgres Test Knowledge")
            result = await storage.aupsert_cultural_knowledge(ck, deserialize=True)
            
            if result is not None and isinstance(result, CulturalKnowledge):
                if verify_cultural_knowledge_attributes(ck, result, test_name):
                    passed += 1
                else:
                    failed += 1
            else:
                log_test_result(test_name, False, f"Expected CulturalKnowledge, got {type(result)}")
                failed += 1
        except Exception as e:
            log_test_result(test_name, False, str(e))
            failed += 1
        
        # Test 2: Get cultural knowledge by ID
        test_name = "AsyncPostgres: aget_cultural_knowledge - by ID"
        try:
            retrieved = await storage.aget_cultural_knowledge(ck.id, deserialize=True)
            
            if retrieved is not None and isinstance(retrieved, CulturalKnowledge):
                if verify_cultural_knowledge_attributes(ck, retrieved, test_name):
                    passed += 1
                else:
                    failed += 1
            else:
                log_test_result(test_name, False, f"Expected CulturalKnowledge, got {type(retrieved)}")
                failed += 1
        except Exception as e:
            log_test_result(test_name, False, str(e))
            failed += 1
        
        # Test 3: Get all cultural knowledge
        test_name = "AsyncPostgres: aget_all_cultural_knowledge - basic"
        try:
            ck2 = create_test_cultural_knowledge(name="AsyncPostgres Test Knowledge 2")
            await storage.aupsert_cultural_knowledge(ck2)
            
            all_ck = await storage.aget_all_cultural_knowledge(deserialize=True)
            
            if isinstance(all_ck, list) and len(all_ck) >= 2:
                log_test_result(test_name, True, f"Retrieved {len(all_ck)} entries")
                passed += 1
            else:
                log_test_result(test_name, False, f"Expected list with >= 2 entries")
                failed += 1
        except Exception as e:
            log_test_result(test_name, False, str(e))
            failed += 1
        
        # Test 4: Delete cultural knowledge
        test_name = "AsyncPostgres: adelete_cultural_knowledge"
        try:
            await storage.adelete_cultural_knowledge(ck.id)
            await storage.adelete_cultural_knowledge(ck2.id)
            deleted = await storage.aget_cultural_knowledge(ck.id)
            
            if deleted is None:
                log_test_result(test_name, True)
                passed += 1
            else:
                log_test_result(test_name, False, "Entry still exists")
                failed += 1
        except Exception as e:
            log_test_result(test_name, False, str(e))
            failed += 1
        
        # Cleanup
        await storage.close()
        
    except ImportError as e:
        log_test_result("AsyncPostgres: Import", False, f"AsyncPostgresStorage not available: {e}")
        failed += 1
    except Exception as e:
        error_msg = str(e)
        if "connection" in error_msg.lower() or "refused" in error_msg.lower():
            log_test_result("AsyncPostgres: Connection", False, f"PostgreSQL not available: {e}")
        else:
            log_test_result("AsyncPostgres: Setup/Teardown", False, str(e))
        failed += 1
    
    return passed, failed


# =============================================================================
# MONGO STORAGE TESTS (Sync)
# =============================================================================

def test_mongo_storage() -> Tuple[int, int]:
    """Test cultural knowledge methods for MongoStorage."""
    from upsonic.culture.cultural_knowledge import CulturalKnowledge
    
    print_separator("MongoStorage - Cultural Knowledge Tests")
    
    passed = 0
    failed = 0
    
    mongo_url = os.getenv(
        "MONGO_URL",
        "mongodb://upsonic_test:test_password@localhost:27017"
    )
    
    try:
        from upsonic.storage.mongo import MongoStorage
        
        storage = MongoStorage(db_url=mongo_url, db_name="upsonic_test")
        
        # Test 1: Upsert cultural knowledge
        test_name = "Mongo: upsert_cultural_knowledge - basic insert"
        try:
            ck = create_test_cultural_knowledge(name="Mongo Test Knowledge")
            result = storage.upsert_cultural_knowledge(ck, deserialize=True)
            
            if result is not None and isinstance(result, CulturalKnowledge):
                if verify_cultural_knowledge_attributes(ck, result, test_name):
                    passed += 1
                else:
                    failed += 1
            else:
                log_test_result(test_name, False, f"Expected CulturalKnowledge, got {type(result)}")
                failed += 1
        except Exception as e:
            log_test_result(test_name, False, str(e))
            failed += 1
        
        # Test 2: Get cultural knowledge by ID
        test_name = "Mongo: get_cultural_knowledge - by ID"
        try:
            retrieved = storage.get_cultural_knowledge(ck.id, deserialize=True)
            
            if retrieved is not None and isinstance(retrieved, CulturalKnowledge):
                if verify_cultural_knowledge_attributes(ck, retrieved, test_name):
                    passed += 1
                else:
                    failed += 1
            else:
                log_test_result(test_name, False, f"Expected CulturalKnowledge, got {type(retrieved)}")
                failed += 1
        except Exception as e:
            log_test_result(test_name, False, str(e))
            failed += 1
        
        # Test 3: Get all cultural knowledge
        test_name = "Mongo: get_all_cultural_knowledge - basic"
        try:
            ck2 = create_test_cultural_knowledge(name="Mongo Test Knowledge 2", agent_id="mongo_agent_002")
            storage.upsert_cultural_knowledge(ck2)
            
            all_ck = storage.get_all_cultural_knowledge(deserialize=True)
            
            if isinstance(all_ck, list) and len(all_ck) >= 2:
                log_test_result(test_name, True, f"Retrieved {len(all_ck)} entries")
                passed += 1
            else:
                log_test_result(test_name, False, f"Expected list with >= 2 entries")
                failed += 1
        except Exception as e:
            log_test_result(test_name, False, str(e))
            failed += 1
        
        # Test 4: Delete cultural knowledge
        test_name = "Mongo: delete_cultural_knowledge"
        try:
            storage.delete_cultural_knowledge(ck.id)
            storage.delete_cultural_knowledge(ck2.id)
            deleted = storage.get_cultural_knowledge(ck.id)
            
            if deleted is None:
                log_test_result(test_name, True)
                passed += 1
            else:
                log_test_result(test_name, False, "Entry still exists")
                failed += 1
        except Exception as e:
            log_test_result(test_name, False, str(e))
            failed += 1
        
        # Cleanup
        storage.close()
        
    except ImportError as e:
        log_test_result("Mongo: Import", False, f"MongoStorage not available: {e}")
        failed += 1
    except Exception as e:
        error_msg = str(e)
        if "connection" in error_msg.lower() or "refused" in error_msg.lower() or "ServerSelectionTimeoutError" in str(type(e)):
            log_test_result("Mongo: Connection", False, f"MongoDB not available (is docker-compose running?): {e}")
        else:
            log_test_result("Mongo: Setup/Teardown", False, str(e))
        failed += 1
    
    return passed, failed


# =============================================================================
# ASYNC MONGO STORAGE TESTS
# =============================================================================

@pytest.mark.asyncio
async def test_async_mongo_storage() -> Tuple[int, int]:
    """Test cultural knowledge methods for AsyncMongoStorage."""
    from upsonic.culture.cultural_knowledge import CulturalKnowledge
    
    print_separator("AsyncMongoStorage - Cultural Knowledge Tests")
    
    passed = 0
    failed = 0
    
    mongo_url = os.getenv(
        "MONGO_URL",
        "mongodb://upsonic_test:test_password@localhost:27017"
    )
    
    try:
        from upsonic.storage.mongo import AsyncMongoStorage
        
        storage = AsyncMongoStorage(db_url=mongo_url, db_name="upsonic_test")
        
        # Test 1: Upsert cultural knowledge
        test_name = "AsyncMongo: aupsert_cultural_knowledge - basic insert"
        try:
            ck = create_test_cultural_knowledge(name="AsyncMongo Test Knowledge")
            result = await storage.aupsert_cultural_knowledge(ck, deserialize=True)
            
            if result is not None and isinstance(result, CulturalKnowledge):
                if verify_cultural_knowledge_attributes(ck, result, test_name):
                    passed += 1
                else:
                    failed += 1
            else:
                log_test_result(test_name, False, f"Expected CulturalKnowledge, got {type(result)}")
                failed += 1
        except Exception as e:
            log_test_result(test_name, False, str(e))
            failed += 1
        
        # Test 2: Get cultural knowledge by ID
        test_name = "AsyncMongo: aget_cultural_knowledge - by ID"
        try:
            retrieved = await storage.aget_cultural_knowledge(ck.id, deserialize=True)
            
            if retrieved is not None and isinstance(retrieved, CulturalKnowledge):
                if verify_cultural_knowledge_attributes(ck, retrieved, test_name):
                    passed += 1
                else:
                    failed += 1
            else:
                log_test_result(test_name, False, f"Expected CulturalKnowledge, got {type(retrieved)}")
                failed += 1
        except Exception as e:
            log_test_result(test_name, False, str(e))
            failed += 1
        
        # Test 3: Get all cultural knowledge
        test_name = "AsyncMongo: aget_all_cultural_knowledge - basic"
        try:
            ck2 = create_test_cultural_knowledge(name="AsyncMongo Test Knowledge 2")
            await storage.aupsert_cultural_knowledge(ck2)
            
            all_ck = await storage.aget_all_cultural_knowledge(deserialize=True)
            
            if isinstance(all_ck, list) and len(all_ck) >= 2:
                log_test_result(test_name, True, f"Retrieved {len(all_ck)} entries")
                passed += 1
            else:
                log_test_result(test_name, False, f"Expected list with >= 2 entries")
                failed += 1
        except Exception as e:
            log_test_result(test_name, False, str(e))
            failed += 1
        
        # Test 4: Delete cultural knowledge
        test_name = "AsyncMongo: adelete_cultural_knowledge"
        try:
            await storage.adelete_cultural_knowledge(ck.id)
            await storage.adelete_cultural_knowledge(ck2.id)
            deleted = await storage.aget_cultural_knowledge(ck.id)
            
            if deleted is None:
                log_test_result(test_name, True)
                passed += 1
            else:
                log_test_result(test_name, False, "Entry still exists")
                failed += 1
        except Exception as e:
            log_test_result(test_name, False, str(e))
            failed += 1
        
        # Cleanup
        await storage.close()
        
    except ImportError as e:
        log_test_result("AsyncMongo: Import", False, f"AsyncMongoStorage not available: {e}")
        failed += 1
    except Exception as e:
        error_msg = str(e)
        if "connection" in error_msg.lower() or "refused" in error_msg.lower():
            log_test_result("AsyncMongo: Connection", False, f"MongoDB not available: {e}")
        else:
            log_test_result("AsyncMongo: Setup/Teardown", False, str(e))
        failed += 1
    
    return passed, failed


# =============================================================================
# REDIS STORAGE TESTS
# =============================================================================

def test_redis_storage() -> Tuple[int, int]:
    """Test cultural knowledge methods for RedisStorage."""
    from upsonic.culture.cultural_knowledge import CulturalKnowledge
    
    print_separator("RedisStorage - Cultural Knowledge Tests")
    
    passed = 0
    failed = 0
    
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
    
    try:
        from upsonic.storage.redis import RedisStorage
        
        storage = RedisStorage(db_url=redis_url, db_prefix="test_culture_")
        
        # Test 1: Upsert cultural knowledge
        test_name = "Redis: upsert_cultural_knowledge - basic insert"
        try:
            ck = create_test_cultural_knowledge(name="Redis Test Knowledge")
            result = storage.upsert_cultural_knowledge(ck, deserialize=True)
            
            if result is not None and isinstance(result, CulturalKnowledge):
                if verify_cultural_knowledge_attributes(ck, result, test_name):
                    passed += 1
                else:
                    failed += 1
            else:
                log_test_result(test_name, False, f"Expected CulturalKnowledge, got {type(result)}")
                failed += 1
        except Exception as e:
            log_test_result(test_name, False, str(e))
            failed += 1
        
        # Test 2: Get cultural knowledge by ID
        test_name = "Redis: get_cultural_knowledge - by ID"
        try:
            retrieved = storage.get_cultural_knowledge(ck.id, deserialize=True)
            
            if retrieved is not None and isinstance(retrieved, CulturalKnowledge):
                if verify_cultural_knowledge_attributes(ck, retrieved, test_name):
                    passed += 1
                else:
                    failed += 1
            else:
                log_test_result(test_name, False, f"Expected CulturalKnowledge, got {type(retrieved)}")
                failed += 1
        except Exception as e:
            log_test_result(test_name, False, str(e))
            failed += 1
        
        # Test 3: Get all cultural knowledge
        test_name = "Redis: get_all_cultural_knowledge - basic"
        try:
            ck2 = create_test_cultural_knowledge(name="Redis Test Knowledge 2", agent_id="redis_agent_002")
            storage.upsert_cultural_knowledge(ck2)
            
            all_ck = storage.get_all_cultural_knowledge(deserialize=True)
            
            if isinstance(all_ck, list) and len(all_ck) >= 2:
                log_test_result(test_name, True, f"Retrieved {len(all_ck)} entries")
                passed += 1
            else:
                log_test_result(test_name, False, f"Expected list with >= 2 entries")
                failed += 1
        except Exception as e:
            log_test_result(test_name, False, str(e))
            failed += 1
        
        # Test 4: Delete cultural knowledge
        test_name = "Redis: delete_cultural_knowledge"
        try:
            storage.delete_cultural_knowledge(ck.id)
            storage.delete_cultural_knowledge(ck2.id)
            deleted = storage.get_cultural_knowledge(ck.id)
            
            if deleted is None:
                log_test_result(test_name, True)
                passed += 1
            else:
                log_test_result(test_name, False, "Entry still exists")
                failed += 1
        except Exception as e:
            log_test_result(test_name, False, str(e))
            failed += 1
        
        # Cleanup
        storage.close()
        
    except ImportError as e:
        log_test_result("Redis: Import", False, f"RedisStorage not available: {e}")
        failed += 1
    except Exception as e:
        error_msg = str(e)
        if "connection" in error_msg.lower() or "refused" in error_msg.lower():
            log_test_result("Redis: Connection", False, f"Redis not available (is docker-compose running?): {e}")
        else:
            log_test_result("Redis: Setup/Teardown", False, str(e))
        failed += 1
    
    return passed, failed


# =============================================================================
# MAIN TEST RUNNER
# =============================================================================

def main():
    """Run all storage tests for cultural knowledge."""
    print("\n" + "=" * 80)
    print("  CULTURAL KNOWLEDGE STORAGE TESTS - ALL PROVIDERS")
    print("=" * 80)
    print(f"\nStarted at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nNote: PostgreSQL, MongoDB, and Redis tests require docker-compose to be running.")
    print("      Run: docker-compose -f tests/smoke_tests/docker-compose.yml up -d\n")
    
    total_passed = 0
    total_failed = 0
    
    # SQLite Storage (Sync)
    passed, failed = test_sqlite_storage()
    total_passed += passed
    total_failed += failed
    
    # AsyncSQLite Storage
    passed, failed = asyncio.run(test_async_sqlite_storage())
    total_passed += passed
    total_failed += failed
    
    # InMemory Storage
    passed, failed = test_in_memory_storage()
    total_passed += passed
    total_failed += failed
    
    # JSON Storage
    passed, failed = test_json_storage()
    total_passed += passed
    total_failed += failed
    
    # PostgreSQL Storage (Sync)
    passed, failed = test_postgres_storage()
    total_passed += passed
    total_failed += failed
    
    # AsyncPostgres Storage
    passed, failed = asyncio.run(test_async_postgres_storage())
    total_passed += passed
    total_failed += failed
    
    # MongoDB Storage (Sync)
    passed, failed = test_mongo_storage()
    total_passed += passed
    total_failed += failed
    
    # AsyncMongo Storage
    passed, failed = asyncio.run(test_async_mongo_storage())
    total_passed += passed
    total_failed += failed
    
    # Redis Storage
    passed, failed = test_redis_storage()
    total_passed += passed
    total_failed += failed
    
    # Print summary
    print_separator("TEST SUMMARY")
    print(f"Total Tests: {total_passed + total_failed}")
    print(f"Passed: {total_passed}")
    print(f"Failed: {total_failed}")
    
    if total_failed > 0:
        print("\n❌ FAILED TESTS:")
        for result in test_results:
            if not result["passed"]:
                msg = f"  - {result['name']}"
                if result["message"]:
                    msg += f": {result['message']}"
                print(msg)
    
    print(f"\nCompleted at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Return exit code
    sys.exit(1 if total_failed > 0 else 0)


if __name__ == "__main__":
    main()

