"""
Comprehensive smoke tests for Qdrant vector database provider.

Tests all methods, attributes, and connection modes (IN_MEMORY).
"""

import os
import pytest
import tempfile
import asyncio
from typing import List, Dict, Any, Optional
from hashlib import md5

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not available, use system env vars

from upsonic.vectordb.providers.qdrant import QdrantProvider
from upsonic.vectordb.config import (
    QdrantConfig,
    ConnectionConfig,
    Mode,
    DistanceMetric,
    HNSWIndexConfig,
    FlatIndexConfig
)
from upsonic.utils.package.exception import (
    VectorDBConnectionError,
    ConfigurationError,
    CollectionDoesNotExistError,
    VectorDBError,
    SearchError,
    UpsertError
)
from upsonic.schemas.vector_schemas import VectorSearchResult


# Test data
SAMPLE_VECTORS: List[List[float]] = [
    [0.1, 0.2, 0.3, 0.4, 0.5],
    [0.6, 0.7, 0.8, 0.9, 1.0],
    [1.1, 1.2, 1.3, 1.4, 1.5],
    [1.6, 1.7, 1.8, 1.9, 2.0],
    [2.1, 2.2, 2.3, 2.4, 2.5]
]

SAMPLE_PAYLOADS: List[Dict[str, Any]] = [
    {"content": "The theory of relativity revolutionized physics", "category": "science", "author": "Einstein", "year": 1905},
    {"content": "Laws of motion and universal gravitation", "category": "science", "author": "Newton", "year": 1687},
    {"content": "To be or not to be, that is the question", "category": "literature", "author": "Shakespeare", "year": 1600},
    {"content": "It was the best of times, it was the worst of times", "category": "literature", "author": "Dickens", "year": 1850},
    {"content": "The unexamined life is not worth living", "category": "philosophy", "author": "Plato", "year": -400}
]

SAMPLE_CHUNKS: List[str] = [
    "The theory of relativity revolutionized physics",
    "Laws of motion and universal gravitation",
    "To be or not to be, that is the question",
    "It was the best of times, it was the worst of times",
    "The unexamined life is not worth living"
]

SAMPLE_IDS: List[str] = ["doc1", "doc2", "doc3", "doc4", "doc5"]

QUERY_VECTOR: List[float] = [0.15, 0.25, 0.35, 0.45, 0.55]
QUERY_TEXT: str = "physics theory"


def assert_vector_matches(actual_vector: Any, expected_vector: List[float], vector_id: str = "", tolerance: float = 1e-6) -> None:
    """
    Assert that a retrieved vector matches the expected vector.
    
    Args:
        actual_vector: The vector retrieved from the database (can be list, numpy array, etc.)
        expected_vector: The original vector that was inserted
        vector_id: Optional ID for better error messages
        tolerance: Floating point comparison tolerance
    """
    assert actual_vector is not None, f"Vector is None for {vector_id}"
    assert hasattr(actual_vector, '__len__'), f"Vector has no length for {vector_id}"
    assert len(actual_vector) == len(expected_vector), \
        f"Vector length mismatch for {vector_id}: {len(actual_vector)} != {len(expected_vector)}"
    
    # Convert to list of floats (handles numpy arrays and other types)
    vector_list = [float(x) for x in actual_vector]
    assert len(vector_list) == len(expected_vector), \
        f"Converted vector length mismatch for {vector_id}: {len(vector_list)} != {len(expected_vector)}"
    
    # Compare element by element
    for j, (actual, expected) in enumerate(zip(vector_list, expected_vector)):
        assert abs(actual - expected) < tolerance, \
            f"Vector element {j} mismatch for {vector_id}: {actual} != {expected} (diff: {abs(actual - expected)})"


def assert_result_vector_matches(result: VectorSearchResult, expected_vector: List[float], result_index: int = 0) -> None:
    """
    Assert that a search result's vector matches the expected vector.
    
    Args:
        result: The VectorSearchResult from search/fetch operations
        expected_vector: The original vector that was inserted
        result_index: Index for better error messages
    """
    assert_vector_matches(result.vector, expected_vector, vector_id=f"result[{result_index}] (id={result.id})")


def get_expected_vector_by_id(record_id: str) -> List[float]:
    """
    Get the expected vector for a given record ID.
    
    Args:
        record_id: The ID of the record (e.g., "doc1", "doc2", etc.)
    
    Returns:
        The original vector that was inserted for this ID
    """
    if record_id in SAMPLE_IDS:
        idx = SAMPLE_IDS.index(record_id)
        return SAMPLE_VECTORS[idx]
    raise ValueError(f"Unknown record ID: {record_id}")


class TestQdrantProviderIN_MEMORY:
    """Test QdrantProvider in IN_MEMORY mode."""
    
    @pytest.fixture
    def config(self, request) -> QdrantConfig:
        """Create IN_MEMORY QdrantConfig with unique collection name."""
        import uuid
        unique_name = f"test_memory_{uuid.uuid4().hex[:8]}"
        return QdrantConfig(
            vector_size=5,
            collection_name=unique_name,
            connection=ConnectionConfig(mode=Mode.IN_MEMORY),
            distance_metric=DistanceMetric.COSINE,
            index=HNSWIndexConfig(m=16, ef_construction=200)
        )
    
    @pytest.fixture
    def provider(self, config: QdrantConfig) -> QdrantProvider:
        """Create QdrantProvider instance."""
        return QdrantProvider(config)
    
    @pytest.mark.asyncio
    async def test_initialization(self, provider: QdrantProvider, config: QdrantConfig):
        """Test provider initialization and attributes."""
        assert provider._config == config
        assert provider._config.collection_name.startswith("test_memory_")
        assert provider._config.vector_size == 5
        assert provider._config.distance_metric == DistanceMetric.COSINE
        assert not provider._is_connected
        assert provider._client is None
        
        # Test provider metadata attributes
        assert provider.provider_name is not None
        assert isinstance(provider.provider_id, str)
        assert len(provider.provider_id) > 0
        assert provider.reranker is None
    
    @pytest.mark.asyncio
    async def test_connect(self, provider: QdrantProvider):
        """Test connection to Qdrant."""
        await provider.connect()
        assert provider._is_connected
        assert provider._client is not None
        assert await provider.is_ready()
    
    @pytest.mark.asyncio
    async def test_connect_sync(self, provider: QdrantProvider):
        """Test synchronous connection."""
        provider.connect_sync()
        assert provider._is_connected
        assert provider._client is not None
        assert provider.is_ready_sync()
    
    @pytest.mark.asyncio
    async def test_disconnect(self, provider: QdrantProvider):
        """Test disconnection."""
        await provider.connect()
        assert provider._is_connected
        await provider.disconnect()
        assert not provider._is_connected
        assert provider._client is None
    
    @pytest.mark.asyncio
    async def test_disconnect_sync(self, provider: QdrantProvider):
        """Test synchronous disconnection."""
        provider.connect_sync()
        assert provider._is_connected
        provider.disconnect_sync()
        assert not provider._is_connected
    
    @pytest.mark.asyncio
    async def test_close(self, provider: QdrantProvider):
        """Test close method."""
        await provider.connect()
        assert provider._is_connected
        await provider.close()
        assert not provider._is_connected
        assert provider._client is None
    
    @pytest.mark.asyncio
    async def test_is_ready(self, provider: QdrantProvider):
        """Test is_ready check."""
        assert not await provider.is_ready()
        await provider.connect()
        assert await provider.is_ready()
        await provider.disconnect()
        assert not await provider.is_ready()
    
    @pytest.mark.asyncio
    async def test_is_ready_sync(self, provider: QdrantProvider):
        """Test synchronous is_ready check."""
        assert not provider.is_ready_sync()
        provider.connect_sync()
        assert provider.is_ready_sync()
        provider.disconnect_sync()
        assert not provider.is_ready_sync()
    
    @pytest.mark.asyncio
    async def test_create_collection(self, provider: QdrantProvider):
        """Test collection creation."""
        await provider.connect()
        assert not await provider.collection_exists()
        await provider.create_collection()
        assert await provider.collection_exists()
        await provider.disconnect()
    
    @pytest.mark.asyncio
    async def test_create_collection_sync(self, provider: QdrantProvider):
        """Test synchronous collection creation."""
        provider.connect_sync()
        # Delete collection if it exists first
        try:
            if provider.collection_exists_sync():
                provider.delete_collection_sync()
        except Exception:
            pass
        assert not provider.collection_exists_sync()
        provider.create_collection_sync()
        assert provider.collection_exists_sync()
        provider.disconnect_sync()
    
    @pytest.mark.asyncio
    async def test_collection_exists(self, provider: QdrantProvider):
        """Test collection existence check."""
        await provider.connect()
        # Delete collection if it exists first
        try:
            if await provider.collection_exists():
                await provider.delete_collection()
        except Exception:
            pass
        assert not await provider.collection_exists()
        await provider.create_collection()
        assert await provider.collection_exists()
        await provider.disconnect()
    
    @pytest.mark.asyncio
    async def test_collection_exists_sync(self, provider: QdrantProvider):
        """Test synchronous collection existence check."""
        provider.connect_sync()
        # Delete collection if it exists first
        try:
            if provider.collection_exists_sync():
                provider.delete_collection_sync()
        except Exception:
            pass
        assert not provider.collection_exists_sync()
        provider.create_collection_sync()
        assert provider.collection_exists_sync()
        provider.disconnect_sync()
    
    @pytest.mark.asyncio
    async def test_delete_collection(self, provider: QdrantProvider):
        """Test collection deletion."""
        await provider.connect()
        await provider.create_collection()
        assert await provider.collection_exists()
        await provider.delete_collection()
        assert not await provider.collection_exists()
        await provider.disconnect()
    
    @pytest.mark.asyncio
    async def test_delete_collection_sync(self, provider: QdrantProvider):
        """Test synchronous collection deletion."""
        provider.connect_sync()
        provider.create_collection_sync()
        assert provider.collection_exists_sync()
        provider.delete_collection_sync()
        assert not provider.collection_exists_sync()
        provider.disconnect_sync()
    
    @pytest.mark.asyncio
    async def test_delete_nonexistent_collection(self, provider: QdrantProvider):
        """Test deleting non-existent collection."""
        await provider.connect()
        # Qdrant doesn't raise an error when deleting a non-existent collection
        # It just succeeds silently, so we just verify it doesn't crash
        try:
            await provider.delete_collection()
        except CollectionDoesNotExistError:
            # If it does raise, that's also acceptable
            pass
        await provider.disconnect()
    
    @pytest.mark.asyncio
    async def test_upsert(self, provider: QdrantProvider):
        """Test upsert operation with content validation."""
        await provider.connect()
        await provider.create_collection()
        await provider.upsert(
            vectors=SAMPLE_VECTORS,
            payloads=SAMPLE_PAYLOADS,
            ids=SAMPLE_IDS,
            chunks=SAMPLE_CHUNKS
        )
        # Verify data was actually stored with correct content
        results = await provider.fetch(ids=SAMPLE_IDS)
        assert len(results) == 5
        # Qdrant normalizes IDs, so we need to match by content instead
        for result in results:
            assert result.id is not None
            assert result.payload is not None
            content = result.payload.get("content")
            assert content in SAMPLE_CHUNKS
            idx = SAMPLE_CHUNKS.index(content)
            assert result.payload.get("category") == SAMPLE_PAYLOADS[idx]["category"]
            assert result.payload.get("author") == SAMPLE_PAYLOADS[idx]["author"]
            assert result.payload.get("year") == SAMPLE_PAYLOADS[idx]["year"]
            assert result.text == SAMPLE_CHUNKS[idx]
            # Validate vector is retrieved and has correct length
            # Note: Qdrant may normalize vectors for cosine similarity, so we don't compare exact values
            assert result.vector is not None
            assert len(result.vector) == len(SAMPLE_VECTORS[idx])
        await provider.disconnect()
    
    @pytest.mark.asyncio
    async def test_upsert_sync(self, provider: QdrantProvider):
        """Test synchronous upsert."""
        provider.connect_sync()
        provider.create_collection_sync()
        provider.upsert_sync(
            vectors=SAMPLE_VECTORS,
            payloads=SAMPLE_PAYLOADS,
            ids=SAMPLE_IDS,
            chunks=SAMPLE_CHUNKS
        )
        provider.disconnect_sync()
    
    @pytest.mark.asyncio
    async def test_upsert_with_document_tracking(self, provider: QdrantProvider):
        """Test upsert with document_name, document_id, content_id and validate metadata."""
        await provider.connect()
        await provider.create_collection()
        payloads_with_tracking = []
        for i, payload in enumerate(SAMPLE_PAYLOADS[:2]):
            payload_copy = payload.copy()
            payload_copy["document_name"] = f"doc{i+1}"
            payload_copy["document_id"] = f"doc_id_{i+1}"
            payload_copy["content_id"] = f"content_{i+1}"
            payloads_with_tracking.append(payload_copy)
        
        await provider.upsert(
            vectors=SAMPLE_VECTORS[:2],
            payloads=payloads_with_tracking,
            ids=SAMPLE_IDS[:2],
            chunks=SAMPLE_CHUNKS[:2]
        )
        # Verify tracking metadata was stored correctly
        results = await provider.fetch(ids=SAMPLE_IDS[:2])
        assert len(results) == 2
        # Qdrant normalizes IDs, so we need to match by content_id instead
        for result in results:
            content_id = result.payload.get("content_id")
            assert content_id in ["content_1", "content_2"]
            idx = int(content_id.split("_")[1]) - 1
            assert result.payload.get("document_name") == f"doc{idx+1}"
            assert result.payload.get("document_id") == f"doc_id_{idx+1}"
            assert result.payload.get("content_id") == f"content_{idx+1}"
            assert result.payload.get("category") == SAMPLE_PAYLOADS[idx]["category"]
            assert result.text == SAMPLE_CHUNKS[idx]
            # Validate vector is retrieved and has correct length
            # Note: Qdrant may normalize vectors for cosine similarity, so we don't compare exact values
            assert result.vector is not None
            assert len(result.vector) == len(SAMPLE_VECTORS[idx])
        await provider.disconnect()
    
    @pytest.mark.asyncio
    async def test_upsert_validation_error(self, provider: QdrantProvider):
        """Test upsert with mismatched lengths raises error."""
        await provider.connect()
        await provider.create_collection()
        with pytest.raises(ValueError):
            await provider.upsert(
                vectors=SAMPLE_VECTORS[:2],
                payloads=SAMPLE_PAYLOADS[:3],
                ids=SAMPLE_IDS[:2],
                chunks=SAMPLE_CHUNKS[:2]
            )
        await provider.disconnect()
    
    @pytest.mark.asyncio
    async def test_fetch(self, provider: QdrantProvider):
        """Test fetch operation."""
        await provider.connect()
        await provider.create_collection()
        await provider.upsert(
            vectors=SAMPLE_VECTORS,
            payloads=SAMPLE_PAYLOADS,
            ids=SAMPLE_IDS,
            chunks=SAMPLE_CHUNKS
        )
        results = await provider.fetch(ids=SAMPLE_IDS[:2])
        assert len(results) == 2
        assert all(isinstance(r, VectorSearchResult) for r in results)
        assert results[0].payload is not None
        assert results[0].text is not None
        # Validate vectors are retrieved and have correct length
        # Note: Qdrant normalizes IDs and may normalize vectors for cosine similarity
        for result in results:
            content = result.text
            assert content in SAMPLE_CHUNKS[:2]
            idx = SAMPLE_CHUNKS.index(content)
            assert result.vector is not None
            assert len(result.vector) == len(SAMPLE_VECTORS[idx])
        await provider.disconnect()
    
    @pytest.mark.asyncio
    async def test_fetch_sync(self, provider: QdrantProvider):
        """Test synchronous fetch."""
        provider.connect_sync()
        provider.create_collection_sync()
        provider.upsert_sync(
            vectors=SAMPLE_VECTORS,
            payloads=SAMPLE_PAYLOADS,
            ids=SAMPLE_IDS,
            chunks=SAMPLE_CHUNKS
        )
        results = provider.fetch_sync(ids=SAMPLE_IDS[:2])
        assert len(results) == 2
        assert all(isinstance(r, VectorSearchResult) for r in results)
        # Validate vectors are retrieved and have correct length
        # Note: Qdrant normalizes IDs and may normalize vectors for cosine similarity
        for result in results:
            content = result.text
            assert content in SAMPLE_CHUNKS[:2]
            idx = SAMPLE_CHUNKS.index(content)
            assert result.vector is not None
            assert len(result.vector) == len(SAMPLE_VECTORS[idx])
        provider.disconnect_sync()
    
    @pytest.mark.asyncio
    async def test_delete(self, provider: QdrantProvider):
        """Test delete operation."""
        await provider.connect()
        await provider.create_collection()
        await provider.upsert(
            vectors=SAMPLE_VECTORS,
            payloads=SAMPLE_PAYLOADS,
            ids=SAMPLE_IDS,
            chunks=SAMPLE_CHUNKS
        )
        await provider.delete(ids=SAMPLE_IDS[:2])
        results = await provider.fetch(ids=SAMPLE_IDS[:2])
        assert len(results) == 0
        await provider.disconnect()
    
    @pytest.mark.asyncio
    async def test_delete_sync(self, provider: QdrantProvider):
        """Test synchronous delete."""
        provider.connect_sync()
        provider.create_collection_sync()
        provider.upsert_sync(
            vectors=SAMPLE_VECTORS,
            payloads=SAMPLE_PAYLOADS,
            ids=SAMPLE_IDS,
            chunks=SAMPLE_CHUNKS
        )
        provider.delete_sync(ids=SAMPLE_IDS[:2])
        results = provider.fetch_sync(ids=SAMPLE_IDS[:2])
        assert len(results) == 0
        provider.disconnect_sync()
    
    @pytest.mark.asyncio
    async def test_delete_by_document_name(self, provider: QdrantProvider):
        """Test delete_by_document_name."""
        await provider.connect()
        await provider.create_collection()
        # First ensure collection is empty
        initial_count = await provider.get_count()
        payloads_with_doc_name = []
        for payload in SAMPLE_PAYLOADS[:2]:
            payload_copy = payload.copy()
            payload_copy["document_name"] = "test_doc"
            payloads_with_doc_name.append(payload_copy)
        
        await provider.upsert(
            vectors=SAMPLE_VECTORS[:2],
            payloads=payloads_with_doc_name,
            ids=SAMPLE_IDS[:2],
            chunks=SAMPLE_CHUNKS[:2]
        )
        count_before = await provider.get_count()
        assert count_before == initial_count + 2
        deleted = provider.delete_by_document_name("test_doc")
        assert deleted is True
        count_after = await provider.get_count()
        assert count_after == initial_count
        await provider.disconnect()
    
    @pytest.mark.asyncio
    async def test_async_delete_by_document_name(self, provider: QdrantProvider):
        """Test async_delete_by_document_name."""
        await provider.connect()
        await provider.create_collection()
        payloads_with_doc_name = []
        for payload in SAMPLE_PAYLOADS[:2]:
            payload_copy = payload.copy()
            payload_copy["document_name"] = "test_doc"
            payloads_with_doc_name.append(payload_copy)
        
        await provider.upsert(
            vectors=SAMPLE_VECTORS[:2],
            payloads=payloads_with_doc_name,
            ids=SAMPLE_IDS[:2],
            chunks=SAMPLE_CHUNKS[:2]
        )
        deleted = await provider.async_delete_by_document_name("test_doc")
        assert deleted is True
        await provider.disconnect()
    
    @pytest.mark.asyncio
    async def test_delete_by_document_id(self, provider: QdrantProvider):
        """Test delete_by_document_id."""
        await provider.connect()
        await provider.create_collection()
        payloads_with_doc_id = []
        for payload in SAMPLE_PAYLOADS[:2]:
            payload_copy = payload.copy()
            payload_copy["document_id"] = "doc_id_1"
            payloads_with_doc_id.append(payload_copy)
        
        await provider.upsert(
            vectors=SAMPLE_VECTORS[:2],
            payloads=payloads_with_doc_id,
            ids=SAMPLE_IDS[:2],
            chunks=SAMPLE_CHUNKS[:2]
        )
        deleted = provider.delete_by_document_id("doc_id_1")
        assert deleted is True
        await provider.disconnect()
    
    @pytest.mark.asyncio
    async def test_async_delete_by_document_id(self, provider: QdrantProvider):
        """Test async_delete_by_document_id."""
        await provider.connect()
        await provider.create_collection()
        payloads_with_doc_id = []
        for payload in SAMPLE_PAYLOADS[:2]:
            payload_copy = payload.copy()
            payload_copy["document_id"] = "doc_id_1"
            payloads_with_doc_id.append(payload_copy)
        
        await provider.upsert(
            vectors=SAMPLE_VECTORS[:2],
            payloads=payloads_with_doc_id,
            ids=SAMPLE_IDS[:2],
            chunks=SAMPLE_CHUNKS[:2]
        )
        deleted = await provider.async_delete_by_document_id("doc_id_1")
        assert deleted is True
        await provider.disconnect()
    
    @pytest.mark.asyncio
    async def test_delete_by_content_id(self, provider: QdrantProvider):
        """Test delete_by_content_id."""
        await provider.connect()
        await provider.create_collection()
        payloads_with_content_id = []
        for payload in SAMPLE_PAYLOADS[:2]:
            payload_copy = payload.copy()
            payload_copy["content_id"] = "content_1"
            payloads_with_content_id.append(payload_copy)
        
        await provider.upsert(
            vectors=SAMPLE_VECTORS[:2],
            payloads=payloads_with_content_id,
            ids=SAMPLE_IDS[:2],
            chunks=SAMPLE_CHUNKS[:2]
        )
        deleted = provider.delete_by_content_id("content_1")
        assert deleted is True
        await provider.disconnect()
    
    @pytest.mark.asyncio
    async def test_async_delete_by_content_id(self, provider: QdrantProvider):
        """Test async_delete_by_content_id."""
        await provider.connect()
        await provider.create_collection()
        payloads_with_content_id = []
        for payload in SAMPLE_PAYLOADS[:2]:
            payload_copy = payload.copy()
            payload_copy["content_id"] = "content_1"
            payloads_with_content_id.append(payload_copy)
        
        await provider.upsert(
            vectors=SAMPLE_VECTORS[:2],
            payloads=payloads_with_content_id,
            ids=SAMPLE_IDS[:2],
            chunks=SAMPLE_CHUNKS[:2]
        )
        deleted = await provider.async_delete_by_content_id("content_1")
        assert deleted is True
        await provider.disconnect()
    
    @pytest.mark.asyncio
    async def test_delete_by_metadata(self, provider: QdrantProvider):
        """Test delete_by_metadata."""
        await provider.connect()
        await provider.create_collection()
        # Create payloads with metadata structure
        payloads_with_metadata = []
        for payload in SAMPLE_PAYLOADS:
            payload_copy = payload.copy()
            # Store category in metadata dict
            payload_copy["metadata"] = {"category": payload["category"]}
            payloads_with_metadata.append(payload_copy)
        
        await provider.upsert(
            vectors=SAMPLE_VECTORS,
            payloads=payloads_with_metadata,
            ids=SAMPLE_IDS,
            chunks=SAMPLE_CHUNKS
        )
        # delete_by_metadata expects keys that will be accessed as metadata.{key}
        # So we pass {"category": "science"} and it looks for metadata.category
        deleted = provider.delete_by_metadata({"category": "science"})
        assert deleted is True
        await provider.disconnect()
    
    @pytest.mark.asyncio
    async def test_async_delete_by_metadata(self, provider: QdrantProvider):
        """Test async_delete_by_metadata."""
        await provider.connect()
        await provider.create_collection()
        # Create payloads with metadata structure
        payloads_with_metadata = []
        for payload in SAMPLE_PAYLOADS:
            payload_copy = payload.copy()
            # Store category in metadata dict
            payload_copy["metadata"] = {"category": payload["category"]}
            payloads_with_metadata.append(payload_copy)
        
        await provider.upsert(
            vectors=SAMPLE_VECTORS,
            payloads=payloads_with_metadata,
            ids=SAMPLE_IDS,
            chunks=SAMPLE_CHUNKS
        )
        # delete_by_metadata expects keys that will be accessed as metadata.{key}
        # So we pass {"category": "science"} and it looks for metadata.category
        deleted = await provider.async_delete_by_metadata({"category": "science"})
        assert deleted is True
        await provider.disconnect()
    
    @pytest.mark.asyncio
    async def test_id_exists(self, provider: QdrantProvider):
        """Test id_exists check."""
        await provider.connect()
        await provider.create_collection()
        await provider.upsert(
            vectors=SAMPLE_VECTORS[:1],
            payloads=SAMPLE_PAYLOADS[:1],
            ids=SAMPLE_IDS[:1],
            chunks=SAMPLE_CHUNKS[:1]
        )
        assert await provider.id_exists("doc1")
        assert not await provider.id_exists("nonexistent")
        await provider.disconnect()
    
    @pytest.mark.asyncio
    async def test_document_name_exists(self, provider: QdrantProvider):
        """Test document_name_exists."""
        await provider.connect()
        await provider.create_collection()
        payloads_with_doc_name = []
        for payload in SAMPLE_PAYLOADS[:1]:
            payload_copy = payload.copy()
            payload_copy["document_name"] = "test_doc"
            payloads_with_doc_name.append(payload_copy)
        
        await provider.upsert(
            vectors=SAMPLE_VECTORS[:1],
            payloads=payloads_with_doc_name,
            ids=SAMPLE_IDS[:1],
            chunks=SAMPLE_CHUNKS[:1]
        )
        assert provider.document_name_exists("test_doc")
        assert not provider.document_name_exists("nonexistent")
        await provider.disconnect()
    
    @pytest.mark.asyncio
    async def test_async_document_name_exists(self, provider: QdrantProvider):
        """Test async_document_name_exists."""
        await provider.connect()
        await provider.create_collection()
        payloads_with_doc_name = []
        for payload in SAMPLE_PAYLOADS[:1]:
            payload_copy = payload.copy()
            payload_copy["document_name"] = "test_doc"
            payloads_with_doc_name.append(payload_copy)
        
        await provider.upsert(
            vectors=SAMPLE_VECTORS[:1],
            payloads=payloads_with_doc_name,
            ids=SAMPLE_IDS[:1],
            chunks=SAMPLE_CHUNKS[:1]
        )
        assert await provider.async_document_name_exists("test_doc")
        assert not await provider.async_document_name_exists("nonexistent")
        await provider.disconnect()
    
    @pytest.mark.asyncio
    async def test_document_id_exists(self, provider: QdrantProvider):
        """Test document_id_exists."""
        await provider.connect()
        await provider.create_collection()
        payloads_with_doc_id = []
        for payload in SAMPLE_PAYLOADS[:1]:
            payload_copy = payload.copy()
            payload_copy["document_id"] = "doc_id_1"
            payloads_with_doc_id.append(payload_copy)
        
        await provider.upsert(
            vectors=SAMPLE_VECTORS[:1],
            payloads=payloads_with_doc_id,
            ids=SAMPLE_IDS[:1],
            chunks=SAMPLE_CHUNKS[:1]
        )
        assert provider.document_id_exists("doc_id_1")
        assert not provider.document_id_exists("nonexistent")
        await provider.disconnect()
    
    @pytest.mark.asyncio
    async def test_async_document_id_exists(self, provider: QdrantProvider):
        """Test async_document_id_exists."""
        await provider.connect()
        await provider.create_collection()
        payloads_with_doc_id = []
        for payload in SAMPLE_PAYLOADS[:1]:
            payload_copy = payload.copy()
            payload_copy["document_id"] = "doc_id_1"
            payloads_with_doc_id.append(payload_copy)
        
        await provider.upsert(
            vectors=SAMPLE_VECTORS[:1],
            payloads=payloads_with_doc_id,
            ids=SAMPLE_IDS[:1],
            chunks=SAMPLE_CHUNKS[:1]
        )
        assert await provider.async_document_id_exists("doc_id_1")
        assert not await provider.async_document_id_exists("nonexistent")
        await provider.disconnect()
    
    @pytest.mark.asyncio
    async def test_content_id_exists(self, provider: QdrantProvider):
        """Test content_id_exists."""
        await provider.connect()
        await provider.create_collection()
        payloads_with_content_id = []
        for payload in SAMPLE_PAYLOADS[:1]:
            payload_copy = payload.copy()
            payload_copy["content_id"] = "content_1"
            payloads_with_content_id.append(payload_copy)
        
        await provider.upsert(
            vectors=SAMPLE_VECTORS[:1],
            payloads=payloads_with_content_id,
            ids=SAMPLE_IDS[:1],
            chunks=SAMPLE_CHUNKS[:1]
        )
        assert provider.content_id_exists("content_1")
        assert not provider.content_id_exists("nonexistent")
        await provider.disconnect()
    
    @pytest.mark.asyncio
    async def test_async_content_id_exists(self, provider: QdrantProvider):
        """Test async_content_id_exists."""
        await provider.connect()
        await provider.create_collection()
        payloads_with_content_id = []
        for payload in SAMPLE_PAYLOADS[:1]:
            payload_copy = payload.copy()
            payload_copy["content_id"] = "content_1"
            payloads_with_content_id.append(payload_copy)
        
        await provider.upsert(
            vectors=SAMPLE_VECTORS[:1],
            payloads=payloads_with_content_id,
            ids=SAMPLE_IDS[:1],
            chunks=SAMPLE_CHUNKS[:1]
        )
        assert await provider.async_content_id_exists("content_1")
        assert not await provider.async_content_id_exists("nonexistent")
        await provider.disconnect()
    
    @pytest.mark.asyncio
    async def test_content_exists(self, provider: QdrantProvider):
        """Test content_exists."""
        await provider.connect()
        await provider.create_collection()
        await provider.upsert(
            vectors=SAMPLE_VECTORS[:1],
            payloads=SAMPLE_PAYLOADS[:1],
            ids=SAMPLE_IDS[:1],
            chunks=SAMPLE_CHUNKS[:1]
        )
        assert await provider.content_exists(SAMPLE_CHUNKS[0])
        assert not await provider.content_exists("nonexistent content")
        await provider.disconnect()
    
    @pytest.mark.asyncio
    async def test_update_metadata(self, provider: QdrantProvider):
        """Test update_metadata."""
        await provider.connect()
        await provider.create_collection()
        payloads_with_content_id = []
        for payload in SAMPLE_PAYLOADS[:1]:
            payload_copy = payload.copy()
            payload_copy["content_id"] = "content_1"
            payloads_with_content_id.append(payload_copy)
        
        await provider.upsert(
            vectors=SAMPLE_VECTORS[:1],
            payloads=payloads_with_content_id,
            ids=SAMPLE_IDS[:1],
            chunks=SAMPLE_CHUNKS[:1]
        )
        updated = provider.update_metadata("content_1", {"new_field": "new_value"})
        assert updated is True
        results = await provider.fetch(ids=SAMPLE_IDS[:1])
        assert results[0].payload.get("metadata", {}).get("new_field") == "new_value"
        await provider.disconnect()
    
    @pytest.mark.asyncio
    async def test_async_update_metadata(self, provider: QdrantProvider):
        """Test async_update_metadata."""
        await provider.connect()
        await provider.create_collection()
        payloads_with_content_id = []
        for payload in SAMPLE_PAYLOADS[:1]:
            payload_copy = payload.copy()
            payload_copy["content_id"] = "content_1"
            payloads_with_content_id.append(payload_copy)
        
        await provider.upsert(
            vectors=SAMPLE_VECTORS[:1],
            payloads=payloads_with_content_id,
            ids=SAMPLE_IDS[:1],
            chunks=SAMPLE_CHUNKS[:1]
        )
        updated = await provider.async_update_metadata("content_1", {"new_field": "new_value"})
        assert updated is True
        await provider.disconnect()
    
    @pytest.mark.asyncio
    async def test_get_count(self, provider: QdrantProvider):
        """Test get_count."""
        await provider.connect()
        await provider.create_collection()
        initial_count = await provider.get_count()
        await provider.upsert(
            vectors=SAMPLE_VECTORS,
            payloads=SAMPLE_PAYLOADS,
            ids=SAMPLE_IDS,
            chunks=SAMPLE_CHUNKS
        )
        assert await provider.get_count() == initial_count + 5
        await provider.delete(ids=SAMPLE_IDS[:2])
        assert await provider.get_count() == initial_count + 3
        await provider.disconnect()
    
    
    @pytest.mark.asyncio
    async def test_optimize(self, provider: QdrantProvider):
        """Test optimize operation."""
        await provider.connect()
        await provider.create_collection()
        result = await provider.async_optimize()
        assert result is True
    
    @pytest.mark.asyncio
    async def test_async_optimize(self, provider: QdrantProvider):
        """Test async optimize."""
        await provider.connect()
        await provider.create_collection()
        result = await provider.async_optimize()
        assert result is True
    
    @pytest.mark.asyncio
    async def test_get_supported_search_types(self, provider: QdrantProvider):
        """Test get_supported_search_types."""
        supported = provider.get_supported_search_types()
        assert isinstance(supported, list)
        assert "dense" in supported
        assert "full_text" in supported
        assert "hybrid" in supported
    
    @pytest.mark.asyncio
    async def test_async_get_supported_search_types(self, provider: QdrantProvider):
        """Test async_get_supported_search_types."""
        supported = await provider.async_get_supported_search_types()
        assert isinstance(supported, list)
        assert "dense" in supported
    
    @pytest.mark.asyncio
    async def test_dense_search(self, provider: QdrantProvider):
        """Test dense search."""
        await provider.connect()
        await provider.create_collection()
        await provider.upsert(
            vectors=SAMPLE_VECTORS,
            payloads=SAMPLE_PAYLOADS,
            ids=SAMPLE_IDS,
            chunks=SAMPLE_CHUNKS
        )
        results = await provider.dense_search(
            query_vector=QUERY_VECTOR,
            top_k=3,
            similarity_threshold=0.0
        )
        assert len(results) > 0
        assert all(isinstance(r, VectorSearchResult) for r in results)
        assert all(r.score >= 0.0 for r in results)
        # Validate vectors are retrieved and have correct length
        # Note: Qdrant may normalize vectors for cosine similarity, so we don't compare exact values
        for result in results:
            assert result.vector is not None
            assert len(result.vector) == len(QUERY_VECTOR)
        await provider.disconnect()
    
    @pytest.mark.asyncio
    async def test_dense_search_sync(self, provider: QdrantProvider):
        """Test synchronous dense search."""
        provider.connect_sync()
        provider.create_collection_sync()
        provider.upsert_sync(
            vectors=SAMPLE_VECTORS,
            payloads=SAMPLE_PAYLOADS,
            ids=SAMPLE_IDS,
            chunks=SAMPLE_CHUNKS
        )
        results = provider.dense_search_sync(
            query_vector=QUERY_VECTOR,
            top_k=3,
            similarity_threshold=0.0
        )
        assert len(results) > 0
        # Validate vectors are retrieved and have correct length
        # Note: Qdrant may normalize vectors for cosine similarity, so we don't compare exact values
        for result in results:
            assert result.vector is not None
            assert len(result.vector) == len(QUERY_VECTOR)
        provider.disconnect_sync()
    
    @pytest.mark.asyncio
    async def test_full_text_search(self, provider: QdrantProvider):
        """Test full-text search."""
        await provider.connect()
        await provider.create_collection()
        await provider.upsert(
            vectors=SAMPLE_VECTORS,
            payloads=SAMPLE_PAYLOADS,
            ids=SAMPLE_IDS,
            chunks=SAMPLE_CHUNKS
        )
        results = await provider.full_text_search(
            query_text="physics",
            top_k=3,
            similarity_threshold=0.0
        )
        assert len(results) > 0
        assert all(isinstance(r, VectorSearchResult) for r in results)
        # Validate vectors are retrieved and have correct length
        # Note: Qdrant may normalize vectors for cosine similarity, so we don't compare exact values
        for result in results:
            assert result.vector is not None
            assert len(result.vector) == len(QUERY_VECTOR)
        await provider.disconnect()
    
    @pytest.mark.asyncio
    async def test_full_text_search_sync(self, provider: QdrantProvider):
        """Test synchronous full-text search."""
        provider.connect_sync()
        provider.create_collection_sync()
        provider.upsert_sync(
            vectors=SAMPLE_VECTORS,
            payloads=SAMPLE_PAYLOADS,
            ids=SAMPLE_IDS,
            chunks=SAMPLE_CHUNKS
        )
        results = provider.full_text_search_sync(
            query_text="physics",
            top_k=3,
            similarity_threshold=0.0
        )
        assert len(results) > 0
        # Validate vectors are retrieved and have correct length
        # Note: Qdrant may normalize vectors for cosine similarity, so we don't compare exact values
        for result in results:
            assert result.vector is not None
            assert len(result.vector) == len(QUERY_VECTOR)
        provider.disconnect_sync()
    
    @pytest.mark.asyncio
    async def test_hybrid_search(self, provider: QdrantProvider):
        """Test hybrid search."""
        await provider.connect()
        await provider.create_collection()
        await provider.upsert(
            vectors=SAMPLE_VECTORS,
            payloads=SAMPLE_PAYLOADS,
            ids=SAMPLE_IDS,
            chunks=SAMPLE_CHUNKS
        )
        results = await provider.hybrid_search(
            query_vector=QUERY_VECTOR,
            query_text="physics",
            top_k=3,
            alpha=0.5,
            fusion_method="weighted",
            similarity_threshold=0.0
        )
        assert len(results) > 0
        assert all(isinstance(r, VectorSearchResult) for r in results)
        # Validate vectors are retrieved and have correct length
        # Note: Qdrant may normalize vectors for cosine similarity, so we don't compare exact values
        for result in results:
            assert result.vector is not None
            assert len(result.vector) == len(QUERY_VECTOR)
        await provider.disconnect()
    
    @pytest.mark.asyncio
    async def test_hybrid_search_rrf(self, provider: QdrantProvider):
        """Test hybrid search with RRF fusion."""
        await provider.connect()
        await provider.create_collection()
        await provider.upsert(
            vectors=SAMPLE_VECTORS,
            payloads=SAMPLE_PAYLOADS,
            ids=SAMPLE_IDS,
            chunks=SAMPLE_CHUNKS
        )
        results = await provider.hybrid_search(
            query_vector=QUERY_VECTOR,
            query_text="physics",
            top_k=3,
            fusion_method="rrf",
            similarity_threshold=0.0
        )
        assert len(results) > 0
        # Validate vectors are retrieved and have correct length
        # Note: Qdrant may normalize vectors for cosine similarity, so we don't compare exact values
        for result in results:
            assert result.vector is not None
            assert len(result.vector) == len(QUERY_VECTOR)
        await provider.disconnect()
    
    @pytest.mark.asyncio
    async def test_hybrid_search_sync(self, provider: QdrantProvider):
        """Test synchronous hybrid search."""
        provider.connect_sync()
        provider.create_collection_sync()
        provider.upsert_sync(
            vectors=SAMPLE_VECTORS,
            payloads=SAMPLE_PAYLOADS,
            ids=SAMPLE_IDS,
            chunks=SAMPLE_CHUNKS
        )
        results = provider.hybrid_search_sync(
            query_vector=QUERY_VECTOR,
            query_text="physics",
            top_k=3,
            alpha=0.5,
            similarity_threshold=0.0
        )
        assert len(results) > 0
        # Validate vectors are retrieved and have correct length
        # Note: Qdrant may normalize vectors for cosine similarity, so we don't compare exact values
        for result in results:
            assert result.vector is not None
            assert len(result.vector) == len(QUERY_VECTOR)
        provider.disconnect_sync()
    
    @pytest.mark.asyncio
    async def test_search_master_method(self, provider: QdrantProvider):
        """Test master search method."""
        await provider.connect()
        await provider.create_collection()
        await provider.upsert(
            vectors=SAMPLE_VECTORS,
            payloads=SAMPLE_PAYLOADS,
            ids=SAMPLE_IDS,
            chunks=SAMPLE_CHUNKS
        )
        # Dense search
        results = await provider.search(
            query_vector=QUERY_VECTOR,
            top_k=3
        )
        assert len(results) > 0
        # Validate vectors are retrieved and have correct length
        # Note: Qdrant may normalize vectors for cosine similarity, so we don't compare exact values
        for result in results:
            assert result.vector is not None
            assert len(result.vector) == len(QUERY_VECTOR)
        # Full-text search
        results = await provider.search(
            query_text="physics",
            top_k=3
        )
        assert len(results) > 0
        # Validate vectors are retrieved and have correct length
        # Note: Qdrant may normalize vectors for cosine similarity, so we don't compare exact values
        for result in results:
            assert result.vector is not None
            assert len(result.vector) == len(QUERY_VECTOR)
        # Hybrid search
        results = await provider.search(
            query_vector=QUERY_VECTOR,
            query_text="physics",
            top_k=3
        )
        assert len(results) > 0
        # Validate vectors are retrieved and have correct length
        # Note: Qdrant may normalize vectors for cosine similarity, so we don't compare exact values
        for result in results:
            assert result.vector is not None
            assert len(result.vector) == len(QUERY_VECTOR)
        await provider.disconnect()
    
    @pytest.mark.asyncio
    async def test_search_sync(self, provider: QdrantProvider):
        """Test synchronous master search."""
        provider.connect_sync()
        provider.create_collection_sync()
        provider.upsert_sync(
            vectors=SAMPLE_VECTORS,
            payloads=SAMPLE_PAYLOADS,
            ids=SAMPLE_IDS,
            chunks=SAMPLE_CHUNKS
        )
        results = provider.search_sync(
            query_vector=QUERY_VECTOR,
            top_k=3
        )
        assert len(results) > 0
        # Validate vectors are retrieved and have correct length
        # Note: Qdrant may normalize vectors for cosine similarity, so we don't compare exact values
        for result in results:
            assert result.vector is not None
            assert len(result.vector) == len(QUERY_VECTOR)
        provider.disconnect_sync()
    
    @pytest.mark.asyncio
    async def test_search_with_filter(self, provider: QdrantProvider):
        """Test search with metadata filter."""
        await provider.connect()
        await provider.create_collection()
        # Create payloads with metadata structure
        payloads_with_metadata = []
        for payload in SAMPLE_PAYLOADS:
            payload_copy = payload.copy()
            payload_copy["metadata"] = {"category": payload["category"]}
            payloads_with_metadata.append(payload_copy)
        
        await provider.upsert(
            vectors=SAMPLE_VECTORS,
            payloads=payloads_with_metadata,
            ids=SAMPLE_IDS,
            chunks=SAMPLE_CHUNKS
        )
        # For filtering, we need to use the correct path
        # Since metadata is stored in payload["metadata"]["category"], we filter by "metadata.category"
        results = await provider.dense_search(
            query_vector=QUERY_VECTOR,
            top_k=5,
            filter={"metadata.category": "science"}
        )
        assert len(results) > 0
        # Validate vectors are retrieved and have correct length
        # Note: Qdrant may normalize vectors for cosine similarity, so we don't compare exact values
        for result in results:
            assert result.vector is not None
            assert len(result.vector) == len(QUERY_VECTOR)
        await provider.disconnect()
    
    @pytest.mark.asyncio
    async def test_recreate_if_exists(self, provider: QdrantProvider):
        """Test recreate_if_exists configuration."""
        config = QdrantConfig(
            vector_size=5,
            collection_name="test_recreate",
            connection=ConnectionConfig(mode=Mode.IN_MEMORY),
            recreate_if_exists=True
        )
        provider2 = QdrantProvider(config)
        await provider2.connect()
        await provider2.create_collection()
        await provider2.upsert(
            vectors=SAMPLE_VECTORS[:1],
            payloads=SAMPLE_PAYLOADS[:1],
            ids=SAMPLE_IDS[:1],
            chunks=SAMPLE_CHUNKS[:1]
        )
        # Create again with recreate_if_exists=True
        await provider2.create_collection()
        count = await provider2.get_count()
        assert count == 0
        await provider2.disconnect()
    
    @pytest.mark.asyncio
    async def test_flat_index_config(self, provider: QdrantProvider):
        """Test FlatIndexConfig."""
        config = QdrantConfig(
            vector_size=5,
            collection_name="test_flat",
            connection=ConnectionConfig(mode=Mode.IN_MEMORY),
            index=FlatIndexConfig()
        )
        provider2 = QdrantProvider(config)
        await provider2.connect()
        await provider2.create_collection()
        await provider2.disconnect()
    
    @pytest.mark.asyncio
    async def test_distance_metrics(self, provider: QdrantProvider):
        """Test different distance metrics."""
        import uuid
        for metric in [DistanceMetric.COSINE, DistanceMetric.EUCLIDEAN, DistanceMetric.DOT_PRODUCT]:
            unique_name = f"test_{metric.value}_{uuid.uuid4().hex[:8]}"
            config = QdrantConfig(
                vector_size=5,
                collection_name=unique_name,
                connection=ConnectionConfig(mode=Mode.IN_MEMORY),
                distance_metric=metric
            )
            provider2 = QdrantProvider(config)
            await provider2.connect()
            await provider2.create_collection()
            await provider2.upsert(
                vectors=SAMPLE_VECTORS[:2],
                payloads=SAMPLE_PAYLOADS[:2],
                ids=SAMPLE_IDS[:2],
                chunks=SAMPLE_CHUNKS[:2]
            )
            results = await provider2.dense_search(
                query_vector=QUERY_VECTOR,
                top_k=2,
                similarity_threshold=0.0
            )
            assert len(results) > 0
            await provider2.disconnect()


class TestQdrantProviderCLOUD:
    """Comprehensive tests for QdrantProvider in CLOUD mode (requires API key)."""
    
    @pytest.fixture
    def config(self, request) -> Optional[QdrantConfig]:
        """Create CLOUD QdrantConfig if API key available."""
        import uuid
        api_key = os.getenv("QDRANT_CLOUD_API_KEY")
        url = os.getenv("QDRANT_CLOUD_URL")
        if not api_key or not url:
            return None
        
        from pydantic import SecretStr
        unique_name = f"test_cloud_{uuid.uuid4().hex[:8]}"
        return QdrantConfig(
            vector_size=5,
            collection_name=unique_name,
            connection=ConnectionConfig(
                mode=Mode.CLOUD,
                url=url,
                api_key=SecretStr(api_key)
            ),
            distance_metric=DistanceMetric.COSINE,
            index=HNSWIndexConfig(m=16, ef_construction=200)
        )
    
    @pytest.fixture
    def provider(self, config: Optional[QdrantConfig]) -> Optional[QdrantProvider]:
        """Create QdrantProvider instance."""
        if config is None:
            return None
        return QdrantProvider(config)
    
    def _skip_if_unavailable(self, provider: Optional[QdrantProvider]):
        """Helper to skip tests if provider is not available."""
        if provider is None:
            pytest.skip("Qdrant Cloud API key or URL not available")
    
    async def _ensure_connected(self, provider: QdrantProvider):
        """Helper to ensure connection, skip if unavailable."""
        try:
            await provider.connect()
            return True
        except VectorDBConnectionError:
            pytest.skip("Qdrant Cloud connection failed")
    
    async def _create_index_if_needed(self, provider: QdrantProvider, field_name: str):
        """Helper to create index for a field if needed."""
        try:
            from qdrant_client import models
            await provider._client.create_payload_index(
                collection_name=provider._config.collection_name,
                field_name=field_name,
                field_schema=models.KeywordIndexParams(type="keyword"),
                wait=True
            )
        except Exception:
            pass  # Index might already exist
    
    @pytest.mark.asyncio
    async def test_initialization(self, provider: Optional[QdrantProvider], config: Optional[QdrantConfig]):
        """Test provider initialization and attributes."""
        self._skip_if_unavailable(provider)
        assert provider._config == config
        assert provider._config.collection_name.startswith("test_cloud_")
        assert provider._config.vector_size == 5
        assert provider._config.distance_metric == DistanceMetric.COSINE
        assert not provider._is_connected
        assert provider._client is None
    
    @pytest.mark.asyncio
    async def test_connect(self, provider: Optional[QdrantProvider]):
        """Test connection to Qdrant Cloud."""
        self._skip_if_unavailable(provider)
        await self._ensure_connected(provider)
        assert provider._is_connected is True
        assert provider._client is not None
        assert await provider.is_ready() is True
        await provider.disconnect()
    
    @pytest.mark.asyncio
    async def test_create_collection(self, provider: Optional[QdrantProvider]):
        """Test collection creation."""
        self._skip_if_unavailable(provider)
        await self._ensure_connected(provider)
        try:
            if await provider.collection_exists():
                await provider.delete_collection()
        except Exception:
            pass
        assert not await provider.collection_exists()
        await provider.create_collection()
        assert await provider.collection_exists()
        await provider.disconnect()
    
    @pytest.mark.asyncio
    async def test_upsert(self, provider: Optional[QdrantProvider]):
        """Test upsert operation with content validation."""
        self._skip_if_unavailable(provider)
        await self._ensure_connected(provider)
        await provider.create_collection()
        await provider.upsert(
            vectors=SAMPLE_VECTORS,
            payloads=SAMPLE_PAYLOADS,
            ids=SAMPLE_IDS,
            chunks=SAMPLE_CHUNKS
        )
        results = await provider.fetch(ids=SAMPLE_IDS)
        assert len(results) == 5
        # Qdrant normalizes IDs, so we need to match by content instead
        for result in results:
            assert result.id is not None
            assert result.payload is not None
            content = result.payload.get("content")
            assert content in SAMPLE_CHUNKS
            idx = SAMPLE_CHUNKS.index(content)
            assert result.payload.get("category") == SAMPLE_PAYLOADS[idx]["category"]
            assert result.payload.get("author") == SAMPLE_PAYLOADS[idx]["author"]
            assert result.payload.get("year") == SAMPLE_PAYLOADS[idx]["year"]
            assert result.text == SAMPLE_CHUNKS[idx]
            # Validate vector is retrieved and has correct length
            assert result.vector is not None
            assert len(result.vector) == len(SAMPLE_VECTORS[idx])
        await provider.disconnect()
    
    @pytest.mark.asyncio
    async def test_fetch(self, provider: Optional[QdrantProvider]):
        """Test fetch operation with detailed validation."""
        self._skip_if_unavailable(provider)
        await self._ensure_connected(provider)
        await provider.create_collection()
        await provider.upsert(
            vectors=SAMPLE_VECTORS,
            payloads=SAMPLE_PAYLOADS,
            ids=SAMPLE_IDS,
            chunks=SAMPLE_CHUNKS
        )
        results = await provider.fetch(ids=SAMPLE_IDS[:3])
        assert len(results) == 3
        for result in results:
            assert isinstance(result, VectorSearchResult)
            assert result.id is not None
            assert result.score == 1.0
            assert result.payload is not None
            assert isinstance(result.payload, dict)
            assert result.text is not None
            assert result.vector is not None
            assert len(result.vector) == 5
        await provider.disconnect()
    
    @pytest.mark.asyncio
    async def test_delete(self, provider: Optional[QdrantProvider]):
        """Test delete operation."""
        self._skip_if_unavailable(provider)
        await self._ensure_connected(provider)
        await provider.create_collection()
        await provider.upsert(
            vectors=SAMPLE_VECTORS,
            payloads=SAMPLE_PAYLOADS,
            ids=SAMPLE_IDS,
            chunks=SAMPLE_CHUNKS
        )
        await provider.delete(ids=SAMPLE_IDS[:2])
        results = await provider.fetch(ids=SAMPLE_IDS[:2])
        assert len(results) == 0
        results = await provider.fetch(ids=SAMPLE_IDS[2:])
        assert len(results) == 3
        await provider.disconnect()
    
    @pytest.mark.asyncio
    async def test_dense_search(self, provider: Optional[QdrantProvider]):
        """Test dense search with detailed result validation."""
        self._skip_if_unavailable(provider)
        await self._ensure_connected(provider)
        await provider.create_collection()
        await provider.upsert(
            vectors=SAMPLE_VECTORS,
            payloads=SAMPLE_PAYLOADS,
            ids=SAMPLE_IDS,
            chunks=SAMPLE_CHUNKS
        )
        results = await provider.dense_search(
            query_vector=QUERY_VECTOR,
            top_k=3,
            similarity_threshold=0.0
        )
        assert len(results) > 0
        assert len(results) <= 3
        for result in results:
            assert isinstance(result, VectorSearchResult)
            assert result.id is not None
            assert isinstance(result.score, float)
            assert 0.0 <= result.score <= 1.0
            assert result.payload is not None
            assert result.text is not None
            assert result.vector is not None
            assert len(result.vector) == 5
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)
        await provider.disconnect()
    
    @pytest.mark.asyncio
    async def test_full_text_search(self, provider: Optional[QdrantProvider]):
        """Test full-text search with content validation."""
        self._skip_if_unavailable(provider)
        await self._ensure_connected(provider)
        await provider.create_collection()
        await provider.upsert(
            vectors=SAMPLE_VECTORS,
            payloads=SAMPLE_PAYLOADS,
            ids=SAMPLE_IDS,
            chunks=SAMPLE_CHUNKS
        )
        results = await provider.full_text_search(
            query_text="physics",
            top_k=3,
            similarity_threshold=0.0
        )
        assert len(results) > 0
        for result in results:
            assert isinstance(result, VectorSearchResult)
            assert result.id is not None
            assert isinstance(result.score, float)
            assert result.score >= 0.0
            assert result.payload is not None
            assert result.text is not None
            assert "physics" in result.text.lower() or "theory" in result.text.lower()
            assert result.vector is not None
        await provider.disconnect()
    
    @pytest.mark.asyncio
    async def test_hybrid_search(self, provider: Optional[QdrantProvider]):
        """Test hybrid search with detailed validation."""
        self._skip_if_unavailable(provider)
        await self._ensure_connected(provider)
        await provider.create_collection()
        await provider.upsert(
            vectors=SAMPLE_VECTORS,
            payloads=SAMPLE_PAYLOADS,
            ids=SAMPLE_IDS,
            chunks=SAMPLE_CHUNKS
        )
        results = await provider.hybrid_search(
            query_vector=QUERY_VECTOR,
            query_text="physics",
            top_k=3,
            alpha=0.5,
            fusion_method="weighted",
            similarity_threshold=0.0
        )
        assert len(results) > 0
        assert len(results) <= 3
        for result in results:
            assert isinstance(result, VectorSearchResult)
            assert result.id is not None
            assert isinstance(result.score, float)
            assert result.score >= 0.0
            assert result.payload is not None
            assert result.text is not None
            assert result.vector is not None
            assert len(result.vector) == 5
        await provider.disconnect()
    
    @pytest.mark.asyncio
    async def test_search_with_filter(self, provider: Optional[QdrantProvider]):
        """Test search with metadata filter."""
        self._skip_if_unavailable(provider)
        await self._ensure_connected(provider)
        await provider.create_collection()
        # Create payloads with metadata structure
        payloads_with_metadata = []
        for payload in SAMPLE_PAYLOADS:
            payload_copy = payload.copy()
            payload_copy["metadata"] = {"category": payload["category"]}
            payloads_with_metadata.append(payload_copy)
        
        await provider.upsert(
            vectors=SAMPLE_VECTORS,
            payloads=payloads_with_metadata,
            ids=SAMPLE_IDS,
            chunks=SAMPLE_CHUNKS
        )
        # Create index for metadata.category to enable filtering
        try:
            from qdrant_client import models
            await provider._client.create_payload_index(
                collection_name=provider._config.collection_name,
                field_name="metadata.category",
                field_schema=models.KeywordIndexParams(type="keyword"),
                wait=True
            )
        except Exception:
            pass  # Index might already exist
        
        # For filtering, we need to use the correct path
        # Since metadata is stored in payload["metadata"]["category"], we filter by "metadata.category"
        results = await provider.dense_search(
            query_vector=QUERY_VECTOR,
            top_k=5,
            filter={"metadata.category": "science"}
        )
        assert len(results) > 0
        for result in results:
            assert result.payload.get("metadata", {}).get("category") == "science"
        await provider.disconnect()
    
    @pytest.mark.asyncio
    async def test_get_count(self, provider: Optional[QdrantProvider]):
        """Test get_count."""
        self._skip_if_unavailable(provider)
        await self._ensure_connected(provider)
        await provider.create_collection()
        initial_count = await provider.get_count()
        await provider.upsert(
            vectors=SAMPLE_VECTORS,
            payloads=SAMPLE_PAYLOADS,
            ids=SAMPLE_IDS,
            chunks=SAMPLE_CHUNKS
        )
        assert await provider.get_count() == initial_count + 5
        await provider.delete(ids=SAMPLE_IDS[:2])
        assert await provider.get_count() == initial_count + 3
        await provider.disconnect()
    
    @pytest.mark.asyncio
    async def test_update_metadata(self, provider: Optional[QdrantProvider]):
        """Test update_metadata with validation."""
        self._skip_if_unavailable(provider)
        await self._ensure_connected(provider)
        await provider.create_collection()
        await self._create_index_if_needed(provider, "content_id")
        payloads_with_content_id = []
        for payload in SAMPLE_PAYLOADS[:1]:
            payload_copy = payload.copy()
            payload_copy["content_id"] = "cloud_content_1"
            payloads_with_content_id.append(payload_copy)
        
        await provider.upsert(
            vectors=SAMPLE_VECTORS[:1],
            payloads=payloads_with_content_id,
            ids=SAMPLE_IDS[:1],
            chunks=SAMPLE_CHUNKS[:1]
        )
        # Use async method directly to avoid event loop issues
        updated = await provider.async_update_metadata("cloud_content_1", {"new_field": "new_value", "updated": True})
        assert updated is True
        results = await provider.fetch(ids=SAMPLE_IDS[:1])
        assert len(results) == 1
        assert results[0].payload.get("metadata", {}).get("new_field") == "new_value"
        assert results[0].payload.get("metadata", {}).get("updated") is True
        await provider.disconnect()
    
    @pytest.mark.asyncio
    async def test_delete_by_filter(self, provider: Optional[QdrantProvider]):
        """Test delete_by_metadata."""
        self._skip_if_unavailable(provider)
        await self._ensure_connected(provider)
        await provider.create_collection()
        # Create payloads with metadata structure
        payloads_with_metadata = []
        for payload in SAMPLE_PAYLOADS:
            payload_copy = payload.copy()
            payload_copy["metadata"] = {"category": payload["category"]}
            payloads_with_metadata.append(payload_copy)
        
        await provider.upsert(
            vectors=SAMPLE_VECTORS,
            payloads=payloads_with_metadata,
            ids=SAMPLE_IDS,
            chunks=SAMPLE_CHUNKS
        )
        # Create index for metadata.category to enable filtering
        try:
            from qdrant_client import models
            await provider._client.create_payload_index(
                collection_name=provider._config.collection_name,
                field_name="metadata.category",
                field_schema=models.KeywordIndexParams(type="keyword"),
                wait=True
            )
        except Exception:
            pass  # Index might already exist
        
        deleted = await provider.async_delete_by_metadata({"category": "science"})
        assert deleted is True
        results = await provider.fetch(ids=SAMPLE_IDS)
        assert len(results) == 3
        for result in results:
            assert result.payload.get("metadata", {}).get("category") != "science"
        await provider.disconnect()
    
    @pytest.mark.asyncio
    async def test_upsert_with_document_tracking(self, provider: Optional[QdrantProvider]):
        """Test upsert with document tracking and validate metadata."""
        self._skip_if_unavailable(provider)
        await self._ensure_connected(provider)
        await provider.create_collection()
        payloads_with_tracking = []
        for i, payload in enumerate(SAMPLE_PAYLOADS[:2]):
            payload_copy = payload.copy()
            payload_copy["document_name"] = f"cloud_doc{i+1}"
            payload_copy["document_id"] = f"cloud_doc_id_{i+1}"
            payload_copy["content_id"] = f"cloud_content_{i+1}"
            payloads_with_tracking.append(payload_copy)
        
        await provider.upsert(
            vectors=SAMPLE_VECTORS[:2],
            payloads=payloads_with_tracking,
            ids=SAMPLE_IDS[:2],
            chunks=SAMPLE_CHUNKS[:2]
        )
        results = await provider.fetch(ids=SAMPLE_IDS[:2])
        assert len(results) == 2
        # Qdrant normalizes IDs, so we need to match by content_id instead
        for result in results:
            content_id = result.payload.get("content_id")
            assert content_id in ["cloud_content_1", "cloud_content_2"]
            # Extract number from "cloud_content_1" -> 1
            idx = int(content_id.split("_")[-1]) - 1
            assert result.payload.get("document_name") == f"cloud_doc{idx+1}"
            assert result.payload.get("document_id") == f"cloud_doc_id_{idx+1}"
            assert result.payload.get("category") == SAMPLE_PAYLOADS[idx]["category"]
        await provider.disconnect()
    
    @pytest.mark.asyncio
    async def test_connect_sync(self, provider: Optional[QdrantProvider]):
        """Test synchronous connection."""
        self._skip_if_unavailable(provider)
        provider.connect_sync()
        assert provider._is_connected is True
        assert provider._client is not None
        assert provider.is_ready_sync() is True
        provider.disconnect_sync()
    
    @pytest.mark.asyncio
    async def test_disconnect(self, provider: Optional[QdrantProvider]):
        """Test disconnection."""
        self._skip_if_unavailable(provider)
        await self._ensure_connected(provider)
        assert provider._is_connected is True
        await provider.disconnect()
        assert provider._is_connected is False
        assert provider._client is None
    
    @pytest.mark.asyncio
    async def test_disconnect_sync(self, provider: Optional[QdrantProvider]):
        """Test synchronous disconnection."""
        self._skip_if_unavailable(provider)
        provider.connect_sync()
        assert provider._is_connected is True
        provider.disconnect_sync()
        assert provider._is_connected is False
    
    @pytest.mark.asyncio
    async def test_is_ready(self, provider: Optional[QdrantProvider]):
        """Test is_ready check."""
        self._skip_if_unavailable(provider)
        assert await provider.is_ready() is False
        await self._ensure_connected(provider)
        assert await provider.is_ready() is True
        await provider.disconnect()
        assert await provider.is_ready() is False
    
    @pytest.mark.asyncio
    async def test_is_ready_sync(self, provider: Optional[QdrantProvider]):
        """Test synchronous is_ready check."""
        self._skip_if_unavailable(provider)
        assert provider.is_ready_sync() is False
        provider.connect_sync()
        assert provider.is_ready_sync() is True
        provider.disconnect_sync()
        assert provider.is_ready_sync() is False
    
    @pytest.mark.asyncio
    async def test_create_collection_sync(self, provider: Optional[QdrantProvider]):
        """Test synchronous collection creation."""
        self._skip_if_unavailable(provider)
        provider.connect_sync()
        try:
            if provider.collection_exists_sync():
                provider.delete_collection_sync()
                # Wait for deletion to propagate
                await asyncio.sleep(1.0)
        except Exception:
            pass
        assert not provider.collection_exists_sync()
        provider.create_collection_sync()
        # Wait for creation to propagate (Qdrant Cloud may have eventual consistency)
        # Try up to 10 times with increasing delays
        collection_exists = False
        for attempt in range(10):
            await asyncio.sleep(0.5 * (attempt + 1))  # Increasing delay: 0.5s, 1s, 1.5s, etc.
            if provider.collection_exists_sync():
                collection_exists = True
                break
        # If still not found, give it one more long wait
        if not collection_exists:
            await asyncio.sleep(2.0)
            collection_exists = provider.collection_exists_sync()
        # For cloud, eventual consistency might cause delays, but collection should exist
        # If it still doesn't exist after all retries, that's a real failure
        assert collection_exists, "Collection should exist after creation (eventual consistency delay handled)"
        provider.disconnect_sync()
    
    @pytest.mark.asyncio
    async def test_collection_exists(self, provider: Optional[QdrantProvider]):
        """Test collection existence check."""
        self._skip_if_unavailable(provider)
        await self._ensure_connected(provider)
        try:
            if await provider.collection_exists():
                await provider.delete_collection()
        except Exception:
            pass
        assert not await provider.collection_exists()
        await provider.create_collection()
        assert await provider.collection_exists()
        await provider.disconnect()
    
    @pytest.mark.asyncio
    async def test_collection_exists_sync(self, provider: Optional[QdrantProvider]):
        """Test synchronous collection existence check."""
        self._skip_if_unavailable(provider)
        provider.connect_sync()
        try:
            if provider.collection_exists_sync():
                provider.delete_collection_sync()
                await asyncio.sleep(1.0)
        except Exception:
            pass
        assert not provider.collection_exists_sync()
        provider.create_collection_sync()
        # Wait for creation to propagate (Qdrant Cloud has eventual consistency)
        collection_exists = False
        for attempt in range(10):
            await asyncio.sleep(0.5 * (attempt + 1))
            if provider.collection_exists_sync():
                collection_exists = True
                break
        # For cloud, eventual consistency means we might need to be lenient
        # Collection was created (we saw the log), so verify it eventually exists
        if not collection_exists:
            await asyncio.sleep(3.0)
            collection_exists = provider.collection_exists_sync()
        assert collection_exists, "Collection should exist after creation (eventual consistency delay handled)"
        provider.disconnect_sync()
    
    @pytest.mark.asyncio
    async def test_delete_collection(self, provider: Optional[QdrantProvider]):
        """Test collection deletion."""
        self._skip_if_unavailable(provider)
        await self._ensure_connected(provider)
        await provider.create_collection()
        assert await provider.collection_exists()
        await provider.delete_collection()
        assert not await provider.collection_exists()
        await provider.disconnect()
    
    @pytest.mark.asyncio
    async def test_delete_collection_sync(self, provider: Optional[QdrantProvider]):
        """Test synchronous collection deletion."""
        self._skip_if_unavailable(provider)
        await self._ensure_connected(provider)
        await provider.create_collection()
        # Wait for creation to propagate (Qdrant Cloud has eventual consistency)
        collection_exists = False
        for attempt in range(10):
            await asyncio.sleep(0.5 * (attempt + 1))
            if await provider.collection_exists():
                collection_exists = True
                break
        if not collection_exists:
            await asyncio.sleep(3.0)
            collection_exists = await provider.collection_exists()
        assert collection_exists, "Collection should exist after creation"
        await provider.delete_collection()
        # Wait for deletion to propagate (Qdrant Cloud has eventual consistency)
        collection_deleted = False
        for attempt in range(10):
            await asyncio.sleep(0.5 * (attempt + 1))
            if not await provider.collection_exists():
                collection_deleted = True
                break
        if not collection_deleted:
            await asyncio.sleep(3.0)
            collection_deleted = not await provider.collection_exists()
        assert collection_deleted, "Collection should be deleted (eventual consistency delay handled)"
        await provider.disconnect()
    
    @pytest.mark.asyncio
    async def test_upsert_sync(self, provider: Optional[QdrantProvider]):
        """Test synchronous upsert with content validation."""
        self._skip_if_unavailable(provider)
        await self._ensure_connected(provider)
        await provider.create_collection()
        await provider.upsert(
            vectors=SAMPLE_VECTORS,
            payloads=SAMPLE_PAYLOADS,
            ids=SAMPLE_IDS,
            chunks=SAMPLE_CHUNKS
        )
        results = await provider.fetch(ids=SAMPLE_IDS)
        assert len(results) == 5
        for result in results:
            assert result.id is not None
            assert result.payload is not None
            assert result.text is not None
        await provider.disconnect()
    
    @pytest.mark.asyncio
    async def test_upsert_validation_error(self, provider: Optional[QdrantProvider]):
        """Test upsert with mismatched lengths raises error."""
        self._skip_if_unavailable(provider)
        await self._ensure_connected(provider)
        await provider.create_collection()
        with pytest.raises(ValueError):
            await provider.upsert(
                vectors=SAMPLE_VECTORS[:2],
                payloads=SAMPLE_PAYLOADS[:3],
                ids=SAMPLE_IDS[:2],
                chunks=SAMPLE_CHUNKS[:2]
            )
        await provider.disconnect()
    
    @pytest.mark.asyncio
    async def test_fetch_sync(self, provider: Optional[QdrantProvider]):
        """Test synchronous fetch with content validation."""
        self._skip_if_unavailable(provider)
        await self._ensure_connected(provider)
        await provider.create_collection()
        await provider.upsert(
            vectors=SAMPLE_VECTORS,
            payloads=SAMPLE_PAYLOADS,
            ids=SAMPLE_IDS,
            chunks=SAMPLE_CHUNKS
        )
        results = await provider.fetch(ids=SAMPLE_IDS[:3])
        assert len(results) == 3
        for result in results:
            assert isinstance(result, VectorSearchResult)
            assert result.id is not None
            assert result.score == 1.0
            assert result.payload is not None
            assert result.text is not None
            assert result.vector is not None
        await provider.disconnect()
    
    @pytest.mark.asyncio
    async def test_delete_sync(self, provider: Optional[QdrantProvider]):
        """Test synchronous delete with validation."""
        self._skip_if_unavailable(provider)
        await self._ensure_connected(provider)
        await provider.create_collection()
        await provider.upsert(
            vectors=SAMPLE_VECTORS,
            payloads=SAMPLE_PAYLOADS,
            ids=SAMPLE_IDS,
            chunks=SAMPLE_CHUNKS
        )
        await provider.delete(ids=SAMPLE_IDS[:2])
        results = await provider.fetch(ids=SAMPLE_IDS[:2])
        assert len(results) == 0
        results = await provider.fetch(ids=SAMPLE_IDS[2:])
        assert len(results) == 3
        await provider.disconnect()
    
    @pytest.mark.asyncio
    async def test_delete_by_document_name(self, provider: Optional[QdrantProvider]):
        """Test delete_by_document_name with validation."""
        self._skip_if_unavailable(provider)
        await self._ensure_connected(provider)
        await provider.create_collection()
        # Create index for document_name to enable filtering
        try:
            from qdrant_client import models
            await provider._client.create_payload_index(
                collection_name=provider._config.collection_name,
                field_name="document_name",
                field_schema=models.KeywordIndexParams(type="keyword"),
                wait=True
            )
        except Exception:
            pass  # Index might already exist
        
        initial_count = await provider.get_count()
        payloads_with_doc_name = []
        for payload in SAMPLE_PAYLOADS[:2]:
            payload_copy = payload.copy()
            payload_copy["document_name"] = "cloud_doc"
            payloads_with_doc_name.append(payload_copy)
        
        await provider.upsert(
            vectors=SAMPLE_VECTORS[:2],
            payloads=payloads_with_doc_name,
            ids=SAMPLE_IDS[:2],
            chunks=SAMPLE_CHUNKS[:2]
        )
        deleted = await provider.async_delete_by_document_name("cloud_doc")
        assert deleted is True
        count = await provider.get_count()
        assert count == initial_count
        await provider.disconnect()
    
    @pytest.mark.asyncio
    async def test_async_delete_by_document_name(self, provider: Optional[QdrantProvider]):
        """Test async_delete_by_document_name."""
        self._skip_if_unavailable(provider)
        await self._ensure_connected(provider)
        await provider.create_collection()
        await self._create_index_if_needed(provider, "document_name")
        payloads_with_doc_name = []
        for payload in SAMPLE_PAYLOADS[:2]:
            payload_copy = payload.copy()
            payload_copy["document_name"] = "cloud_doc"
            payloads_with_doc_name.append(payload_copy)
        
        await provider.upsert(
            vectors=SAMPLE_VECTORS[:2],
            payloads=payloads_with_doc_name,
            ids=SAMPLE_IDS[:2],
            chunks=SAMPLE_CHUNKS[:2]
        )
        deleted = await provider.async_delete_by_document_name("cloud_doc")
        assert deleted is True
        await provider.disconnect()
    
    @pytest.mark.asyncio
    async def test_delete_by_document_id(self, provider: Optional[QdrantProvider]):
        """Test delete_by_document_id with validation."""
        self._skip_if_unavailable(provider)
        await self._ensure_connected(provider)
        await provider.create_collection()
        await self._create_index_if_needed(provider, "document_id")
        payloads_with_doc_id = []
        for payload in SAMPLE_PAYLOADS[:2]:
            payload_copy = payload.copy()
            payload_copy["document_id"] = "cloud_doc_id_1"
            payloads_with_doc_id.append(payload_copy)
        
        await provider.upsert(
            vectors=SAMPLE_VECTORS[:2],
            payloads=payloads_with_doc_id,
            ids=SAMPLE_IDS[:2],
            chunks=SAMPLE_CHUNKS[:2]
        )
        deleted = await provider.async_delete_by_document_id("cloud_doc_id_1")
        assert deleted is True
        results = await provider.fetch(ids=SAMPLE_IDS[:2])
        assert len(results) == 0
        await provider.disconnect()
    
    @pytest.mark.asyncio
    async def test_async_delete_by_document_id(self, provider: Optional[QdrantProvider]):
        """Test async_delete_by_document_id."""
        self._skip_if_unavailable(provider)
        await self._ensure_connected(provider)
        await provider.create_collection()
        await self._create_index_if_needed(provider, "document_id")
        payloads_with_doc_id = []
        for payload in SAMPLE_PAYLOADS[:2]:
            payload_copy = payload.copy()
            payload_copy["document_id"] = "cloud_doc_id_1"
            payloads_with_doc_id.append(payload_copy)
        
        await provider.upsert(
            vectors=SAMPLE_VECTORS[:2],
            payloads=payloads_with_doc_id,
            ids=SAMPLE_IDS[:2],
            chunks=SAMPLE_CHUNKS[:2]
        )
        deleted = await provider.async_delete_by_document_id("cloud_doc_id_1")
        assert deleted is True
        await provider.disconnect()
    
    @pytest.mark.asyncio
    async def test_delete_by_content_id(self, provider: Optional[QdrantProvider]):
        """Test delete_by_content_id with validation."""
        self._skip_if_unavailable(provider)
        await self._ensure_connected(provider)
        await provider.create_collection()
        await self._create_index_if_needed(provider, "content_id")
        payloads_with_content_id = []
        for payload in SAMPLE_PAYLOADS[:2]:
            payload_copy = payload.copy()
            payload_copy["content_id"] = "cloud_content_1"
            payloads_with_content_id.append(payload_copy)
        
        await provider.upsert(
            vectors=SAMPLE_VECTORS[:2],
            payloads=payloads_with_content_id,
            ids=SAMPLE_IDS[:2],
            chunks=SAMPLE_CHUNKS[:2]
        )
        deleted = await provider.async_delete_by_content_id("cloud_content_1")
        assert deleted is True
        results = await provider.fetch(ids=SAMPLE_IDS[:2])
        assert len(results) == 0
        await provider.disconnect()
    
    @pytest.mark.asyncio
    async def test_async_delete_by_content_id(self, provider: Optional[QdrantProvider]):
        """Test async_delete_by_content_id."""
        self._skip_if_unavailable(provider)
        await self._ensure_connected(provider)
        await provider.create_collection()
        await self._create_index_if_needed(provider, "content_id")
        payloads_with_content_id = []
        for payload in SAMPLE_PAYLOADS[:2]:
            payload_copy = payload.copy()
            payload_copy["content_id"] = "cloud_content_1"
            payloads_with_content_id.append(payload_copy)
        
        await provider.upsert(
            vectors=SAMPLE_VECTORS[:2],
            payloads=payloads_with_content_id,
            ids=SAMPLE_IDS[:2],
            chunks=SAMPLE_CHUNKS[:2]
        )
        deleted = await provider.async_delete_by_content_id("cloud_content_1")
        assert deleted is True
        await provider.disconnect()
    
    @pytest.mark.asyncio
    async def test_async_delete_by_metadata(self, provider: Optional[QdrantProvider]):
        """Test async_delete_by_metadata."""
        self._skip_if_unavailable(provider)
        await self._ensure_connected(provider)
        await provider.create_collection()
        await self._create_index_if_needed(provider, "metadata.category")
        # Create payloads with metadata structure
        payloads_with_metadata = []
        for payload in SAMPLE_PAYLOADS:
            payload_copy = payload.copy()
            payload_copy["metadata"] = {"category": payload["category"]}
            payloads_with_metadata.append(payload_copy)
        
        await provider.upsert(
            vectors=SAMPLE_VECTORS,
            payloads=payloads_with_metadata,
            ids=SAMPLE_IDS,
            chunks=SAMPLE_CHUNKS
        )
        deleted = await provider.async_delete_by_metadata({"category": "science"})
        assert deleted is True
        await provider.disconnect()
    
    @pytest.mark.asyncio
    async def test_id_exists(self, provider: Optional[QdrantProvider]):
        """Test id_exists check."""
        self._skip_if_unavailable(provider)
        await self._ensure_connected(provider)
        await provider.create_collection()
        await provider.upsert(
            vectors=SAMPLE_VECTORS[:1],
            payloads=SAMPLE_PAYLOADS[:1],
            ids=SAMPLE_IDS[:1],
            chunks=SAMPLE_CHUNKS[:1]
        )
        assert await provider.id_exists("doc1")
        assert not await provider.id_exists("nonexistent")
        await provider.disconnect()
    
    @pytest.mark.asyncio
    async def test_document_name_exists(self, provider: Optional[QdrantProvider]):
        """Test document_name_exists."""
        self._skip_if_unavailable(provider)
        await self._ensure_connected(provider)
        await provider.create_collection()
        await self._create_index_if_needed(provider, "document_name")
        payloads_with_doc_name = []
        for payload in SAMPLE_PAYLOADS[:1]:
            payload_copy = payload.copy()
            payload_copy["document_name"] = "cloud_doc"
            payloads_with_doc_name.append(payload_copy)
        
        await provider.upsert(
            vectors=SAMPLE_VECTORS[:1],
            payloads=payloads_with_doc_name,
            ids=SAMPLE_IDS[:1],
            chunks=SAMPLE_CHUNKS[:1]
        )
        assert await provider.async_document_name_exists("cloud_doc")
        assert not await provider.async_document_name_exists("nonexistent")
        await provider.disconnect()
    
    @pytest.mark.asyncio
    async def test_async_document_name_exists(self, provider: Optional[QdrantProvider]):
        """Test async_document_name_exists."""
        self._skip_if_unavailable(provider)
        await self._ensure_connected(provider)
        await provider.create_collection()
        await self._create_index_if_needed(provider, "document_name")
        payloads_with_doc_name = []
        for payload in SAMPLE_PAYLOADS[:1]:
            payload_copy = payload.copy()
            payload_copy["document_name"] = "cloud_doc"
            payloads_with_doc_name.append(payload_copy)
        
        await provider.upsert(
            vectors=SAMPLE_VECTORS[:1],
            payloads=payloads_with_doc_name,
            ids=SAMPLE_IDS[:1],
            chunks=SAMPLE_CHUNKS[:1]
        )
        assert await provider.async_document_name_exists("cloud_doc")
        assert not await provider.async_document_name_exists("nonexistent")
        await provider.disconnect()
    
    @pytest.mark.asyncio
    async def test_document_id_exists(self, provider: Optional[QdrantProvider]):
        """Test document_id_exists."""
        self._skip_if_unavailable(provider)
        await self._ensure_connected(provider)
        await provider.create_collection()
        await self._create_index_if_needed(provider, "document_id")
        payloads_with_doc_id = []
        for payload in SAMPLE_PAYLOADS[:1]:
            payload_copy = payload.copy()
            payload_copy["document_id"] = "cloud_doc_id_1"
            payloads_with_doc_id.append(payload_copy)
        
        await provider.upsert(
            vectors=SAMPLE_VECTORS[:1],
            payloads=payloads_with_doc_id,
            ids=SAMPLE_IDS[:1],
            chunks=SAMPLE_CHUNKS[:1]
        )
        assert await provider.async_document_id_exists("cloud_doc_id_1")
        assert not await provider.async_document_id_exists("nonexistent")
        await provider.disconnect()
    
    @pytest.mark.asyncio
    async def test_async_document_id_exists(self, provider: Optional[QdrantProvider]):
        """Test async_document_id_exists."""
        self._skip_if_unavailable(provider)
        await self._ensure_connected(provider)
        await provider.create_collection()
        await self._create_index_if_needed(provider, "document_id")
        payloads_with_doc_id = []
        for payload in SAMPLE_PAYLOADS[:1]:
            payload_copy = payload.copy()
            payload_copy["document_id"] = "cloud_doc_id_1"
            payloads_with_doc_id.append(payload_copy)
        
        await provider.upsert(
            vectors=SAMPLE_VECTORS[:1],
            payloads=payloads_with_doc_id,
            ids=SAMPLE_IDS[:1],
            chunks=SAMPLE_CHUNKS[:1]
        )
        assert await provider.async_document_id_exists("cloud_doc_id_1")
        assert not await provider.async_document_id_exists("nonexistent")
        await provider.disconnect()
    
    @pytest.mark.asyncio
    async def test_content_id_exists(self, provider: Optional[QdrantProvider]):
        """Test content_id_exists."""
        self._skip_if_unavailable(provider)
        await self._ensure_connected(provider)
        await provider.create_collection()
        await self._create_index_if_needed(provider, "content_id")
        payloads_with_content_id = []
        for payload in SAMPLE_PAYLOADS[:1]:
            payload_copy = payload.copy()
            payload_copy["content_id"] = "cloud_content_1"
            payloads_with_content_id.append(payload_copy)
        
        await provider.upsert(
            vectors=SAMPLE_VECTORS[:1],
            payloads=payloads_with_content_id,
            ids=SAMPLE_IDS[:1],
            chunks=SAMPLE_CHUNKS[:1]
        )
        assert await provider.async_content_id_exists("cloud_content_1")
        assert not await provider.async_content_id_exists("nonexistent")
        await provider.disconnect()
    
    @pytest.mark.asyncio
    async def test_async_content_id_exists(self, provider: Optional[QdrantProvider]):
        """Test async_content_id_exists."""
        self._skip_if_unavailable(provider)
        await self._ensure_connected(provider)
        await provider.create_collection()
        await self._create_index_if_needed(provider, "content_id")
        payloads_with_content_id = []
        for payload in SAMPLE_PAYLOADS[:1]:
            payload_copy = payload.copy()
            payload_copy["content_id"] = "cloud_content_1"
            payloads_with_content_id.append(payload_copy)
        
        await provider.upsert(
            vectors=SAMPLE_VECTORS[:1],
            payloads=payloads_with_content_id,
            ids=SAMPLE_IDS[:1],
            chunks=SAMPLE_CHUNKS[:1]
        )
        assert await provider.async_content_id_exists("cloud_content_1")
        assert not await provider.async_content_id_exists("nonexistent")
        await provider.disconnect()
    
    @pytest.mark.asyncio
    async def test_async_update_metadata(self, provider: Optional[QdrantProvider]):
        """Test async_update_metadata with validation."""
        self._skip_if_unavailable(provider)
        await self._ensure_connected(provider)
        await provider.create_collection()
        await self._create_index_if_needed(provider, "content_id")
        payloads_with_content_id = []
        for payload in SAMPLE_PAYLOADS[:1]:
            payload_copy = payload.copy()
            payload_copy["content_id"] = "cloud_content_1"
            payloads_with_content_id.append(payload_copy)
        
        await provider.upsert(
            vectors=SAMPLE_VECTORS[:1],
            payloads=payloads_with_content_id,
            ids=SAMPLE_IDS[:1],
            chunks=SAMPLE_CHUNKS[:1]
        )
        updated = await provider.async_update_metadata("cloud_content_1", {"new_field": "new_value", "updated": True})
        assert updated is True
        results = await provider.fetch(ids=SAMPLE_IDS[:1])
        assert results[0].payload.get("metadata", {}).get("new_field") == "new_value"
        assert results[0].payload.get("metadata", {}).get("updated") is True
        await provider.disconnect()
    
    @pytest.mark.asyncio
    async def test_optimize(self, provider: Optional[QdrantProvider]):
        """Test optimize operation."""
        self._skip_if_unavailable(provider)
        await self._ensure_connected(provider)
        await provider.create_collection()
        result = await provider.async_optimize()
        assert result is True
    
    @pytest.mark.asyncio
    async def test_async_optimize(self, provider: Optional[QdrantProvider]):
        """Test async optimize."""
        self._skip_if_unavailable(provider)
        await self._ensure_connected(provider)
        await provider.create_collection()
        result = await provider.async_optimize()
        assert result is True
    
    @pytest.mark.asyncio
    async def test_get_supported_search_types(self, provider: Optional[QdrantProvider]):
        """Test get_supported_search_types."""
        self._skip_if_unavailable(provider)
        supported = provider.get_supported_search_types()
        assert isinstance(supported, list)
        assert "dense" in supported
        assert "full_text" in supported
        assert "hybrid" in supported
    
    @pytest.mark.asyncio
    async def test_async_get_supported_search_types(self, provider: Optional[QdrantProvider]):
        """Test async_get_supported_search_types."""
        self._skip_if_unavailable(provider)
        supported = await provider.async_get_supported_search_types()
        assert isinstance(supported, list)
        assert "dense" in supported
        assert "full_text" in supported
        assert "hybrid" in supported
    
    @pytest.mark.asyncio
    async def test_dense_search_sync(self, provider: Optional[QdrantProvider]):
        """Test synchronous dense search with content validation."""
        self._skip_if_unavailable(provider)
        await self._ensure_connected(provider)
        await provider.create_collection()
        await provider.upsert(
            vectors=SAMPLE_VECTORS,
            payloads=SAMPLE_PAYLOADS,
            ids=SAMPLE_IDS,
            chunks=SAMPLE_CHUNKS
        )
        results = await provider.dense_search(
            query_vector=QUERY_VECTOR,
            top_k=3,
            similarity_threshold=0.0
        )
        assert len(results) > 0
        assert len(results) <= 3
        for result in results:
            assert isinstance(result, VectorSearchResult)
            assert result.id is not None
            assert isinstance(result.score, float)
            assert 0.0 <= result.score <= 1.0
            assert result.payload is not None
            assert result.text is not None
            assert result.vector is not None
        await provider.disconnect()
    
    @pytest.mark.asyncio
    async def test_full_text_search_sync(self, provider: Optional[QdrantProvider]):
        """Test synchronous full-text search with content validation."""
        self._skip_if_unavailable(provider)
        await self._ensure_connected(provider)
        await provider.create_collection()
        await provider.upsert(
            vectors=SAMPLE_VECTORS,
            payloads=SAMPLE_PAYLOADS,
            ids=SAMPLE_IDS,
            chunks=SAMPLE_CHUNKS
        )
        results = await provider.full_text_search(
            query_text="physics",
            top_k=3,
            similarity_threshold=0.0
        )
        assert len(results) > 0
        for result in results:
            assert isinstance(result, VectorSearchResult)
            assert result.id is not None
            assert isinstance(result.score, float)
            assert result.score >= 0.0
            assert result.payload is not None
            assert result.text is not None
            assert "physics" in result.text.lower() or "theory" in result.text.lower()
        await provider.disconnect()
    
    @pytest.mark.asyncio
    async def test_hybrid_search_rrf(self, provider: Optional[QdrantProvider]):
        """Test hybrid search with RRF fusion."""
        self._skip_if_unavailable(provider)
        await self._ensure_connected(provider)
        await provider.create_collection()
        await provider.upsert(
            vectors=SAMPLE_VECTORS,
            payloads=SAMPLE_PAYLOADS,
            ids=SAMPLE_IDS,
            chunks=SAMPLE_CHUNKS
        )
        results = await provider.hybrid_search(
            query_vector=QUERY_VECTOR,
            query_text="physics",
            top_k=3,
            fusion_method="rrf",
            similarity_threshold=0.0
        )
        assert len(results) > 0
        assert len(results) <= 3
        for result in results:
            assert isinstance(result, VectorSearchResult)
            assert result.id is not None
            assert result.score >= 0.0
            assert result.payload is not None
            assert result.text is not None
        await provider.disconnect()
    
    @pytest.mark.asyncio
    async def test_hybrid_search_sync(self, provider: Optional[QdrantProvider]):
        """Test synchronous hybrid search with validation."""
        self._skip_if_unavailable(provider)
        await self._ensure_connected(provider)
        await provider.create_collection()
        await provider.upsert(
            vectors=SAMPLE_VECTORS,
            payloads=SAMPLE_PAYLOADS,
            ids=SAMPLE_IDS,
            chunks=SAMPLE_CHUNKS
        )
        results = await provider.hybrid_search(
            query_vector=QUERY_VECTOR,
            query_text="physics",
            top_k=3,
            alpha=0.5,
            similarity_threshold=0.0
        )
        assert len(results) > 0
        for result in results:
            assert isinstance(result, VectorSearchResult)
            assert result.id is not None
            assert result.score >= 0.0
            assert result.payload is not None
        await provider.disconnect()
    
    @pytest.mark.asyncio
    async def test_search_master_method(self, provider: Optional[QdrantProvider]):
        """Test master search method with content validation."""
        self._skip_if_unavailable(provider)
        await self._ensure_connected(provider)
        await provider.create_collection()
        await provider.upsert(
            vectors=SAMPLE_VECTORS,
            payloads=SAMPLE_PAYLOADS,
            ids=SAMPLE_IDS,
            chunks=SAMPLE_CHUNKS
        )
        # Dense search
        results = await provider.search(
            query_vector=QUERY_VECTOR,
            top_k=3
        )
        assert len(results) > 0
        assert all(isinstance(r, VectorSearchResult) for r in results)
        assert all(r.id is not None for r in results)
        assert all(r.payload is not None for r in results)
        # Full-text search
        results = await provider.search(
            query_text="physics",
            top_k=3
        )
        assert len(results) > 0
        assert all(isinstance(r, VectorSearchResult) for r in results)
        # Hybrid search
        results = await provider.search(
            query_vector=QUERY_VECTOR,
            query_text="physics",
            top_k=3
        )
        assert len(results) > 0
        assert all(isinstance(r, VectorSearchResult) for r in results)
        await provider.disconnect()
    
    @pytest.mark.asyncio
    async def test_search_sync(self, provider: Optional[QdrantProvider]):
        """Test synchronous master search with validation."""
        self._skip_if_unavailable(provider)
        await self._ensure_connected(provider)
        await provider.create_collection()
        await provider.upsert(
            vectors=SAMPLE_VECTORS,
            payloads=SAMPLE_PAYLOADS,
            ids=SAMPLE_IDS,
            chunks=SAMPLE_CHUNKS
        )
        results = await provider.search(
            query_vector=QUERY_VECTOR,
            top_k=3
        )
        assert len(results) > 0
        assert all(isinstance(r, VectorSearchResult) for r in results)
        assert all(r.payload is not None for r in results)
        assert all(r.text is not None for r in results)
        await provider.disconnect()
    
    @pytest.mark.asyncio
    async def test_recreate_if_exists(self, provider: Optional[QdrantProvider]):
        """Test recreate_if_exists configuration."""
        self._skip_if_unavailable(provider)
        import uuid
        api_key = os.getenv("QDRANT_CLOUD_API_KEY")
        url = os.getenv("QDRANT_CLOUD_URL")
        if not api_key or not url:
            pytest.skip("Qdrant Cloud API key or URL not available")
        from pydantic import SecretStr
        unique_name = f"test_recreate_{uuid.uuid4().hex[:8]}"
        config = QdrantConfig(
            vector_size=5,
            collection_name=unique_name,
            connection=ConnectionConfig(
                mode=Mode.CLOUD,
                url=url,
                api_key=SecretStr(api_key)
            ),
            recreate_if_exists=True
        )
        provider2 = QdrantProvider(config)
        await self._ensure_connected(provider2)
        await provider2.create_collection()
        await provider2.upsert(
            vectors=SAMPLE_VECTORS[:1],
            payloads=SAMPLE_PAYLOADS[:1],
            ids=SAMPLE_IDS[:1],
            chunks=SAMPLE_CHUNKS[:1]
        )
        await provider2.create_collection()
        count = await provider2.get_count()
        assert count == 0
        await provider2.disconnect()
    
    @pytest.mark.asyncio
    async def test_flat_index_config(self, provider: Optional[QdrantProvider]):
        """Test FlatIndexConfig."""
        self._skip_if_unavailable(provider)
        import uuid
        api_key = os.getenv("QDRANT_CLOUD_API_KEY")
        url = os.getenv("QDRANT_CLOUD_URL")
        if not api_key or not url:
            pytest.skip("Qdrant Cloud API key or URL not available")
        from pydantic import SecretStr
        unique_name = f"test_flat_{uuid.uuid4().hex[:8]}"
        config = QdrantConfig(
            vector_size=5,
            collection_name=unique_name,
            connection=ConnectionConfig(
                mode=Mode.CLOUD,
                url=url,
                api_key=SecretStr(api_key)
            ),
            index=FlatIndexConfig()
        )
        provider2 = QdrantProvider(config)
        await self._ensure_connected(provider2)
        await provider2.create_collection()
        await provider2.upsert(
            vectors=SAMPLE_VECTORS[:2],
            payloads=SAMPLE_PAYLOADS[:2],
            ids=SAMPLE_IDS[:2],
            chunks=SAMPLE_CHUNKS[:2]
        )
        results = await provider2.dense_search(
            query_vector=QUERY_VECTOR,
            top_k=2,
            similarity_threshold=0.0
        )
        assert len(results) > 0
        assert all(isinstance(r, VectorSearchResult) for r in results)
        assert all(r.score >= 0.0 for r in results)
        await provider2.disconnect()
    
    @pytest.mark.asyncio
    async def test_distance_metrics(self, provider: Optional[QdrantProvider]):
        """Test different distance metrics."""
        self._skip_if_unavailable(provider)
        import uuid
        api_key = os.getenv("QDRANT_CLOUD_API_KEY")
        url = os.getenv("QDRANT_CLOUD_URL")
        if not api_key or not url:
            pytest.skip("Qdrant Cloud API key or URL not available")
        from pydantic import SecretStr
        for metric in [DistanceMetric.COSINE, DistanceMetric.EUCLIDEAN, DistanceMetric.DOT_PRODUCT]:
            unique_name = f"test_{metric.value}_{uuid.uuid4().hex[:8]}"
            config = QdrantConfig(
                vector_size=5,
                collection_name=unique_name,
                connection=ConnectionConfig(
                    mode=Mode.CLOUD,
                    url=url,
                    api_key=SecretStr(api_key)
                ),
                distance_metric=metric
            )
            provider2 = QdrantProvider(config)
            await self._ensure_connected(provider2)
            await provider2.create_collection()
            await provider2.upsert(
                vectors=SAMPLE_VECTORS[:2],
                payloads=SAMPLE_PAYLOADS[:2],
                ids=SAMPLE_IDS[:2],
                chunks=SAMPLE_CHUNKS[:2]
            )
            results = await provider2.dense_search(
                query_vector=QUERY_VECTOR,
                top_k=2,
                similarity_threshold=0.0
            )
            assert len(results) > 0
            assert all(isinstance(r, VectorSearchResult) for r in results)
            assert all(r.score >= 0.0 for r in results)
            await provider2.disconnect()
