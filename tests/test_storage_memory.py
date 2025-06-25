"""Tests for in-memory storage backend."""

import pytest
from sifaka.storage.memory import MemoryStorage
from sifaka.core.models import SifakaResult
from sifaka.core.exceptions import StorageError


class TestMemoryStorage:
    """Test the MemoryStorage class."""

    @pytest.fixture
    def storage(self):
        """Create a fresh storage instance."""
        return MemoryStorage()

    @pytest.fixture
    def sample_result(self):
        """Create a sample SifakaResult."""
        return SifakaResult(
            original_text="Original text for testing",
            final_text="Improved text after processing",
            iterations=2,
            was_improved=True
        )

    @pytest.mark.asyncio
    async def test_initialization(self, storage):
        """Test storage initialization."""
        assert storage._storage == {}
        assert storage.size() == 0

    @pytest.mark.asyncio
    async def test_save_and_load(self, storage, sample_result):
        """Test saving and loading a result."""
        # Save result
        result_id = await storage.save(sample_result)
        assert result_id == sample_result.id
        assert storage.size() == 1

        # Load result
        loaded_result = await storage.load(result_id)
        assert loaded_result is not None
        assert loaded_result.id == sample_result.id
        assert loaded_result.original_text == sample_result.original_text
        assert loaded_result.final_text == sample_result.final_text

    @pytest.mark.asyncio
    async def test_load_nonexistent(self, storage):
        """Test loading a non-existent result."""
        result = await storage.load("nonexistent-id")
        assert result is None

    @pytest.mark.asyncio
    async def test_delete_existing(self, storage, sample_result):
        """Test deleting an existing result."""
        # Save first
        await storage.save(sample_result)
        assert storage.size() == 1

        # Delete
        deleted = await storage.delete(sample_result.id)
        assert deleted is True
        assert storage.size() == 0

        # Verify it's gone
        loaded = await storage.load(sample_result.id)
        assert loaded is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent(self, storage):
        """Test deleting a non-existent result."""
        deleted = await storage.delete("nonexistent-id")
        assert deleted is False

    @pytest.mark.asyncio
    async def test_list_empty(self, storage):
        """Test listing with empty storage."""
        ids = await storage.list()
        assert ids == []

    @pytest.mark.asyncio
    async def test_list_with_results(self, storage):
        """Test listing with multiple results."""
        # Create and save multiple results
        results = []
        for i in range(5):
            result = SifakaResult(
                original_text=f"Original {i}",
                final_text=f"Final {i}"
            )
            results.append(result)
            await storage.save(result)

        # List all
        ids = await storage.list()
        assert len(ids) == 5
        assert all(r.id in ids for r in results)

    @pytest.mark.asyncio
    async def test_list_with_pagination(self, storage):
        """Test list with limit and offset."""
        # Save 10 results
        for i in range(10):
            result = SifakaResult(
                original_text=f"Original {i}",
                final_text=f"Final {i}"
            )
            await storage.save(result)

        # Test limit
        ids = await storage.list(limit=3)
        assert len(ids) == 3

        # Test offset
        ids_offset = await storage.list(limit=3, offset=3)
        assert len(ids_offset) == 3
        assert ids[0] != ids_offset[0]  # Different results

        # Test offset beyond range
        ids_beyond = await storage.list(limit=5, offset=8)
        assert len(ids_beyond) == 2  # Only 2 remaining

    @pytest.mark.asyncio
    async def test_search_in_original_text(self, storage):
        """Test searching in original text."""
        # Create results with specific content
        result1 = SifakaResult(
            original_text="Machine learning is fascinating",
            final_text="AI is amazing"
        )
        result2 = SifakaResult(
            original_text="Deep learning networks",
            final_text="Neural networks"
        )
        result3 = SifakaResult(
            original_text="Data science",
            final_text="Analytics"
        )

        await storage.save(result1)
        await storage.save(result2)
        await storage.save(result3)

        # Search for "learning"
        matches = await storage.search("learning")
        assert len(matches) == 2
        assert result1.id in matches
        assert result2.id in matches

    @pytest.mark.asyncio
    async def test_search_in_final_text(self, storage):
        """Test searching in final text."""
        result1 = SifakaResult(
            original_text="Original 1",
            final_text="Machine learning applications"
        )
        result2 = SifakaResult(
            original_text="Original 2",
            final_text="Deep learning models"
        )

        await storage.save(result1)
        await storage.save(result2)

        # Search in final text
        matches = await storage.search("learning")
        assert len(matches) == 2

    @pytest.mark.asyncio
    async def test_search_case_insensitive(self, storage):
        """Test case-insensitive search."""
        result = SifakaResult(
            original_text="MACHINE LEARNING",
            final_text="Final text"
        )
        await storage.save(result)

        # Search with different cases
        matches1 = await storage.search("machine")
        matches2 = await storage.search("MACHINE")
        matches3 = await storage.search("MaChInE")

        assert len(matches1) == 1
        assert len(matches2) == 1
        assert len(matches3) == 1

    @pytest.mark.asyncio
    async def test_search_with_limit(self, storage):
        """Test search with result limit."""
        # Create many matching results
        for i in range(20):
            result = SifakaResult(
                original_text=f"Machine learning example {i}",
                final_text="Final"
            )
            await storage.save(result)

        # Search with limit
        matches = await storage.search("machine", limit=5)
        assert len(matches) == 5

    @pytest.mark.asyncio
    async def test_search_no_matches(self, storage, sample_result):
        """Test search with no matches."""
        await storage.save(sample_result)
        matches = await storage.search("nonexistent-term")
        assert matches == []

    @pytest.mark.asyncio
    async def test_clear(self, storage):
        """Test clearing all storage."""
        # Add multiple results
        for i in range(5):
            result = SifakaResult(
                original_text=f"Text {i}",
                final_text=f"Final {i}"
            )
            await storage.save(result)

        assert storage.size() == 5

        # Clear
        storage.clear()
        assert storage.size() == 0

        # Verify all are gone
        ids = await storage.list()
        assert ids == []

    @pytest.mark.asyncio
    async def test_save_exception_handling(self, storage, monkeypatch):
        """Test exception handling in save."""
        # Mock the storage to raise an exception
        def mock_setitem(self, key, value):
            raise RuntimeError("Storage failed")

        monkeypatch.setattr(dict, "__setitem__", mock_setitem)

        result = SifakaResult(original_text="Test", final_text="Test")
        
        with pytest.raises(StorageError) as exc_info:
            await storage.save(result)
        
        assert "Failed to save result" in str(exc_info.value)
        assert exc_info.value.storage_type == "memory"
        assert exc_info.value.operation == "save"

    @pytest.mark.asyncio
    async def test_load_exception_handling(self, storage, monkeypatch):
        """Test exception handling in load."""
        # Mock the storage to raise an exception
        def mock_get(self, key, default=None):
            raise RuntimeError("Storage failed")

        monkeypatch.setattr(dict, "get", mock_get)

        with pytest.raises(StorageError) as exc_info:
            await storage.load("test-id")
        
        assert "Failed to load result" in str(exc_info.value)
        assert exc_info.value.storage_type == "memory"
        assert exc_info.value.operation == "load"

    @pytest.mark.asyncio
    async def test_search_exception_handling(self, storage, sample_result, monkeypatch):
        """Test exception handling in search."""
        await storage.save(sample_result)

        # Mock to raise exception during iteration
        def mock_items(self):
            raise RuntimeError("Iteration failed")

        monkeypatch.setattr(dict, "items", mock_items)

        with pytest.raises(StorageError) as exc_info:
            await storage.search("test")
        
        assert "Search failed for query" in str(exc_info.value)
        assert exc_info.value.storage_type == "memory"
        assert exc_info.value.operation == "search"

    @pytest.mark.asyncio
    async def test_multiple_operations(self, storage):
        """Test multiple operations in sequence."""
        # Save multiple
        results = []
        for i in range(3):
            result = SifakaResult(
                original_text=f"Text {i}",
                final_text=f"Improved {i}"
            )
            results.append(result)
            await storage.save(result)

        # List
        ids = await storage.list()
        assert len(ids) == 3

        # Search
        matches = await storage.search("Text")
        assert len(matches) == 3

        # Delete one
        await storage.delete(results[1].id)
        assert storage.size() == 2

        # Search again
        matches = await storage.search("Text")
        assert len(matches) == 2

        # Clear
        storage.clear()
        assert storage.size() == 0