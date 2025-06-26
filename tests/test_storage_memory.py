"""Tests for in-memory storage backend."""

import pytest
from sifaka.storage.memory import MemoryStorage
from sifaka.core.models import SifakaResult
from sifaka.core.exceptions import StorageError
from unittest.mock import patch


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
        )

    @pytest.fixture
    def multiple_results(self):
        """Create multiple SifakaResults for testing."""
        results = []
        for i in range(5):
            results.append(
                SifakaResult(
                    original_text=f"Original text {i}", final_text=f"Final text {i}"
                )
            )
        return results

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
        # Save result
        await storage.save(sample_result)
        assert storage.size() == 1

        # Delete result
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
    async def test_list_results(self, storage, multiple_results):
        """Test listing stored results."""
        # Save multiple results
        saved_ids = []
        for result in multiple_results:
            result_id = await storage.save(result)
            saved_ids.append(result_id)

        # List all
        all_ids = await storage.list()
        assert len(all_ids) == 5
        assert set(all_ids) == set(saved_ids)

    @pytest.mark.asyncio
    async def test_list_with_pagination(self, storage, multiple_results):
        """Test listing with limit and offset."""
        # Save multiple results
        for result in multiple_results:
            await storage.save(result)

        # Test limit
        limited = await storage.list(limit=3)
        assert len(limited) == 3

        # Test offset
        offset_results = await storage.list(limit=2, offset=2)
        assert len(offset_results) == 2

        # Test offset beyond results
        beyond = await storage.list(limit=10, offset=10)
        assert len(beyond) == 0

    @pytest.mark.asyncio
    async def test_search_in_original_text(self, storage, multiple_results):
        """Test searching in original text."""
        # Save results
        for result in multiple_results:
            await storage.save(result)

        # Search for "text 2"
        matches = await storage.search("text 2")
        assert len(matches) == 1
        assert matches[0] == multiple_results[2].id

    @pytest.mark.asyncio
    async def test_search_in_final_text(self, storage):
        """Test searching in final text."""
        # Create results with specific final text
        result1 = SifakaResult(
            original_text="Original", final_text="The cat sat on the mat"
        )
        result2 = SifakaResult(
            original_text="Original", final_text="The dog ran in the park"
        )

        await storage.save(result1)
        await storage.save(result2)

        # Search for "cat"
        matches = await storage.search("cat")
        assert len(matches) == 1
        assert matches[0] == result1.id

    @pytest.mark.asyncio
    async def test_search_case_insensitive(self, storage):
        """Test case-insensitive search."""
        result = SifakaResult(
            original_text="Original TEXT with CAPS", final_text="Final text"
        )
        await storage.save(result)

        # Search with different cases
        matches1 = await storage.search("TEXT")
        matches2 = await storage.search("text")
        matches3 = await storage.search("TExT")

        assert len(matches1) == 1
        assert len(matches2) == 1
        assert len(matches3) == 1
        assert matches1[0] == matches2[0] == matches3[0] == result.id

    @pytest.mark.asyncio
    async def test_search_with_limit(self, storage):
        """Test search with result limit."""
        # Create multiple matching results
        for i in range(5):
            result = SifakaResult(
                original_text=f"Document {i} contains keyword", final_text=f"Final {i}"
            )
            await storage.save(result)

        # Search with limit
        matches = await storage.search("keyword", limit=3)
        assert len(matches) == 3

    @pytest.mark.asyncio
    async def test_search_no_matches(self, storage, sample_result):
        """Test search with no matches."""
        await storage.save(sample_result)

        matches = await storage.search("nonexistent")
        assert len(matches) == 0

    def test_clear(self, storage):
        """Test clearing all stored results."""
        # Add some results synchronously
        storage._storage["id1"] = SifakaResult(original_text="1", final_text="1")
        storage._storage["id2"] = SifakaResult(original_text="2", final_text="2")
        assert storage.size() == 2

        # Clear
        storage.clear()
        assert storage.size() == 0
        assert storage._storage == {}

    def test_size(self, storage):
        """Test getting storage size."""
        assert storage.size() == 0

        # Add results directly
        storage._storage["id1"] = SifakaResult(original_text="1", final_text="1")
        assert storage.size() == 1

        storage._storage["id2"] = SifakaResult(original_text="2", final_text="2")
        assert storage.size() == 2

    @pytest.mark.asyncio
    async def test_save_exception_handling(self, storage):
        """Test exception handling in save operation."""
        result = SifakaResult(original_text="Test", final_text="Test")

        # Mock the storage to raise an exception
        with patch.object(
            storage._storage, "__setitem__", side_effect=Exception("Save failed")
        ):
            with pytest.raises(StorageError) as exc_info:
                await storage.save(result)

            assert "Failed to save result" in str(exc_info.value)
            assert exc_info.value.storage_type == "memory"
            assert exc_info.value.operation == "save"

    @pytest.mark.asyncio
    async def test_load_exception_handling(self, storage):
        """Test exception handling in load operation."""
        # Mock the storage to raise an exception
        with patch.object(
            storage._storage, "get", side_effect=Exception("Load failed")
        ):
            with pytest.raises(StorageError) as exc_info:
                await storage.load("test-id")

            assert "Failed to load result" in str(exc_info.value)
            assert exc_info.value.storage_type == "memory"
            assert exc_info.value.operation == "load"

    @pytest.mark.asyncio
    async def test_search_exception_handling(self, storage):
        """Test exception handling in search operation."""
        # Add a result
        result = SifakaResult(original_text="Test", final_text="Test")
        storage._storage[result.id] = result

        # Mock to raise an exception during iteration
        with patch.object(
            storage._storage, "items", side_effect=Exception("Search failed")
        ):
            with pytest.raises(StorageError) as exc_info:
                await storage.search("test")

            assert "Search failed for query" in str(exc_info.value)
            assert exc_info.value.storage_type == "memory"
            assert exc_info.value.operation == "search"

    @pytest.mark.asyncio
    async def test_concurrent_operations(self, storage):
        """Test concurrent save and load operations."""
        import asyncio

        # Create multiple results
        results = [
            SifakaResult(original_text=f"Text {i}", final_text=f"Final {i}")
            for i in range(10)
        ]

        # Save concurrently
        save_tasks = [storage.save(result) for result in results]
        saved_ids = await asyncio.gather(*save_tasks)

        assert len(saved_ids) == 10
        assert storage.size() == 10

        # Load concurrently
        load_tasks = [storage.load(result_id) for result_id in saved_ids]
        loaded_results = await asyncio.gather(*load_tasks)

        assert all(result is not None for result in loaded_results)
        assert len(loaded_results) == 10
