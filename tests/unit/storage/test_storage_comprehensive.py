"""Comprehensive tests for all storage backend implementations.

This test suite covers all storage backends with extensive testing of:
- Basic CRUD operations
- Persistence and durability
- Error handling and recovery
- Concurrent access patterns
- Performance characteristics
- Edge cases and boundary conditions
"""

import asyncio
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import List
from unittest.mock import patch

import pytest

from sifaka.core.exceptions import StorageError
from sifaka.core.models import (
    CritiqueResult,
    Generation,
    SifakaResult,
    ValidationResult,
)
from sifaka.storage import (
    FileStorage,
    MemoryStorage,
    StorageBackend,
)


@pytest.fixture
def sample_result():
    """Create a sample SifakaResult for testing."""
    return SifakaResult(
        original_text="Original test text",
        final_text="Improved test text",
        iteration=2,
        generations=[
            Generation(
                iteration=1,
                text="First generation",
                model="gpt-4o-mini",
                prompt_tokens=50,
                completion_tokens=100,
                total_tokens=150,
            ),
            Generation(
                iteration=1,
                text="Second generation",
                model="gpt-4o-mini",
                prompt_tokens=60,
                completion_tokens=120,
                total_tokens=180,
            ),
        ],
        critiques=[
            CritiqueResult(
                critic="reflexion",
                feedback="Good improvement needed",
                suggestions=["Add examples", "Improve clarity"],
                needs_improvement=True,
                confidence=0.8,
            )
        ],
        validations=[
            ValidationResult(
                validator="length",
                passed=True,
                score=0.9,
                details="Length validation passed",
            )
        ],
        processing_time=2.5,
    )


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


class TestMemoryStorage:
    """Comprehensive tests for MemoryStorage backend."""

    def test_memory_storage_initialization(self):
        """Test MemoryStorage initialization."""
        storage = MemoryStorage()
        assert isinstance(storage, StorageBackend)
        assert len(storage._data) == 0

    @pytest.mark.asyncio
    async def test_save_and_load_basic(self, sample_result):
        """Test basic save and load operations."""
        storage = MemoryStorage()

        # Save result
        result_id = await storage.save(sample_result)
        assert result_id == sample_result.id

        # Load result
        loaded_result = await storage.load(result_id)
        assert loaded_result is not None
        assert loaded_result.id == sample_result.id
        assert loaded_result.original_text == sample_result.original_text
        assert loaded_result.final_text == sample_result.final_text
        assert loaded_result.iteration == sample_result.iteration

    @pytest.mark.asyncio
    async def test_load_nonexistent(self):
        """Test loading non-existent result."""
        storage = MemoryStorage()

        result = await storage.load("nonexistent-id")
        assert result is None

    @pytest.mark.asyncio
    async def test_delete_basic(self, sample_result):
        """Test basic delete operation."""
        storage = MemoryStorage()

        # Save and then delete
        result_id = await storage.save(sample_result)
        success = await storage.delete(result_id)
        assert success is True

        # Verify deletion
        loaded_result = await storage.load(result_id)
        assert loaded_result is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent(self):
        """Test deleting non-existent result."""
        storage = MemoryStorage()

        success = await storage.delete("nonexistent-id")
        assert success is False

    @pytest.mark.asyncio
    async def test_list_empty(self):
        """Test listing when storage is empty."""
        storage = MemoryStorage()

        result_ids = await storage.list()
        assert result_ids == []

    @pytest.mark.asyncio
    async def test_list_multiple_results(self, sample_result):
        """Test listing multiple results."""
        storage = MemoryStorage()

        # Save multiple results
        results = []
        for i in range(3):
            result = SifakaResult(
                original_text=f"Original {i}",
                final_text=f"Final {i}",
                iteration=1,
                generations=[],
                critiques=[],
                validations=[],
                processing_time=1.0,
            )
            results.append(result)
            await storage.save(result)

        # List all results
        result_ids = await storage.list()
        assert len(result_ids) == 3
        assert all(result.id in result_ids for result in results)

    @pytest.mark.asyncio
    async def test_search_basic(self, sample_result):
        """Test basic search functionality."""
        storage = MemoryStorage()
        await storage.save(sample_result)

        # Search for text that exists
        results = await storage.search("test text")
        assert len(results) == 1
        assert results[0] == sample_result.id

        # Search for text that doesn't exist
        results = await storage.search("nonexistent")
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_search_case_insensitive(self, sample_result):
        """Test case-insensitive search."""
        storage = MemoryStorage()
        await storage.save(sample_result)

        # Search with different cases
        results1 = await storage.search("TEST TEXT")
        results2 = await storage.search("test text")
        results3 = await storage.search("Test Text")

        assert len(results1) == len(results2) == len(results3) == 1
        assert results1[0] == results2[0] == results3[0] == sample_result.id

    @pytest.mark.asyncio
    async def test_search_multiple_matches(self):
        """Test search with multiple matching results."""
        storage = MemoryStorage()

        # Create multiple results with common text
        results = []
        for i in range(3):
            result = SifakaResult(
                original_text=f"Common keyword in result {i}",
                final_text=f"Final text {i}",
                iteration=1,
                generations=[],
                critiques=[],
                validations=[],
                processing_time=1.0,
            )
            results.append(result)
            await storage.save(result)

        # Search for common keyword
        search_results = await storage.search("keyword")
        assert len(search_results) == 3
        assert all(result.id in search_results for result in results)

    @pytest.mark.asyncio
    async def test_clear_storage(self, sample_result):
        """Test clearing all storage."""
        storage = MemoryStorage()

        # Add some data
        await storage.save(sample_result)
        assert len(await storage.list()) == 1

        # Clear storage
        storage.clear()
        assert len(await storage.list()) == 0
        assert len(storage._data) == 0

    def test_size_tracking(self, sample_result):
        """Test storage size tracking."""
        storage = MemoryStorage()

        # Initially empty
        assert storage.size() == 0

        # Add result
        asyncio.run(storage.save(sample_result))
        assert storage.size() == 1

        # Add another result
        result2 = SifakaResult(
            original_text="Another test",
            final_text="Another final",
            iteration=1,
            generations=[],
            critiques=[],
            validations=[],
            processing_time=1.0,
        )
        asyncio.run(storage.save(result2))
        assert storage.size() == 2

        # Delete one
        asyncio.run(storage.delete(sample_result.id))
        assert storage.size() == 1

        # Clear all
        storage.clear()
        assert storage.size() == 0

    @pytest.mark.asyncio
    async def test_concurrent_access(self):
        """Test concurrent access to memory storage."""
        storage = MemoryStorage()

        async def save_result(i):
            result = SifakaResult(
                original_text=f"Concurrent test {i}",
                final_text=f"Final {i}",
                iteration=1,
                generations=[],
                critiques=[],
                validations=[],
                processing_time=1.0,
            )
            return await storage.save(result)

        # Save multiple results concurrently
        tasks = [save_result(i) for i in range(10)]
        result_ids = await asyncio.gather(*tasks)

        assert len(result_ids) == 10
        assert len(set(result_ids)) == 10  # All unique
        assert storage.size() == 10

    @pytest.mark.asyncio
    async def test_memory_storage_persistence(self, sample_result):
        """Test that MemoryStorage is not persistent across instances."""
        storage1 = MemoryStorage()
        await storage1.save(sample_result)

        # Create new instance
        storage2 = MemoryStorage()
        result = await storage2.load(sample_result.id)
        assert result is None  # Should not persist


class TestFileStorage:
    """Comprehensive tests for FileStorage backend."""

    def test_file_storage_initialization(self, temp_dir):
        """Test FileStorage initialization."""
        storage = FileStorage(temp_dir)
        assert isinstance(storage, StorageBackend)
        assert storage.base_path == Path(temp_dir)
        assert storage.base_path.exists()

    def test_file_storage_default_path(self):
        """Test FileStorage with default path."""
        storage = FileStorage()
        assert storage.base_path.exists()
        assert storage.base_path.name == "sifaka_results"

    @pytest.mark.asyncio
    async def test_save_and_load_basic(self, sample_result, temp_dir):
        """Test basic save and load operations."""
        storage = FileStorage(temp_dir)

        # Save result
        result_id = await storage.save(sample_result)
        assert result_id == sample_result.id

        # Verify file exists
        file_path = storage.base_path / f"{result_id}.json"
        assert file_path.exists()

        # Load result
        loaded_result = await storage.load(result_id)
        assert loaded_result is not None
        assert loaded_result.id == sample_result.id
        assert loaded_result.original_text == sample_result.original_text

    @pytest.mark.asyncio
    async def test_file_persistence(self, sample_result, temp_dir):
        """Test that FileStorage persists across instances."""
        # Save with first instance
        storage1 = FileStorage(temp_dir)
        result_id = await storage1.save(sample_result)

        # Load with second instance
        storage2 = FileStorage(temp_dir)
        loaded_result = await storage2.load(result_id)
        assert loaded_result is not None
        assert loaded_result.id == sample_result.id

    @pytest.mark.asyncio
    async def test_delete_file(self, sample_result, temp_dir):
        """Test file deletion."""
        storage = FileStorage(temp_dir)

        # Save and verify file exists
        result_id = await storage.save(sample_result)
        file_path = storage.base_path / f"{result_id}.json"
        assert file_path.exists()

        # Delete and verify file is gone
        success = await storage.delete(result_id)
        assert success is True
        assert not file_path.exists()

    @pytest.mark.asyncio
    async def test_list_files(self, temp_dir):
        """Test listing files."""
        storage = FileStorage(temp_dir)

        # Save multiple results
        results = []
        for i in range(3):
            result = SifakaResult(
                original_text=f"Test {i}",
                final_text=f"Final {i}",
                iteration=1,
                generations=[],
                critiques=[],
                validations=[],
                processing_time=1.0,
            )
            results.append(result)
            await storage.save(result)

        # List all results
        result_ids = await storage.list()
        assert len(result_ids) == 3
        assert all(result.id in result_ids for result in results)

    @pytest.mark.asyncio
    async def test_search_files(self, temp_dir):
        """Test searching through files."""
        storage = FileStorage(temp_dir)

        # Create results with searchable content
        result1 = SifakaResult(
            original_text="Python programming tutorial",
            final_text="Advanced Python guide",
            iteration=1,
            generations=[],
            critiques=[],
            validations=[],
            processing_time=1.0,
        )
        result2 = SifakaResult(
            original_text="JavaScript web development",
            final_text="Modern JS techniques",
            iteration=1,
            generations=[],
            critiques=[],
            validations=[],
            processing_time=1.0,
        )

        await storage.save(result1)
        await storage.save(result2)

        # Search for Python
        python_results = await storage.search("Python")
        assert len(python_results) == 1
        assert result1.id in python_results

        # Search for programming (should match both)
        prog_results = await storage.search("programming")
        assert len(prog_results) == 1  # Only result1 has "programming"

    @pytest.mark.asyncio
    async def test_file_corruption_handling(self, temp_dir):
        """Test handling of corrupted files."""
        storage = FileStorage(temp_dir)

        # Create a corrupted file
        corrupted_file = storage.base_path / "corrupted.json"
        with open(corrupted_file, "w") as f:
            f.write("invalid json content {")

        # Try to load corrupted file
        result = await storage.load("corrupted")
        assert result is None

    @pytest.mark.asyncio
    async def test_concurrent_file_operations(self, temp_dir):
        """Test concurrent file operations."""
        storage = FileStorage(temp_dir)

        async def save_and_load(i):
            result = SifakaResult(
                original_text=f"Concurrent test {i}",
                final_text=f"Final {i}",
                iteration=1,
                generations=[],
                critiques=[],
                validations=[],
                processing_time=1.0,
            )
            result_id = await storage.save(result)
            loaded = await storage.load(result_id)
            return loaded is not None

        # Run concurrent operations
        tasks = [save_and_load(i) for i in range(5)]
        results = await asyncio.gather(*tasks)

        assert all(results)  # All operations should succeed
        assert len(await storage.list()) == 5

    @pytest.mark.asyncio
    async def test_large_result_handling(self, temp_dir):
        """Test handling of large results."""
        storage = FileStorage(temp_dir)

        # Create a large result
        large_generations = []
        for i in range(100):
            large_generations.append(
                Generation(
                    iteration=1,
                    text=f"Large generation {i} " + "x" * 1000,
                    model="gpt-4o-mini",
                    prompt_tokens=100,
                    completion_tokens=200,
                    total_tokens=300,
                )
            )

        large_result = SifakaResult(
            original_text="Large test " + "y" * 10000,
            final_text="Large final " + "z" * 10000,
            iteration=1,
            generations=large_generations,
            critiques=[],
            validations=[],
            processing_time=1.0,
        )

        # Save and load large result
        result_id = await storage.save(large_result)
        loaded_result = await storage.load(result_id)

        assert loaded_result is not None
        assert len(loaded_result.generations) == 100
        assert len(loaded_result.original_text) > 10000

    @pytest.mark.asyncio
    async def test_directory_permissions(self, temp_dir):
        """Test handling of directory permission issues."""
        storage = FileStorage(temp_dir)

        # Make directory read-only (on Unix systems)
        if os.name != "nt":  # Skip on Windows
            os.chmod(temp_dir, 0o444)

            result = SifakaResult(
                original_text="Permission test",
                final_text="Final",
                iteration=1,
                generations=[],
                critiques=[],
                validations=[],
                processing_time=1.0,
            )

            # Should handle permission error gracefully
            with pytest.raises(StorageError):
                await storage.save(result)

            # Restore permissions for cleanup
            os.chmod(temp_dir, 0o755)


class TestStorageBackendInterface:
    """Test the StorageBackend abstract interface."""

    def test_storage_backend_is_abstract(self):
        """Test that StorageBackend cannot be instantiated directly."""
        with pytest.raises(TypeError):
            StorageBackend()

    def test_custom_storage_implementation(self):
        """Test implementing a custom storage backend."""

        class CustomStorage(StorageBackend):
            def __init__(self):
                self._data = {}

            async def save(self, result: SifakaResult) -> str:
                self._data[result.id] = result
                return result.id

            async def load(self, result_id: str) -> SifakaResult:
                return self._data.get(result_id)

            async def delete(self, result_id: str) -> bool:
                if result_id in self._data:
                    del self._data[result_id]
                    return True
                return False

            async def list(self) -> List[str]:
                return list(self._data.keys())

            async def search(self, query: str) -> List[str]:
                results = []
                for result_id, result in self._data.items():
                    if query.lower() in result.original_text.lower():
                        results.append(result_id)
                return results

        # Test custom implementation
        storage = CustomStorage()
        assert isinstance(storage, StorageBackend)


class TestStorageErrorHandling:
    """Test error handling across storage backends."""

    @pytest.mark.asyncio
    async def test_memory_storage_error_recovery(self):
        """Test MemoryStorage error recovery."""
        storage = MemoryStorage()

        # Simulate internal data corruption
        storage._data = None

        result = SifakaResult(
            original_text="Error test",
            final_text="Final",
            iteration=1,
            generations=[],
            critiques=[],
            validations=[],
            processing_time=1.0,
        )

        # Should handle gracefully
        with pytest.raises(AttributeError):
            await storage.save(result)

    @pytest.mark.asyncio
    async def test_file_storage_disk_full_simulation(self, temp_dir):
        """Test FileStorage behavior when disk is full."""
        storage = FileStorage(temp_dir)

        result = SifakaResult(
            original_text="Disk full test",
            final_text="Final",
            iteration=1,
            generations=[],
            critiques=[],
            validations=[],
            processing_time=1.0,
        )

        # Mock file write to raise OSError (disk full)
        with patch("builtins.open", side_effect=OSError("No space left on device")):
            with pytest.raises(StorageError):
                await storage.save(result)

    @pytest.mark.asyncio
    async def test_invalid_result_data(self, temp_dir):
        """Test handling of invalid result data."""
        storage = FileStorage(temp_dir)

        # Try to save None
        with pytest.raises((AttributeError, TypeError)):
            await storage.save(None)


class TestStoragePerformance:
    """Performance tests for storage backends."""

    @pytest.mark.asyncio
    async def test_memory_storage_performance(self):
        """Test MemoryStorage performance with many operations."""
        storage = MemoryStorage()

        # Create many results
        results = []
        for i in range(100):
            result = SifakaResult(
                original_text=f"Performance test {i}",
                final_text=f"Final {i}",
                iteration=1,
                generations=[],
                critiques=[],
                validations=[],
                processing_time=1.0,
            )
            results.append(result)

        # Time save operations
        start_time = time.time()
        for result in results:
            await storage.save(result)
        save_time = time.time() - start_time

        # Time load operations
        start_time = time.time()
        for result in results:
            await storage.load(result.id)
        load_time = time.time() - start_time

        # Should be reasonably fast
        assert save_time < 1.0  # Less than 1 second for 100 saves
        assert load_time < 1.0  # Less than 1 second for 100 loads

    @pytest.mark.asyncio
    async def test_file_storage_performance(self, temp_dir):
        """Test FileStorage performance with many operations."""
        storage = FileStorage(temp_dir)

        # Create fewer results for file storage (slower)
        results = []
        for i in range(20):
            result = SifakaResult(
                original_text=f"File performance test {i}",
                final_text=f"Final {i}",
                iteration=1,
                generations=[],
                critiques=[],
                validations=[],
                processing_time=1.0,
            )
            results.append(result)

        # Time save operations
        start_time = time.time()
        for result in results:
            await storage.save(result)
        save_time = time.time() - start_time

        # Should complete in reasonable time
        assert save_time < 5.0  # Less than 5 seconds for 20 saves
