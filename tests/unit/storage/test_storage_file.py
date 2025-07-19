"""Tests for file-based storage backend."""

import json
import shutil
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from sifaka.core.exceptions import StorageError
from sifaka.core.models import SifakaResult
from sifaka.storage.file import FileStorage


class TestFileStorage:
    """Test the FileStorage class."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def storage(self, temp_dir):
        """Create a FileStorage instance with temp directory."""
        return FileStorage(storage_dir=temp_dir)

    @pytest.fixture
    def sample_result(self):
        """Create a sample SifakaResult."""
        return SifakaResult(
            original_text="Original text for testing",
            final_text="Improved text after processing",
            iteration=2,
            was_improved=True,
        )

    def test_initialization(self, temp_dir):
        """Test storage initialization."""
        storage = FileStorage(storage_dir=temp_dir)
        assert storage.storage_dir == Path(temp_dir)
        assert storage.storage_dir.exists()

    def test_initialization_creates_directory(self):
        """Test that initialization creates directory if it doesn't exist."""
        non_existent = "/tmp/test_sifaka_storage_" + str(datetime.now().timestamp())
        storage = FileStorage(storage_dir=non_existent)
        assert storage.storage_dir.exists()
        # Clean up
        shutil.rmtree(non_existent)

    def test_get_file_path(self, storage):
        """Test file path generation."""
        path = storage._get_file_path("test-id-123")
        assert path == storage.storage_dir / "test-id-123.json"

    @pytest.mark.asyncio
    async def test_save_success(self, storage, sample_result):
        """Test successful save operation."""
        result_id = await storage.save(sample_result)

        assert result_id == sample_result.id

        # Verify file was created
        file_path = storage._get_file_path(result_id)
        assert file_path.exists()

        # Verify content
        with open(file_path) as f:
            data = json.load(f)
            assert data["id"] == sample_result.id
            assert data["original_text"] == sample_result.original_text
            assert data["final_text"] == sample_result.final_text

    @pytest.mark.asyncio
    async def test_save_io_error(self, storage, sample_result):
        """Test save with IO error."""
        # Make directory read-only to cause permission error
        storage.storage_dir.chmod(0o444)

        try:
            with pytest.raises(StorageError) as exc_info:
                await storage.save(sample_result)

            assert "Failed to save result" in str(exc_info.value)
            assert exc_info.value.storage_type == "file"
            assert exc_info.value.operation == "save"
        finally:
            # Restore permissions
            storage.storage_dir.chmod(0o755)

    @pytest.mark.asyncio
    async def test_load_existing(self, storage, sample_result):
        """Test loading an existing result."""
        # Save first
        await storage.save(sample_result)

        # Load
        loaded = await storage.load(sample_result.id)

        assert loaded is not None
        assert loaded.id == sample_result.id
        assert loaded.original_text == sample_result.original_text
        assert loaded.final_text == sample_result.final_text

    @pytest.mark.asyncio
    async def test_load_non_existent(self, storage):
        """Test loading a non-existent result."""
        result = await storage.load("non-existent-id")
        assert result is None

    @pytest.mark.asyncio
    async def test_load_corrupted_json(self, storage):
        """Test loading a file with corrupted JSON."""
        # Create a corrupted file
        file_path = storage._get_file_path("corrupted-id")
        with open(file_path, "w") as f:
            f.write("{ invalid json content")

        with pytest.raises(StorageError) as exc_info:
            await storage.load("corrupted-id")

        assert "Failed to parse result" in str(exc_info.value)
        assert exc_info.value.storage_type == "file"
        assert exc_info.value.operation == "load"

    @pytest.mark.asyncio
    async def test_load_invalid_model_data(self, storage):
        """Test loading a file with invalid model data."""
        # Create a file with invalid data structure
        file_path = storage._get_file_path("invalid-id")
        with open(file_path, "w") as f:
            json.dump({"id": "invalid-id", "missing": "required_fields"}, f)

        with pytest.raises(StorageError) as exc_info:
            await storage.load("invalid-id")

        assert "Failed to parse result" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_list_empty(self, storage):
        """Test listing with no stored results."""
        ids = await storage.list()
        assert ids == []

    @pytest.mark.asyncio
    async def test_list_with_results(self, storage):
        """Test listing with multiple results."""
        # Create multiple results
        results = []
        for i in range(5):
            result = SifakaResult(
                original_text=f"Original {i}", final_text=f"Final {i}"
            )
            results.append(result)
            await storage.save(result)

        # List all
        ids = await storage.list()
        assert len(ids) == 5
        assert all(r.id in ids for r in results)

    @pytest.mark.asyncio
    async def test_list_with_pagination(self, storage):
        """Test listing with limit and offset."""
        # Create 10 results
        for i in range(10):
            result = SifakaResult(
                original_text=f"Original {i}", final_text=f"Final {i}"
            )
            await storage.save(result)

        # Test limit
        ids = await storage.list(limit=3)
        assert len(ids) == 3

        # Test offset
        ids_offset = await storage.list(limit=3, offset=3)
        assert len(ids_offset) == 3
        assert set(ids).isdisjoint(set(ids_offset))  # No overlap

    @pytest.mark.asyncio
    async def test_list_sorted_by_modification_time(self, storage):
        """Test that list returns results sorted by modification time."""
        # Create results with delays
        result1 = SifakaResult(original_text="First", final_text="First")
        await storage.save(result1)

        # Small delay to ensure different modification times
        import asyncio

        await asyncio.sleep(0.01)

        result2 = SifakaResult(original_text="Second", final_text="Second")
        await storage.save(result2)

        ids = await storage.list()
        # Most recent should be first
        assert ids[0] == result2.id
        assert ids[1] == result1.id

    @pytest.mark.asyncio
    async def test_delete_existing(self, storage, sample_result):
        """Test deleting an existing result."""
        # Save first
        await storage.save(sample_result)
        file_path = storage._get_file_path(sample_result.id)
        assert file_path.exists()

        # Delete
        deleted = await storage.delete(sample_result.id)
        assert deleted is True
        assert not file_path.exists()

    @pytest.mark.asyncio
    async def test_delete_non_existent(self, storage):
        """Test deleting a non-existent result."""
        deleted = await storage.delete("non-existent-id")
        assert deleted is False

    @pytest.mark.asyncio
    async def test_search_in_original_text(self, storage):
        """Test searching in original text."""
        # Create results with specific content
        result1 = SifakaResult(
            original_text="Machine learning is fascinating", final_text="AI is amazing"
        )
        result2 = SifakaResult(
            original_text="Deep learning networks", final_text="Neural networks"
        )
        result3 = SifakaResult(original_text="Data science", final_text="Analytics")

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
            original_text="Original 1", final_text="Machine learning applications"
        )
        result2 = SifakaResult(
            original_text="Original 2", final_text="Deep learning models"
        )

        await storage.save(result1)
        await storage.save(result2)

        # Search in final text
        matches = await storage.search("learning")
        assert len(matches) == 2

    @pytest.mark.asyncio
    async def test_search_case_insensitive(self, storage):
        """Test case-insensitive search."""
        result = SifakaResult(original_text="MACHINE LEARNING", final_text="Final text")
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
                original_text=f"Machine learning example {i}", final_text="Final"
            )
            await storage.save(result)

        # Search with limit
        matches = await storage.search("machine", limit=5)
        assert len(matches) == 5

    @pytest.mark.asyncio
    async def test_search_handles_errors(self, storage, caplog):
        """Test that search continues despite errors."""
        # Create a valid result
        result = SifakaResult(
            original_text="Valid content with search term", final_text="Final"
        )
        await storage.save(result)

        # Create a corrupted file
        corrupted_path = storage._get_file_path("corrupted-id")
        with open(corrupted_path, "w") as f:
            f.write("invalid json")

        # Search should still find the valid result
        matches = await storage.search("search term")
        assert len(matches) == 1
        assert result.id in matches

        # Check that warning was logged
        assert "Error reading result file" in caplog.text

    def test_cleanup_old_files(self, storage):
        """Test cleanup of old files."""
        # Create files with different ages
        now = datetime.now()

        # Create old file (40 days old)
        old_path = storage._get_file_path("old-id")
        old_path.touch()
        old_time = (now - timedelta(days=40)).timestamp()
        import os

        os.utime(old_path, (old_time, old_time))

        # Create recent file (10 days old)
        recent_path = storage._get_file_path("recent-id")
        recent_path.touch()
        recent_time = (now - timedelta(days=10)).timestamp()
        os.utime(recent_path, (recent_time, recent_time))

        # Cleanup files older than 30 days
        deleted_count = storage.cleanup_old_files(days_old=30)

        assert deleted_count == 1
        assert not old_path.exists()
        assert recent_path.exists()

    def test_cleanup_old_files_empty_directory(self, storage):
        """Test cleanup with no files."""
        deleted_count = storage.cleanup_old_files(days_old=30)
        assert deleted_count == 0

    @pytest.mark.asyncio
    async def test_save_with_special_characters(self, storage):
        """Test saving with special characters in content."""
        result = SifakaResult(
            original_text='Text with "quotes" and \nnewlines\tand tabs',
            final_text="Unicode: 你好 🎉",
        )

        result_id = await storage.save(result)
        loaded = await storage.load(result_id)

        assert loaded.original_text == result.original_text
        assert loaded.final_text == result.final_text

    @pytest.mark.asyncio
    async def test_concurrent_operations(self, storage):
        """Test concurrent save/load operations."""
        import asyncio

        # Create multiple results
        results = [
            SifakaResult(original_text=f"Text {i}", final_text=f"Final {i}")
            for i in range(5)
        ]

        # Save concurrently
        save_tasks = [storage.save(r) for r in results]
        await asyncio.gather(*save_tasks)

        # Load concurrently
        load_tasks = [storage.load(r.id) for r in results]
        loaded_results = await asyncio.gather(*load_tasks)

        assert all(loaded is not None for loaded in loaded_results)
        assert len(loaded_results) == 5
