"""Tests for storage backends."""

import pytest
import tempfile
import shutil
from pathlib import Path

from sifaka.storage import MemoryStorage, FileStorage
from sifaka.core.models import SifakaResult
from sifaka.core.exceptions import StorageError


class TestMemoryStorage:
    """Test in-memory storage backend."""

    def test_initialization(self):
        """Test memory storage initialization."""
        storage = MemoryStorage()
        assert storage._storage == {}

    @pytest.mark.asyncio
    async def test_save_and_load(self):
        """Test saving and loading results."""
        storage = MemoryStorage()

        # Create test result
        result = SifakaResult(
            original_text="Test text", final_text="Improved test text"
        )

        # Save result
        saved_id = await storage.save(result)
        assert saved_id == result.id

        # Load result
        loaded_result = await storage.load(result.id)
        assert loaded_result is not None
        assert loaded_result.id == result.id
        assert loaded_result.original_text == result.original_text
        assert loaded_result.final_text == result.final_text

    @pytest.mark.asyncio
    async def test_load_nonexistent(self):
        """Test loading non-existent result."""
        storage = MemoryStorage()
        result = await storage.load("nonexistent-id")
        assert result is None

    @pytest.mark.asyncio
    async def test_list_results(self):
        """Test listing stored results."""
        storage = MemoryStorage()

        # Save multiple results
        results = []
        for i in range(3):
            result = SifakaResult(original_text=f"Test {i}", final_text=f"Improved {i}")
            await storage.save(result)
            results.append(result)

        # List all results
        result_ids = await storage.list()
        assert len(result_ids) == 3
        assert all(r.id in result_ids for r in results)

    @pytest.mark.asyncio
    async def test_list_with_pagination(self):
        """Test listing with pagination."""
        storage = MemoryStorage()

        # Save 5 results
        for i in range(5):
            result = SifakaResult(original_text=f"Test {i}", final_text=f"Improved {i}")
            await storage.save(result)

        # Test pagination
        first_page = await storage.list(limit=2, offset=0)
        assert len(first_page) == 2

        second_page = await storage.list(limit=2, offset=2)
        assert len(second_page) == 2

        # Ensure no overlap
        assert set(first_page).isdisjoint(set(second_page))

    @pytest.mark.asyncio
    async def test_delete_result(self):
        """Test deleting results."""
        storage = MemoryStorage()

        # Save result
        result = SifakaResult(original_text="Test text", final_text="Improved text")
        await storage.save(result)

        # Verify exists
        loaded = await storage.load(result.id)
        assert loaded is not None

        # Delete result
        deleted = await storage.delete(result.id)
        assert deleted is True

        # Verify deleted
        loaded = await storage.load(result.id)
        assert loaded is None

        # Try deleting again
        deleted_again = await storage.delete(result.id)
        assert deleted_again is False

    @pytest.mark.asyncio
    async def test_search_results(self):
        """Test searching results by content."""
        storage = MemoryStorage()

        # Save results with different content
        result1 = SifakaResult(
            original_text="Python programming tutorial",
            final_text="Comprehensive Python guide",
        )
        result2 = SifakaResult(
            original_text="JavaScript web development",
            final_text="Modern JS techniques",
        )
        result3 = SifakaResult(
            original_text="Machine learning with Python",
            final_text="ML algorithms in Python",
        )

        for result in [result1, result2, result3]:
            await storage.save(result)

        # Search for Python-related content
        python_results = await storage.search("python", limit=10)
        assert len(python_results) == 2
        assert result1.id in python_results
        assert result3.id in python_results
        assert result2.id not in python_results

        # Search with limit
        limited_results = await storage.search("python", limit=1)
        assert len(limited_results) == 1


class TestFileStorage:
    """Test file-based storage backend."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    def test_initialization(self, temp_dir):
        """Test file storage initialization."""
        storage = FileStorage(storage_dir=temp_dir)
        assert storage.storage_dir == Path(temp_dir)
        assert storage.storage_dir.exists()

    def test_initialization_creates_directory(self):
        """Test initialization creates non-existent directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage_path = Path(temp_dir) / "new_storage"
            storage = FileStorage(storage_dir=str(storage_path))
            assert storage.storage_dir.exists()

    @pytest.mark.asyncio
    async def test_save_and_load(self, temp_dir):
        """Test saving and loading results to/from files."""
        storage = FileStorage(storage_dir=temp_dir)

        # Create test result
        result = SifakaResult(
            original_text="Test text", final_text="Improved test text"
        )

        # Save result
        saved_id = await storage.save(result)
        assert saved_id == result.id

        # Verify file exists
        file_path = storage._get_file_path(result.id)
        assert file_path.exists()

        # Load result
        loaded_result = await storage.load(result.id)
        assert loaded_result is not None
        assert loaded_result.id == result.id
        assert loaded_result.original_text == result.original_text

    @pytest.mark.asyncio
    async def test_load_nonexistent(self, temp_dir):
        """Test loading non-existent file."""
        storage = FileStorage(storage_dir=temp_dir)
        result = await storage.load("nonexistent-id")
        assert result is None

    @pytest.mark.asyncio
    async def test_save_error_handling(self, temp_dir):
        """Test save error handling."""
        storage = FileStorage(storage_dir=temp_dir)

        # Make directory read-only to trigger permission error
        Path(temp_dir).chmod(0o444)

        result = SifakaResult(original_text="Test", final_text="Test")

        with pytest.raises(StorageError) as exc_info:
            await storage.save(result)

        assert exc_info.value.storage_type == "file"
        assert exc_info.value.operation == "save"

        # Restore permissions for cleanup
        Path(temp_dir).chmod(0o755)

    @pytest.mark.asyncio
    async def test_load_corrupted_file(self, temp_dir):
        """Test loading corrupted JSON file."""
        storage = FileStorage(storage_dir=temp_dir)

        # Create corrupted file
        corrupt_file = storage._get_file_path("corrupt-id")
        with open(corrupt_file, "w") as f:
            f.write("invalid json content")

        with pytest.raises(StorageError) as exc_info:
            await storage.load("corrupt-id")

        assert exc_info.value.storage_type == "file"
        assert exc_info.value.operation == "load"

    @pytest.mark.asyncio
    async def test_list_results(self, temp_dir):
        """Test listing stored files."""
        storage = FileStorage(storage_dir=temp_dir)

        # Save multiple results
        results = []
        for i in range(3):
            result = SifakaResult(original_text=f"Test {i}", final_text=f"Improved {i}")
            await storage.save(result)
            results.append(result)

        # List results
        result_ids = await storage.list()
        assert len(result_ids) == 3
        assert all(r.id in result_ids for r in results)

    @pytest.mark.asyncio
    async def test_delete_result(self, temp_dir):
        """Test deleting result files."""
        storage = FileStorage(storage_dir=temp_dir)

        # Save result
        result = SifakaResult(original_text="Test text", final_text="Improved text")
        await storage.save(result)

        # Verify file exists
        file_path = storage._get_file_path(result.id)
        assert file_path.exists()

        # Delete result
        deleted = await storage.delete(result.id)
        assert deleted is True
        assert not file_path.exists()

        # Try deleting again
        deleted_again = await storage.delete(result.id)
        assert deleted_again is False

    @pytest.mark.asyncio
    async def test_search_results(self, temp_dir):
        """Test searching file contents."""
        storage = FileStorage(storage_dir=temp_dir)

        # Save results with different content
        result1 = SifakaResult(
            original_text="Python programming tutorial",
            final_text="Comprehensive Python guide",
        )
        result2 = SifakaResult(
            original_text="JavaScript web development",
            final_text="Modern JS techniques",
        )

        await storage.save(result1)
        await storage.save(result2)

        # Search for Python content
        python_results = await storage.search("python")
        assert len(python_results) == 1
        assert result1.id in python_results

    def test_cleanup_old_files(self, temp_dir):
        """Test cleanup of old files."""
        storage = FileStorage(storage_dir=temp_dir)

        # Create old file
        old_file = storage._get_file_path("old-result")
        old_file.write_text('{"test": "data"}')

        # Set old timestamp (30+ days ago)
        import os

        old_time = 1640995200  # Jan 1, 2022
        os.utime(old_file, (old_time, old_time))

        # Cleanup files older than 1 day
        deleted_count = storage.cleanup_old_files(days_old=1)
        assert deleted_count == 1
        assert not old_file.exists()


class TestStorageInterface:
    """Test storage backend interface compliance."""

    @pytest.mark.asyncio
    async def test_memory_storage_interface(self):
        """Test MemoryStorage implements interface correctly."""
        storage = MemoryStorage()

        # Test all required methods exist and are async
        assert hasattr(storage, "save")
        assert hasattr(storage, "load")
        assert hasattr(storage, "list")
        assert hasattr(storage, "delete")
        assert hasattr(storage, "search")

        # Test they're actually async
        import inspect

        assert inspect.iscoroutinefunction(storage.save)
        assert inspect.iscoroutinefunction(storage.load)
        assert inspect.iscoroutinefunction(storage.list)
        assert inspect.iscoroutinefunction(storage.delete)
        assert inspect.iscoroutinefunction(storage.search)

    @pytest.mark.asyncio
    async def test_file_storage_interface(self):
        """Test FileStorage implements interface correctly."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = FileStorage(storage_dir=temp_dir)

            # Test all required methods exist and are async
            assert hasattr(storage, "save")
            assert hasattr(storage, "load")
            assert hasattr(storage, "list")
            assert hasattr(storage, "delete")
            assert hasattr(storage, "search")

            # Test they're actually async
            import inspect

            assert inspect.iscoroutinefunction(storage.save)
            assert inspect.iscoroutinefunction(storage.load)
            assert inspect.iscoroutinefunction(storage.list)
            assert inspect.iscoroutinefunction(storage.delete)
            assert inspect.iscoroutinefunction(storage.search)
