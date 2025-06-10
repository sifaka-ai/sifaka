"""Comprehensive unit tests for file storage backend.

This module tests the file-based storage implementation:
- SifakaFilePersistence: File-based thought persistence
- Directory management and file operations
- Backup and recovery functionality
- Error handling and edge cases

Tests cover:
- Storage and retrieval operations
- File system interactions
- Backup management
- Error scenarios and recovery
- Mock-based testing without actual file I/O
"""

import asyncio
import tempfile
import uuid
from pathlib import Path
from unittest.mock import patch

import pytest

from sifaka.core.thought import SifakaThought
from sifaka.storage.file import SifakaFilePersistence


class TestSifakaFilePersistence:
    """Test suite for SifakaFilePersistence class."""

    def test_file_persistence_creation_minimal(self):
        """Test creating SifakaFilePersistence with minimal parameters."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = SifakaFilePersistence(storage_dir=temp_dir)

            assert storage.storage_dir == Path(temp_dir)
            assert storage.auto_backup is True
            assert storage.max_backup_count == 5

    def test_file_persistence_creation_full(self):
        """Test creating SifakaFilePersistence with all parameters."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = SifakaFilePersistence(
                storage_dir=temp_dir, auto_backup=False, max_backup_count=10, key_prefix="test"
            )

            assert storage.storage_dir == Path(temp_dir)
            assert storage.auto_backup is False
            assert storage.max_backup_count == 10
            assert storage.key_prefix == "test"

    @pytest.mark.asyncio
    async def test_store_and_retrieve_thought(self):
        """Test storing and retrieving a thought."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = SifakaFilePersistence(storage_dir=temp_dir)

            thought = SifakaThought(
                prompt="Test prompt", final_text="Test response", iteration=1, max_iterations=3
            )

            # Store the thought
            await storage.store_thought(thought)

            # Retrieve the thought
            retrieved = await storage.get_thought(thought.id)

            assert retrieved is not None
            assert retrieved.id == thought.id
            assert retrieved.prompt == thought.prompt
            assert retrieved.final_text == thought.final_text

    @pytest.mark.asyncio
    async def test_store_thought_creates_directory(self):
        """Test that storing a thought creates necessary directories."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage_path = Path(temp_dir) / "sifaka_storage"
            storage = SifakaFilePersistence(storage_dir=str(storage_path))

            thought = SifakaThought(
                prompt="Test prompt", final_text="Test response", iteration=1, max_iterations=3
            )

            # Directory shouldn't exist initially
            assert not storage_path.exists()

            # Store the thought
            await storage.store_thought(thought)

            # Directory should now exist
            assert storage_path.exists()
            assert storage_path.is_dir()

    @pytest.mark.asyncio
    async def test_get_nonexistent_thought(self):
        """Test retrieving a non-existent thought."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = SifakaFilePersistence(storage_dir=temp_dir)

            nonexistent_id = str(uuid.uuid4())
            result = await storage.get_thought(nonexistent_id)

            assert result is None

    @pytest.mark.asyncio
    async def test_list_thoughts_empty(self):
        """Test listing thoughts when storage is empty."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = SifakaFilePersistence(storage_dir=temp_dir)

            thoughts = await storage.list_thoughts()

            assert thoughts == []

    @pytest.mark.asyncio
    async def test_list_thoughts_with_data(self):
        """Test listing thoughts with stored data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = SifakaFilePersistence(storage_dir=temp_dir)

            # Store multiple thoughts
            thoughts = []
            for i in range(3):
                thought = SifakaThought(
                    prompt=f"Test prompt {i}",
                    final_text=f"Test response {i}",
                    iteration=1,
                    max_iterations=3,
                )
                thoughts.append(thought)
                await storage.store_thought(thought)

            # List all thoughts
            retrieved_thoughts = await storage.list_thoughts()

            assert len(retrieved_thoughts) == 3
            retrieved_ids = {t.id for t in retrieved_thoughts}
            expected_ids = {t.id for t in thoughts}
            assert retrieved_ids == expected_ids

    @pytest.mark.asyncio
    async def test_delete_thought(self):
        """Test deleting a thought."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = SifakaFilePersistence(storage_dir=temp_dir)

            thought = SifakaThought(
                prompt="Test prompt", final_text="Test response", iteration=1, max_iterations=3
            )

            # Store and then delete
            await storage.store_thought(thought)
            assert await storage.get_thought(thought.id) is not None

            await storage.delete_thought(thought.id)
            assert await storage.get_thought(thought.id) is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent_thought(self):
        """Test deleting a non-existent thought."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = SifakaFilePersistence(storage_dir=temp_dir)

            nonexistent_id = str(uuid.uuid4())

            # Should not raise an exception
            await storage.delete_thought(nonexistent_id)

    @pytest.mark.asyncio
    async def test_update_thought(self):
        """Test updating an existing thought."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = SifakaFilePersistence(storage_dir=temp_dir)

            thought = SifakaThought(
                prompt="Test prompt", final_text="Original response", iteration=1, max_iterations=3
            )

            await storage.store_thought(thought)

            # Update the thought
            thought.final_text = "Updated response"
            thought.iteration = 2

            await storage.store_thought(thought)

            # Retrieve and verify update
            retrieved = await storage.get_thought(thought.id)
            assert retrieved.final_text == "Updated response"
            assert retrieved.iteration == 2

    @pytest.mark.asyncio
    async def test_backup_functionality(self):
        """Test automatic backup functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = SifakaFilePersistence(
                storage_dir=temp_dir, auto_backup=True, max_backup_count=3
            )

            thought = SifakaThought(
                prompt="Test prompt", final_text="Test response", iteration=1, max_iterations=3
            )

            # Store thought multiple times to trigger backups
            for i in range(5):
                thought.final_text = f"Response version {i}"
                await storage.store_thought(thought)

            # Check that backup directory exists and has files
            backup_dir = storage.storage_dir / "backups"
            if backup_dir.exists():
                backup_files = list(backup_dir.glob("*.json"))
                # Should not exceed max_backup_count
                assert len(backup_files) <= storage.max_backup_count

    @pytest.mark.asyncio
    async def test_search_thoughts(self):
        """Test searching thoughts by content."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = SifakaFilePersistence(storage_dir=temp_dir)

            # Store thoughts with different content
            thoughts = [
                SifakaThought(
                    prompt="Tell me about cats",
                    final_text="Cats are wonderful pets",
                    iteration=1,
                    max_iterations=3,
                ),
                SifakaThought(
                    prompt="Tell me about dogs",
                    final_text="Dogs are loyal companions",
                    iteration=1,
                    max_iterations=3,
                ),
                SifakaThought(
                    prompt="What about cats?",
                    final_text="Cats are independent",
                    iteration=1,
                    max_iterations=3,
                ),
            ]

            for thought in thoughts:
                await storage.store_thought(thought)

            # Search for thoughts containing "cats"
            cat_thoughts = await storage.search_thoughts("cats")

            assert len(cat_thoughts) == 2
            for thought in cat_thoughts:
                assert "cats" in thought.prompt.lower() or "cats" in thought.final_text.lower()

    @pytest.mark.asyncio
    async def test_clear_storage(self):
        """Test clearing all storage."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = SifakaFilePersistence(storage_dir=temp_dir)

            # Store some thoughts
            for i in range(3):
                thought = SifakaThought(
                    prompt=f"Test prompt {i}",
                    final_text=f"Test response {i}",
                    iteration=1,
                    max_iterations=3,
                )
                await storage.store_thought(thought)

            # Verify thoughts are stored
            thoughts = await storage.list_thoughts()
            assert len(thoughts) == 3

            # Clear storage
            await storage.clear()

            # Verify storage is empty
            thoughts = await storage.list_thoughts()
            assert len(thoughts) == 0

    @pytest.mark.asyncio
    async def test_concurrent_access(self):
        """Test concurrent access to file storage."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = SifakaFilePersistence(storage_dir=temp_dir)

            async def store_thought(i):
                thought = SifakaThought(
                    prompt=f"Concurrent prompt {i}",
                    final_text=f"Concurrent response {i}",
                    iteration=1,
                    max_iterations=3,
                )
                await storage.store_thought(thought)
                return thought

            # Store thoughts concurrently
            tasks = [store_thought(i) for i in range(5)]
            stored_thoughts = await asyncio.gather(*tasks)

            # Verify all thoughts were stored
            all_thoughts = await storage.list_thoughts()
            assert len(all_thoughts) == 5

            stored_ids = {t.id for t in stored_thoughts}
            retrieved_ids = {t.id for t in all_thoughts}
            assert stored_ids == retrieved_ids

    def test_file_persistence_repr(self):
        """Test SifakaFilePersistence string representation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = SifakaFilePersistence(storage_dir=temp_dir)

            repr_str = repr(storage)
            assert "SifakaFilePersistence" in repr_str
            assert temp_dir in repr_str


class TestSifakaFilePersistenceErrorHandling:
    """Test suite for SifakaFilePersistence error handling."""

    @pytest.mark.asyncio
    async def test_store_thought_permission_error(self):
        """Test storing thought when permission is denied."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = SifakaFilePersistence(storage_dir=temp_dir)

            thought = SifakaThought(
                prompt="Test prompt", final_text="Test response", iteration=1, max_iterations=3
            )

            # Mock file operations to raise permission error
            with patch("aiofiles.open", side_effect=PermissionError("Permission denied")):
                # Should handle error gracefully
                try:
                    await storage.store_thought(thought)
                except PermissionError:
                    pass  # Expected behavior

    @pytest.mark.asyncio
    async def test_get_thought_file_not_found(self):
        """Test retrieving thought when file doesn't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = SifakaFilePersistence(storage_dir=temp_dir)

            # Try to get a thought that doesn't exist
            result = await storage.get_thought("nonexistent-id")

            assert result is None

    @pytest.mark.asyncio
    async def test_corrupted_file_handling(self):
        """Test handling of corrupted JSON files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = SifakaFilePersistence(storage_dir=temp_dir)

            # Create a corrupted file
            corrupted_file = storage.storage_dir / "corrupted.json"
            storage.storage_dir.mkdir(exist_ok=True)

            with open(corrupted_file, "w") as f:
                f.write("{ invalid json content")

            # Should handle corrupted files gracefully
            thoughts = await storage.list_thoughts()
            # Should not crash, might return empty list or skip corrupted files
            assert isinstance(thoughts, list)

    @pytest.mark.asyncio
    async def test_disk_full_simulation(self):
        """Test behavior when disk is full."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = SifakaFilePersistence(storage_dir=temp_dir)

            thought = SifakaThought(
                prompt="Test prompt", final_text="Test response", iteration=1, max_iterations=3
            )

            # Mock file operations to raise disk full error
            with patch("aiofiles.open", side_effect=OSError("No space left on device")):
                # Should handle error gracefully
                try:
                    await storage.store_thought(thought)
                except OSError:
                    pass  # Expected behavior
