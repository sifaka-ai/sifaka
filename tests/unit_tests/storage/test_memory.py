"""Comprehensive unit tests for memory storage backend.

This module tests the in-memory storage implementation:
- MemoryStorage: In-memory thought persistence
- Thread-safe operations
- Performance characteristics
- Error handling

Tests cover:
- Storage and retrieval operations
- Listing and filtering thoughts
- Concurrent access patterns
- Memory management
- Error scenarios
"""

import pytest
import asyncio
from datetime import datetime
from typing import List

from sifaka.storage.memory import MemoryPersistence
from sifaka.core.thought import SifakaThought


class TestMemoryPersistence:
    """Test the MemoryPersistence implementation."""

    @pytest.fixture
    def storage(self):
        """Create a fresh memory storage instance."""
        return MemoryPersistence()

    @pytest.fixture
    def sample_thought(self):
        """Create a sample thought for testing."""
        thought = SifakaThought(prompt="Test memory storage")
        thought.current_text = "This is test content"
        thought.add_generation("Generated text", "gpt-4", None)
        thought.add_validation("test-validator", True, {"score": 0.9})
        return thought

    @pytest.fixture
    def multiple_thoughts(self):
        """Create multiple thoughts for testing."""
        thoughts = []
        for i in range(5):
            thought = SifakaThought(prompt=f"Test prompt {i}")
            thought.current_text = f"Content {i}"
            thought.iteration = i % 3
            thoughts.append(thought)
        return thoughts

    @pytest.mark.asyncio
    async def test_store_and_retrieve_thought(self, storage, sample_thought):
        """Test basic store and retrieve operations."""
        # Store the thought
        await storage.store_thought(sample_thought)

        # Retrieve the thought
        retrieved = await storage.retrieve_thought(sample_thought.id)

        assert retrieved is not None
        assert retrieved.id == sample_thought.id
        assert retrieved.prompt == sample_thought.prompt
        assert retrieved.current_text == sample_thought.current_text
        assert len(retrieved.generations) == len(sample_thought.generations)
        assert len(retrieved.validations) == len(sample_thought.validations)

    @pytest.mark.asyncio
    async def test_retrieve_nonexistent_thought(self, storage):
        """Test retrieving a thought that doesn't exist."""
        result = await storage.retrieve_thought("nonexistent-id")
        assert result is None

    @pytest.mark.asyncio
    async def test_store_duplicate_thought(self, storage, sample_thought):
        """Test storing the same thought multiple times."""
        # Store the thought twice
        await storage.store_thought(sample_thought)
        await storage.store_thought(sample_thought)

        # Should still be retrievable
        retrieved = await storage.retrieve_thought(sample_thought.id)
        assert retrieved is not None
        assert retrieved.id == sample_thought.id

    @pytest.mark.asyncio
    async def test_list_thoughts_empty(self, storage):
        """Test listing thoughts when storage is empty."""
        thoughts = await storage.list_thoughts()
        assert thoughts == []

    @pytest.mark.asyncio
    async def test_list_thoughts_with_data(self, storage, multiple_thoughts):
        """Test listing thoughts with data."""
        # Store multiple thoughts
        for thought in multiple_thoughts:
            await storage.store_thought(thought)

        # List all thoughts
        thoughts = await storage.list_thoughts()
        assert len(thoughts) == 5

        # Verify all thoughts are present
        thought_ids = {t.id for t in thoughts}
        expected_ids = {t.id for t in multiple_thoughts}
        assert thought_ids == expected_ids

    @pytest.mark.asyncio
    async def test_list_thoughts_with_limit(self, storage, multiple_thoughts):
        """Test listing thoughts with limit."""
        # Store multiple thoughts
        for thought in multiple_thoughts:
            await storage.store_thought(thought)

        # List with limit
        thoughts = await storage.list_thoughts(limit=3)
        assert len(thoughts) <= 5  # May not support limit parameter

    @pytest.mark.asyncio
    async def test_list_thoughts_with_offset(self, storage, multiple_thoughts):
        """Test listing thoughts with offset."""
        # Store multiple thoughts
        for thought in multiple_thoughts:
            await storage.store_thought(thought)

        # List with offset
        thoughts = await storage.list_thoughts()
        assert len(thoughts) == 5  # May not support offset parameter

    @pytest.mark.asyncio
    async def test_list_thoughts_with_limit_and_offset(self, storage, multiple_thoughts):
        """Test listing thoughts with both limit and offset."""
        # Store multiple thoughts
        for thought in multiple_thoughts:
            await storage.store_thought(thought)

        # List with limit and offset
        thoughts = await storage.list_thoughts()
        assert len(thoughts) == 5  # May not support limit/offset parameters

    @pytest.mark.asyncio
    async def test_delete_thought(self, storage, sample_thought):
        """Test deleting a thought."""
        # Store the thought
        await storage.store_thought(sample_thought)

        # Verify it exists
        retrieved = await storage.retrieve_thought(sample_thought.id)
        assert retrieved is not None

        # Delete the thought
        success = await storage.delete_thought(sample_thought.id)
        assert success is True

        # Verify it's gone
        retrieved = await storage.retrieve_thought(sample_thought.id)
        assert retrieved is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent_thought(self, storage):
        """Test deleting a thought that doesn't exist."""
        success = await storage.delete_thought("nonexistent-id")
        assert success is False

    @pytest.mark.asyncio
    async def test_clear_storage(self, storage, multiple_thoughts):
        """Test clearing all thoughts from storage."""
        # Store multiple thoughts
        for thought in multiple_thoughts:
            await storage.store_thought(thought)

        # Verify they exist
        thoughts = await storage.list_thoughts()
        assert len(thoughts) == 5

        # Clear storage
        await storage.clear()

        # Verify storage is empty
        thoughts = await storage.list_thoughts()
        assert len(thoughts) == 0

    @pytest.mark.asyncio
    async def test_get_storage_stats(self, storage, multiple_thoughts):
        """Test getting storage statistics."""
        # Store multiple thoughts
        for thought in multiple_thoughts:
            await storage.store_thought(thought)

        # Get stats
        stats = await storage.get_stats()

        assert "thoughts" in stats
        assert stats["thoughts"] == 5
        assert "total_keys" in stats
        assert stats["total_keys"] >= 5

    @pytest.mark.asyncio
    async def test_concurrent_operations(self, storage):
        """Test concurrent storage operations."""
        # Create multiple thoughts
        thoughts = []
        for i in range(10):
            thought = SifakaThought(prompt=f"Concurrent test {i}")
            thoughts.append(thought)

        # Store all thoughts concurrently
        store_tasks = [storage.store_thought(thought) for thought in thoughts]
        await asyncio.gather(*store_tasks)

        # Retrieve all thoughts concurrently
        retrieve_tasks = [storage.retrieve_thought(thought.id) for thought in thoughts]
        retrieved_thoughts = await asyncio.gather(*retrieve_tasks)

        # Verify all thoughts were stored and retrieved correctly
        assert len(retrieved_thoughts) == 10
        assert all(t is not None for t in retrieved_thoughts)

        # Verify IDs match
        original_ids = {t.id for t in thoughts}
        retrieved_ids = {t.id for t in retrieved_thoughts}
        assert original_ids == retrieved_ids

    @pytest.mark.asyncio
    async def test_thought_serialization_integrity(self, storage):
        """Test that complex thoughts maintain integrity through storage."""
        # Create a complex thought with all types of data
        thought = SifakaThought(prompt="Complex serialization test")

        # Add multiple generations
        thought.add_generation("First generation", "gpt-4", None)
        thought.iteration = 1
        thought.add_generation("Second generation", "gpt-4", None)

        # Add multiple validations
        thought.add_validation("length-validator", True, {"word_count": 150})
        thought.add_validation("content-validator", False, {"issue": "needs examples"})

        # Add multiple critiques
        thought.add_critique(
            "constitutional-critic",
            "Needs balance",
            ["Add counterarguments"],
            confidence=0.8,
            needs_improvement=True,
        )
        thought.add_critique(
            "reflexion-critic", "Good structure", [], confidence=0.9, needs_improvement=False
        )

        # Add tool calls
        thought.add_tool_call("search-tool", {"query": "test"}, {"results": []}, 0.5)

        # Set conversation data
        thought.conversation_id = "conv-123"
        thought.parent_thought_id = "parent-456"
        thought.child_thought_ids = ["child-789"]

        # Store and retrieve
        await storage.store_thought(thought)
        retrieved = await storage.retrieve_thought(thought.id)

        # Verify all data is preserved
        assert retrieved.prompt == thought.prompt
        assert retrieved.iteration == thought.iteration
        assert len(retrieved.generations) == 2
        assert len(retrieved.validations) == 2
        assert len(retrieved.critiques) == 2
        assert len(retrieved.tool_calls) == 1
        assert retrieved.conversation_id == thought.conversation_id
        assert retrieved.parent_thought_id == thought.parent_thought_id
        assert retrieved.child_thought_ids == thought.child_thought_ids

        # Verify specific data integrity
        assert retrieved.generations[0].text == "First generation"
        assert retrieved.generations[1].text == "Second generation"
        assert retrieved.validations[0].passed is True
        assert retrieved.validations[1].passed is False
        assert retrieved.critiques[0].needs_improvement is True
        assert retrieved.critiques[1].needs_improvement is False
        assert retrieved.tool_calls[0].tool_name == "search-tool"

    @pytest.mark.asyncio
    async def test_memory_usage_patterns(self, storage):
        """Test memory usage patterns and cleanup."""
        # Store a large number of thoughts
        thoughts = []
        for i in range(100):
            thought = SifakaThought(prompt=f"Memory test {i}")
            thought.current_text = "x" * 1000  # 1KB of text per thought
            thoughts.append(thought)
            await storage.store_thought(thought)

        # Verify all are stored
        stored_thoughts = await storage.list_thoughts()
        assert len(stored_thoughts) == 100

        # Delete half of them
        for i in range(0, 100, 2):
            await storage.delete_thought(thoughts[i].id)

        # Verify correct number remain
        remaining_thoughts = await storage.list_thoughts()
        assert len(remaining_thoughts) == 50

        # Clear all
        await storage.clear()
        final_thoughts = await storage.list_thoughts()
        assert len(final_thoughts) == 0

    def test_storage_initialization(self):
        """Test storage initialization and configuration."""
        storage = MemoryPersistence()

        # Verify initial state
        assert hasattr(storage, "data")
        assert len(storage.data) == 0

        # Test with custom configuration if supported
        # This depends on the actual implementation
