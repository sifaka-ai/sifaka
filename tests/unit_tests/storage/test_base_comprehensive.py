"""Comprehensive unit tests for storage base classes and interfaces.

This module provides extensive testing for the storage infrastructure:
- SifakaBasePersistence abstract base class
- Storage protocol compliance and interface contracts
- Serialization and deserialization edge cases
- Key management and indexing functionality
- Error handling and logging scenarios
- Performance characteristics and optimization
"""

import asyncio
import uuid
from typing import Any, Dict, List, Optional

import pytest

from sifaka.core.thought import SifakaThought
from sifaka.storage.base import SifakaBasePersistence
from sifaka.utils.errors import ValidationError


class MockPersistence(SifakaBasePersistence):
    """Mock implementation of SifakaBasePersistence for testing."""

    def __init__(self):
        super().__init__()
        self._storage: Dict[str, Dict[str, Any]] = {}
        self._call_log: List[str] = []

    async def store_thought(self, thought: SifakaThought) -> None:
        """Store a thought in mock storage."""
        self._call_log.append(f"store_thought:{thought.id}")
        serialized = self.serialize_thought(thought)
        self._storage[thought.id] = serialized

    async def get_thought(self, thought_id: str) -> Optional[SifakaThought]:
        """Retrieve a thought from mock storage."""
        self._call_log.append(f"get_thought:{thought_id}")
        if thought_id in self._storage:
            return self.deserialize_thought(self._storage[thought_id])
        return None

    async def list_thoughts(self, limit: Optional[int] = None) -> List[SifakaThought]:
        """List thoughts from mock storage."""
        self._call_log.append(f"list_thoughts:limit={limit}")
        thoughts = []
        for data in list(self._storage.values())[:limit] if limit else self._storage.values():
            thoughts.append(self.deserialize_thought(data))
        return thoughts

    async def delete_thought(self, thought_id: str) -> bool:
        """Delete a thought from mock storage."""
        self._call_log.append(f"delete_thought:{thought_id}")
        if thought_id in self._storage:
            del self._storage[thought_id]
            return True
        return False

    async def search_thoughts(self, query: str, limit: Optional[int] = None) -> List[SifakaThought]:
        """Search thoughts in mock storage."""
        self._call_log.append(f"search_thoughts:{query}:limit={limit}")
        results = []
        for data in self._storage.values():
            thought = self.deserialize_thought(data)
            if (
                query.lower() in thought.prompt.lower()
                or query.lower() in thought.final_text.lower()
            ):
                results.append(thought)
        return results[:limit] if limit else results


class TestSifakaBasePersistence:
    """Test the abstract base persistence class."""

    def test_cannot_instantiate_abstract_class(self):
        """Test that SifakaBasePersistence cannot be instantiated directly."""
        with pytest.raises(TypeError):
            SifakaBasePersistence()

    def test_concrete_implementation_required_methods(self):
        """Test that concrete implementations must implement all abstract methods."""

        class IncompleteImplementation(SifakaBasePersistence):
            # Missing required methods
            pass

        with pytest.raises(TypeError):
            IncompleteImplementation()

    def test_serialization_deserialization_roundtrip(self):
        """Test that serialization and deserialization are inverse operations."""
        persistence = MockPersistence()

        # Create a complex thought with all possible data
        thought = SifakaThought(
            prompt="Test serialization", final_text="Final result", iteration=2, max_iterations=5
        )

        # Add various data types
        thought.add_generation("First generation", "gpt-4", {"temperature": 0.7})
        thought.add_validation("length_validator", True, {"word_count": 150})
        thought.add_critique("constitutional_critic", "Good work", ["Keep it up"], 0.9, False)

        # Serialize and deserialize
        serialized = persistence.serialize_thought(thought)
        deserialized = persistence.deserialize_thought(serialized)

        # Verify all fields are preserved
        assert deserialized.id == thought.id
        assert deserialized.prompt == thought.prompt
        assert deserialized.final_text == thought.final_text
        assert deserialized.iteration == thought.iteration
        assert deserialized.max_iterations == thought.max_iterations
        assert len(deserialized.generations) == len(thought.generations)
        assert len(deserialized.validations) == len(thought.validations)
        assert len(deserialized.critiques) == len(thought.critiques)

    def test_serialization_with_none_values(self):
        """Test serialization handles None values correctly."""
        persistence = MockPersistence()

        thought = SifakaThought(prompt="Test with None values")
        thought.final_text = None  # Explicitly set to None

        serialized = persistence.serialize_thought(thought)
        deserialized = persistence.deserialize_thought(serialized)

        assert deserialized.final_text is None
        assert deserialized.prompt == "Test with None values"

    def test_serialization_with_empty_collections(self):
        """Test serialization handles empty collections correctly."""
        persistence = MockPersistence()

        thought = SifakaThought(prompt="Empty collections test")
        # Don't add any generations, validations, or critiques

        serialized = persistence.serialize_thought(serialized)
        deserialized = persistence.deserialize_thought(serialized)

        assert len(deserialized.generations) == 0
        assert len(deserialized.validations) == 0
        assert len(deserialized.critiques) == 0

    def test_key_generation_uniqueness(self):
        """Test that generated keys are unique."""
        persistence = MockPersistence()

        keys = set()
        for _ in range(100):
            key = persistence.generate_key()
            assert key not in keys
            keys.add(key)
            assert isinstance(key, str)
            assert len(key) > 0

    def test_key_generation_format(self):
        """Test that generated keys follow expected format."""
        persistence = MockPersistence()

        key = persistence.generate_key()

        # Should be a valid UUID string
        try:
            uuid.UUID(key)
        except ValueError:
            pytest.fail(f"Generated key '{key}' is not a valid UUID")

    @pytest.mark.asyncio
    async def test_storage_operations_call_logging(self):
        """Test that storage operations are properly logged."""
        persistence = MockPersistence()

        thought = SifakaThought(prompt="Test logging")

        # Test store operation
        await persistence.store_thought(thought)
        assert f"store_thought:{thought.id}" in persistence._call_log

        # Test get operation
        retrieved = await persistence.get_thought(thought.id)
        assert f"get_thought:{thought.id}" in persistence._call_log
        assert retrieved is not None

        # Test list operation
        thoughts = await persistence.list_thoughts(limit=10)
        assert "list_thoughts:limit=10" in persistence._call_log
        assert len(thoughts) == 1

        # Test search operation
        results = await persistence.search_thoughts("Test", limit=5)
        assert "search_thoughts:Test:limit=5" in persistence._call_log
        assert len(results) == 1

        # Test delete operation
        deleted = await persistence.delete_thought(thought.id)
        assert f"delete_thought:{thought.id}" in persistence._call_log
        assert deleted is True

    @pytest.mark.asyncio
    async def test_nonexistent_thought_operations(self):
        """Test operations on non-existent thoughts."""
        persistence = MockPersistence()

        fake_id = str(uuid.uuid4())

        # Get non-existent thought
        result = await persistence.get_thought(fake_id)
        assert result is None

        # Delete non-existent thought
        deleted = await persistence.delete_thought(fake_id)
        assert deleted is False

    @pytest.mark.asyncio
    async def test_search_functionality(self):
        """Test search functionality with various queries."""
        persistence = MockPersistence()

        # Create test thoughts
        thoughts = [
            SifakaThought(prompt="Python programming", final_text="Code in Python"),
            SifakaThought(prompt="JavaScript development", final_text="Web development"),
            SifakaThought(prompt="Machine learning", final_text="AI and ML concepts"),
        ]

        for thought in thoughts:
            await persistence.store_thought(thought)

        # Test search by prompt
        results = await persistence.search_thoughts("Python")
        assert len(results) == 1
        assert "Python" in results[0].prompt

        # Test search by final_text
        results = await persistence.search_thoughts("development")
        assert len(results) == 1
        assert "development" in results[0].final_text

        # Test case-insensitive search
        results = await persistence.search_thoughts("PYTHON")
        assert len(results) == 1

        # Test no matches
        results = await persistence.search_thoughts("nonexistent")
        assert len(results) == 0

        # Test search with limit
        results = await persistence.search_thoughts("", limit=2)  # Empty query matches all
        assert len(results) == 2

    def test_serialization_error_handling(self):
        """Test error handling in serialization methods."""
        persistence = MockPersistence()

        # Test with invalid data
        with pytest.raises((ValueError, TypeError, ValidationError)):
            persistence.deserialize_thought({"invalid": "data"})

        # Test with None
        with pytest.raises((ValueError, TypeError)):
            persistence.deserialize_thought(None)

    @pytest.mark.asyncio
    async def test_concurrent_operations(self):
        """Test concurrent storage operations."""
        persistence = MockPersistence()

        # Create multiple thoughts
        thoughts = [SifakaThought(prompt=f"Concurrent test {i}") for i in range(10)]

        # Store all thoughts concurrently
        store_tasks = [persistence.store_thought(thought) for thought in thoughts]
        await asyncio.gather(*store_tasks)

        # Retrieve all thoughts concurrently
        get_tasks = [persistence.get_thought(thought.id) for thought in thoughts]
        results = await asyncio.gather(*get_tasks)

        # Verify all thoughts were stored and retrieved
        assert len(results) == 10
        assert all(result is not None for result in results)

        retrieved_ids = {result.id for result in results}
        expected_ids = {thought.id for thought in thoughts}
        assert retrieved_ids == expected_ids
