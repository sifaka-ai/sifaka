"""Comprehensive unit tests for storage base classes.

This module tests the base storage infrastructure:
- SifakaBasePersistence abstract base class
- Storage protocol compliance
- Serialization and deserialization
- Key management and indexing
- Error handling and logging

Tests cover:
- Abstract base class requirements
- Serialization/deserialization methods
- Key generation and management
- Index management for thoughts
- Error handling scenarios
"""

import pytest
import asyncio
from datetime import datetime
from typing import List, Optional, Dict, Any
from unittest.mock import Mock, AsyncMock, patch
from abc import ABC

from sifaka.storage.base import SifakaBasePersistence
from sifaka.core.thought import SifakaThought


class TestSifakaBasePersistence:
    """Test the SifakaBasePersistence abstract base class."""

    def test_base_persistence_is_abstract(self):
        """Test that SifakaBasePersistence cannot be instantiated directly."""
        with pytest.raises(TypeError):
            SifakaBasePersistence()

    def test_base_persistence_subclass_requirements(self):
        """Test that subclasses must implement abstract methods."""
        class IncompleteStorage(SifakaBasePersistence):
            # Missing required abstract methods
            pass
        
        with pytest.raises(TypeError):
            IncompleteStorage()

    def test_base_persistence_initialization(self):
        """Test SifakaBasePersistence initialization."""
        class ConcreteStorage(SifakaBasePersistence):
            async def _store_raw(self, key: str, data: str) -> None:
                pass
            
            async def _retrieve_raw(self, key: str) -> Optional[str]:
                return None
            
            async def _delete_raw(self, key: str) -> bool:
                return False
            
            async def _list_keys(self, pattern: str) -> List[str]:
                return []
        
        # Test with default prefix
        storage = ConcreteStorage()
        assert storage.key_prefix == "sifaka"
        
        # Test with custom prefix
        storage_custom = ConcreteStorage(key_prefix="custom")
        assert storage_custom.key_prefix == "custom"

    def test_key_generation(self):
        """Test key generation methods."""
        class ConcreteStorage(SifakaBasePersistence):
            async def _store_raw(self, key: str, data: str) -> None:
                pass
            
            async def _retrieve_raw(self, key: str) -> Optional[str]:
                return None
            
            async def _delete_raw(self, key: str) -> bool:
                return False
            
            async def _list_keys(self, pattern: str) -> List[str]:
                return []
        
        storage = ConcreteStorage(key_prefix="test")
        
        # Test _make_key method
        key = storage._make_key("thought-123")
        assert key == "test:thought-123"
        
        # Test with different prefix
        storage_custom = ConcreteStorage(key_prefix="custom")
        key_custom = storage_custom._make_key("thought-456")
        assert key_custom == "custom:thought-456"

    @pytest.mark.asyncio
    async def test_serialization_methods(self):
        """Test thought serialization and deserialization."""
        class ConcreteStorage(SifakaBasePersistence):
            async def _store_raw(self, key: str, data: str) -> None:
                self._data = data
            
            async def _retrieve_raw(self, key: str) -> Optional[str]:
                return getattr(self, '_data', None)
            
            async def _delete_raw(self, key: str) -> bool:
                return True
            
            async def _list_keys(self, pattern: str) -> List[str]:
                return []
        
        storage = ConcreteStorage()
        
        # Create a test thought
        thought = SifakaThought(
            prompt="Test serialization",
            current_text="Test content",
            iteration=1,
        )
        
        # Test serialization
        serialized = await storage.serialize_state(thought)
        assert isinstance(serialized, str)
        
        # Test deserialization
        deserialized = await storage.deserialize_state(serialized)
        assert isinstance(deserialized, SifakaThought)
        assert deserialized.prompt == thought.prompt
        assert deserialized.current_text == thought.current_text
        assert deserialized.iteration == thought.iteration

    @pytest.mark.asyncio
    async def test_store_and_retrieve_thought(self):
        """Test storing and retrieving thoughts."""
        class ConcreteStorage(SifakaBasePersistence):
            def __init__(self, key_prefix: str = "sifaka"):
                super().__init__(key_prefix)
                self._storage = {}
            
            async def _store_raw(self, key: str, data: str) -> None:
                self._storage[key] = data
            
            async def _retrieve_raw(self, key: str) -> Optional[str]:
                return self._storage.get(key)
            
            async def _delete_raw(self, key: str) -> bool:
                if key in self._storage:
                    del self._storage[key]
                    return True
                return False
            
            async def _list_keys(self, pattern: str) -> List[str]:
                return list(self._storage.keys())
            
            async def _update_indexes(self, thought: SifakaThought) -> None:
                # Mock index update
                pass
        
        storage = ConcreteStorage()
        
        # Create test thought
        thought = SifakaThought(
            prompt="Test storage",
            current_text="Test content",
            conversation_id="conv-123",
        )
        
        # Store thought
        await storage.store_thought(thought)
        
        # Retrieve thought
        retrieved = await storage.retrieve_thought(thought.id)
        
        assert retrieved is not None
        assert retrieved.id == thought.id
        assert retrieved.prompt == thought.prompt
        assert retrieved.current_text == thought.current_text
        assert retrieved.conversation_id == thought.conversation_id

    @pytest.mark.asyncio
    async def test_retrieve_nonexistent_thought(self):
        """Test retrieving a thought that doesn't exist."""
        class ConcreteStorage(SifakaBasePersistence):
            async def _store_raw(self, key: str, data: str) -> None:
                pass
            
            async def _retrieve_raw(self, key: str) -> Optional[str]:
                return None
            
            async def _delete_raw(self, key: str) -> bool:
                return False
            
            async def _list_keys(self, pattern: str) -> List[str]:
                return []
        
        storage = ConcreteStorage()
        
        # Try to retrieve non-existent thought
        result = await storage.retrieve_thought("non-existent-id")
        assert result is None

    @pytest.mark.asyncio
    async def test_delete_thought(self):
        """Test deleting thoughts."""
        class ConcreteStorage(SifakaBasePersistence):
            def __init__(self, key_prefix: str = "sifaka"):
                super().__init__(key_prefix)
                self._storage = {}
            
            async def _store_raw(self, key: str, data: str) -> None:
                self._storage[key] = data
            
            async def _retrieve_raw(self, key: str) -> Optional[str]:
                return self._storage.get(key)
            
            async def _delete_raw(self, key: str) -> bool:
                if key in self._storage:
                    del self._storage[key]
                    return True
                return False
            
            async def _list_keys(self, pattern: str) -> List[str]:
                return list(self._storage.keys())
            
            async def _update_indexes(self, thought: SifakaThought) -> None:
                pass
        
        storage = ConcreteStorage()
        
        # Create and store thought
        thought = SifakaThought(prompt="Test delete")
        await storage.store_thought(thought)
        
        # Verify it exists
        retrieved = await storage.retrieve_thought(thought.id)
        assert retrieved is not None
        
        # Delete thought
        deleted = await storage.delete_thought(thought.id)
        assert deleted is True
        
        # Verify it's gone
        retrieved_after = await storage.retrieve_thought(thought.id)
        assert retrieved_after is None
        
        # Try to delete again
        deleted_again = await storage.delete_thought(thought.id)
        assert deleted_again is False

    @pytest.mark.asyncio
    async def test_list_thoughts(self):
        """Test listing thoughts with filtering."""
        class ConcreteStorage(SifakaBasePersistence):
            def __init__(self, key_prefix: str = "sifaka"):
                super().__init__(key_prefix)
                self._storage = {}
            
            async def _store_raw(self, key: str, data: str) -> None:
                self._storage[key] = data
            
            async def _retrieve_raw(self, key: str) -> Optional[str]:
                return self._storage.get(key)
            
            async def _delete_raw(self, key: str) -> bool:
                if key in self._storage:
                    del self._storage[key]
                    return True
                return False
            
            async def _list_keys(self, pattern: str) -> List[str]:
                # Simple pattern matching for test
                if "*" in pattern:
                    prefix = pattern.replace("*", "")
                    return [k for k in self._storage.keys() if k.startswith(prefix)]
                return [k for k in self._storage.keys() if pattern in k]
            
            async def _update_indexes(self, thought: SifakaThought) -> None:
                pass
        
        storage = ConcreteStorage()
        
        # Create thoughts with different conversation IDs
        thought1 = SifakaThought(prompt="Thought 1", conversation_id="conv-1")
        thought2 = SifakaThought(prompt="Thought 2", conversation_id="conv-1")
        thought3 = SifakaThought(prompt="Thought 3", conversation_id="conv-2")
        
        # Store thoughts
        await storage.store_thought(thought1)
        await storage.store_thought(thought2)
        await storage.store_thought(thought3)
        
        # List all thoughts
        all_thoughts = await storage.list_thoughts()
        assert len(all_thoughts) == 3
        
        # List thoughts by conversation (if filtering is implemented)
        # Note: This depends on the actual implementation of list_thoughts
        # The test documents expected behavior

    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error handling in storage operations."""
        class ErrorStorage(SifakaBasePersistence):
            async def _store_raw(self, key: str, data: str) -> None:
                raise Exception("Storage error")
            
            async def _retrieve_raw(self, key: str) -> Optional[str]:
                raise Exception("Retrieval error")
            
            async def _delete_raw(self, key: str) -> bool:
                raise Exception("Deletion error")
            
            async def _list_keys(self, pattern: str) -> List[str]:
                raise Exception("List error")
        
        storage = ErrorStorage()
        thought = SifakaThought(prompt="Test error")
        
        # Store should handle errors gracefully
        with pytest.raises(Exception):
            await storage.store_thought(thought)
        
        # Retrieve should handle errors gracefully
        with pytest.raises(Exception):
            await storage.retrieve_thought("test-id")
        
        # Delete should handle errors gracefully
        result = await storage.delete_thought("test-id")
        assert result is False  # Should return False on error

    @patch('sifaka.storage.base.logger')
    @pytest.mark.asyncio
    async def test_logging(self, mock_logger):
        """Test that storage operations are properly logged."""
        class LoggingStorage(SifakaBasePersistence):
            def __init__(self, key_prefix: str = "sifaka"):
                super().__init__(key_prefix)
                self._storage = {}
            
            async def _store_raw(self, key: str, data: str) -> None:
                self._storage[key] = data
            
            async def _retrieve_raw(self, key: str) -> Optional[str]:
                return self._storage.get(key)
            
            async def _delete_raw(self, key: str) -> bool:
                return True
            
            async def _list_keys(self, pattern: str) -> List[str]:
                return []
            
            async def _update_indexes(self, thought: SifakaThought) -> None:
                pass
        
        storage = LoggingStorage()
        thought = SifakaThought(prompt="Test logging")
        
        # Store thought
        await storage.store_thought(thought)
        
        # Verify logging was called
        mock_logger.debug.assert_called()

    @pytest.mark.asyncio
    async def test_serialization_error_handling(self):
        """Test handling of serialization errors."""
        class ConcreteStorage(SifakaBasePersistence):
            async def _store_raw(self, key: str, data: str) -> None:
                pass
            
            async def _retrieve_raw(self, key: str) -> Optional[str]:
                return "invalid json data"
            
            async def _delete_raw(self, key: str) -> bool:
                return True
            
            async def _list_keys(self, pattern: str) -> List[str]:
                return []
        
        storage = ConcreteStorage()
        
        # Try to deserialize invalid data
        with pytest.raises(Exception):  # Could be JSON decode error or validation error
            await storage.deserialize_state("invalid json")

    @pytest.mark.asyncio
    async def test_concurrent_operations(self):
        """Test concurrent storage operations."""
        class ConcurrentStorage(SifakaBasePersistence):
            def __init__(self, key_prefix: str = "sifaka"):
                super().__init__(key_prefix)
                self._storage = {}
                self._lock = asyncio.Lock()
            
            async def _store_raw(self, key: str, data: str) -> None:
                async with self._lock:
                    await asyncio.sleep(0.01)  # Simulate async operation
                    self._storage[key] = data
            
            async def _retrieve_raw(self, key: str) -> Optional[str]:
                async with self._lock:
                    await asyncio.sleep(0.01)
                    return self._storage.get(key)
            
            async def _delete_raw(self, key: str) -> bool:
                return True
            
            async def _list_keys(self, pattern: str) -> List[str]:
                return []
            
            async def _update_indexes(self, thought: SifakaThought) -> None:
                pass
        
        storage = ConcurrentStorage()
        
        # Create multiple thoughts
        thoughts = [
            SifakaThought(prompt=f"Thought {i}")
            for i in range(5)
        ]
        
        # Store thoughts concurrently
        await asyncio.gather(*[
            storage.store_thought(thought)
            for thought in thoughts
        ])
        
        # Retrieve thoughts concurrently
        retrieved = await asyncio.gather(*[
            storage.retrieve_thought(thought.id)
            for thought in thoughts
        ])
        
        # Verify all thoughts were stored and retrieved
        assert len(retrieved) == 5
        assert all(t is not None for t in retrieved)
        
        # Verify thought IDs match
        retrieved_ids = {t.id for t in retrieved if t is not None}
        original_ids = {t.id for t in thoughts}
        assert retrieved_ids == original_ids
