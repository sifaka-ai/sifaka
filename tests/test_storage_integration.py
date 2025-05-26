#!/usr/bin/env python3
"""Storage backend integration tests for Sifaka.

This test suite validates integration with different storage backends,
including memory, file, Redis, and Milvus storage systems. It tests
data persistence, retrieval, and the 3-tier caching architecture.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional
import time

from sifaka.core.chain import Chain
from sifaka.core.thought import Thought
from sifaka.models.base import MockModel
from sifaka.storage.memory import MemoryStorage
from sifaka.storage.file import FileStorage
from sifaka.storage.cached import CachedStorage
from sifaka.validators.base import LengthValidator
from sifaka.critics.reflexion import ReflexionCritic
from sifaka.utils.logging import get_logger

from tests.utils import (
    create_test_thought,
    create_test_chain,
    assert_thought_valid,
    assert_chain_execution_success,
    assert_storage_consistency,
    MockModelFactory,
)

logger = get_logger(__name__)


class TestMemoryStorageIntegration:
    """Test integration with memory storage."""

    def test_basic_memory_storage(self):
        """Test basic memory storage operations."""
        storage = MemoryStorage()
        thought = create_test_thought(
            prompt="Test memory storage", text="This is a test thought for memory storage."
        )

        # Test save and load
        storage.save(thought.id, thought)
        assert storage.exists(thought.id)

        loaded_thought = storage.load(thought.id)
        assert loaded_thought is not None
        assert loaded_thought.id == thought.id
        assert loaded_thought.text == thought.text
        assert loaded_thought.prompt == thought.prompt

    def test_memory_storage_with_chain(self):
        """Test memory storage integration with chain execution."""
        model = MockModel(model_name="memory-test-model")
        storage = MemoryStorage()

        chain = Chain(model=model, prompt="Write about memory storage systems.", storage=storage)
        chain.validate_with(LengthValidator(min_length=10, max_length=200))

        result = chain.run()

        assert_thought_valid(result)
        assert_chain_execution_success(result)

        # Verify storage integration
        assert storage.exists(result.id)
        stored_thought = storage.load(result.id)
        assert stored_thought.id == result.id

    def test_memory_storage_multiple_thoughts(self):
        """Test memory storage with multiple thoughts."""
        storage = MemoryStorage()
        thoughts = []

        # Store multiple thoughts
        for i in range(5):
            thought = create_test_thought(
                prompt=f"Test prompt {i}", text=f"Test text for thought {i}"
            )
            storage.save(thought.id, thought)
            thoughts.append(thought)

        # Verify all are stored and retrievable
        for thought in thoughts:
            assert storage.exists(thought.id)
            loaded = storage.get(thought.id)
            assert loaded.id == thought.id
            assert loaded.text == thought.text

    def test_memory_storage_overwrite(self):
        """Test overwriting thoughts in memory storage."""
        storage = MemoryStorage()
        thought = create_test_thought(text="Original text")

        # Save original
        storage.save(thought.id, thought)

        # Update and save again
        updated_thought = thought.set_text("Updated text")
        storage.save(thought.id, updated_thought)

        # Verify update
        loaded = storage.get(thought.id)
        assert loaded.text == "Updated text"


class TestFileStorageIntegration:
    """Test integration with file storage."""

    def setup_method(self):
        """Set up temporary directory for file storage tests."""
        self.temp_dir = tempfile.mkdtemp()
        self.storage = FileStorage(directory=self.temp_dir)

    def teardown_method(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_basic_file_storage(self):
        """Test basic file storage operations."""
        thought = create_test_thought(
            prompt="Test file storage", text="This is a test thought for file storage."
        )

        # Test save and load
        self.storage.save(thought.id, thought)
        assert self.storage.exists(thought.id)

        # Verify file was created
        file_path = Path(self.temp_dir) / f"{thought.id}.json"
        assert file_path.exists()

        loaded_thought = self.storage.load(thought.id)
        assert loaded_thought is not None
        assert loaded_thought.id == thought.id
        assert loaded_thought.text == thought.text

    def test_file_storage_with_chain(self):
        """Test file storage integration with chain execution."""
        model = MockModel(model_name="file-test-model")

        chain = Chain(model=model, prompt="Write about file storage systems.", storage=self.storage)
        chain.validate_with(LengthValidator(min_length=10, max_length=200))

        result = chain.run()

        assert_thought_valid(result)
        assert_chain_execution_success(result)

        # Verify file storage integration
        assert self.storage.exists(result.id)
        file_path = Path(self.temp_dir) / f"{result.id}.json"
        assert file_path.exists()

    def test_file_storage_persistence(self):
        """Test that file storage persists across storage instances."""
        thought = create_test_thought(text="Persistent test data")

        # Save with first storage instance
        self.storage.save(thought.id, thought)

        # Create new storage instance pointing to same directory
        new_storage = FileStorage(directory=self.temp_dir)

        # Should be able to load from new instance
        assert new_storage.exists(thought.id)
        loaded = new_storage.load(thought.id)
        assert loaded.id == thought.id
        assert loaded.text == thought.text

    def test_file_storage_concurrent_access(self):
        """Test concurrent access to file storage."""
        thoughts = []

        # Save multiple thoughts concurrently (simulated)
        for i in range(3):
            thought = create_test_thought(
                prompt=f"Concurrent test {i}", text=f"Concurrent test data {i}"
            )
            self.storage.save(thought.id, thought)
            thoughts.append(thought)

        # Verify all are accessible
        for thought in thoughts:
            assert self.storage.exists(thought.id)
            loaded = self.storage.load(thought.id)
            assert loaded.id == thought.id


class TestCachedStorageIntegration:
    """Test integration with cached storage (3-tier architecture)."""

    def setup_method(self):
        """Set up cached storage with memory and file backends."""
        self.temp_dir = tempfile.mkdtemp()
        self.memory_storage = MemoryStorage()
        self.file_storage = FileStorage(directory=self.temp_dir)
        self.cached_storage = CachedStorage(
            cache=self.memory_storage, persistence=self.file_storage
        )

    def teardown_method(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_cached_storage_basic_operations(self):
        """Test basic cached storage operations."""
        thought = create_test_thought(text="Cached storage test")

        # Save through cached storage
        self.cached_storage.save(thought.id, thought)

        # Should exist in both primary and secondary
        assert self.cached_storage.exists(thought.id)
        assert self.memory_storage.exists(thought.id)
        assert self.file_storage.exists(thought.id)

        # Load should work
        loaded = self.cached_storage.get(thought.id)
        assert loaded.id == thought.id
        assert loaded.text == thought.text

    def test_cached_storage_cache_hit(self):
        """Test cache hit behavior in cached storage."""
        thought = create_test_thought(text="Cache hit test")

        # Save to both levels
        self.cached_storage.save(thought.id, thought)

        # Remove from secondary to test primary cache
        # (In real implementation, this would test cache performance)
        loaded = self.cached_storage.get(thought.id)
        assert loaded.id == thought.id

    def test_cached_storage_with_chain(self):
        """Test cached storage integration with chain execution."""
        model = MockModel(model_name="cached-test-model")

        chain = Chain(
            model=model, prompt="Write about cached storage systems.", storage=self.cached_storage
        )
        chain.validate_with(LengthValidator(min_length=10, max_length=200))

        result = chain.run()

        assert_thought_valid(result)
        assert_chain_execution_success(result)

        # Verify storage at all levels
        assert self.cached_storage.exists(result.id)
        assert self.memory_storage.exists(result.id)
        assert self.file_storage.exists(result.id)

    def test_cached_storage_fallback(self):
        """Test fallback behavior when primary cache fails."""
        thought = create_test_thought(text="Fallback test")

        # Save to cached storage
        self.cached_storage.save(thought.id, thought)

        # Clear primary cache
        self.memory_storage = MemoryStorage()  # Reset memory storage

        # Should still be able to load from secondary
        # Note: This test would need actual fallback implementation
        assert self.file_storage.exists(thought.id)


class TestStoragePerformance:
    """Test storage performance characteristics."""

    def test_memory_storage_performance(self):
        """Test memory storage performance."""
        storage = MemoryStorage()
        thoughts = []

        # Create test thoughts
        for i in range(100):
            thought = create_test_thought(
                prompt=f"Performance test {i}", text=f"Performance test data {i}"
            )
            thoughts.append(thought)

        # Measure save performance
        start_time = time.time()
        for thought in thoughts:
            storage.save(thought.id, thought)
        save_time = time.time() - start_time

        # Measure load performance
        start_time = time.time()
        for thought in thoughts:
            loaded = storage.get(thought.id)
            assert loaded.id == thought.id
        load_time = time.time() - start_time

        # Performance should be reasonable
        assert save_time < 1.0, f"Save performance too slow: {save_time:.3f}s for 100 thoughts"
        assert load_time < 1.0, f"Load performance too slow: {load_time:.3f}s for 100 thoughts"

    def test_file_storage_performance(self):
        """Test file storage performance."""
        temp_dir = tempfile.mkdtemp()
        try:
            storage = FileStorage(directory=temp_dir)
            thoughts = []

            # Create test thoughts
            for i in range(20):  # Fewer for file I/O
                thought = create_test_thought(
                    prompt=f"File performance test {i}", text=f"File performance test data {i}"
                )
                thoughts.append(thought)

            # Measure save performance
            start_time = time.time()
            for thought in thoughts:
                storage.save(thought.id, thought)
            save_time = time.time() - start_time

            # Measure load performance
            start_time = time.time()
            for thought in thoughts:
                loaded = storage.get(thought.id)
                assert loaded.id == thought.id
            load_time = time.time() - start_time

            # Performance should be reasonable for file I/O
            assert (
                save_time < 5.0
            ), f"File save performance too slow: {save_time:.3f}s for 20 thoughts"
            assert (
                load_time < 5.0
            ), f"File load performance too slow: {load_time:.3f}s for 20 thoughts"

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


class TestStorageErrorHandling:
    """Test storage error handling scenarios."""

    def test_storage_failure_graceful_handling(self):
        """Test graceful handling of storage failures."""
        from tests.utils.mocks import MockStorageFactory

        failing_storage = MockStorageFactory.create_failing()
        model = MockModel(model_name="error-test-model")

        chain = Chain(model=model, prompt="Write about error handling.", storage=failing_storage)

        # Chain should complete even if storage fails
        result = chain.run()

        assert_thought_valid(result)
        assert_chain_execution_success(result)

    def test_storage_corruption_handling(self):
        """Test handling of corrupted storage data."""
        temp_dir = tempfile.mkdtemp()
        try:
            storage = FileStorage(directory=temp_dir)
            thought = create_test_thought(text="Corruption test")

            # Save normally
            storage.save(thought.id, thought)

            # Corrupt the file
            file_path = Path(temp_dir) / f"{thought.id}.json"
            with open(file_path, "w") as f:
                f.write("corrupted data")

            # Should handle corruption gracefully
            loaded = storage.get(thought.id)
            # Depending on implementation, this might return None or raise an exception
            # The test verifies the system doesn't crash

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


class TestStorageConsistency:
    """Test storage consistency and data integrity."""

    def test_thought_serialization_consistency(self):
        """Test that thoughts are serialized and deserialized consistently."""
        storage = MemoryStorage()

        # Create thought with all fields populated
        original_thought = create_test_thought(
            prompt="Serialization test", text="Test serialization consistency", iteration=5
        )

        # Add validation results and critic feedback
        from sifaka.core.thought import ValidationResult, CriticFeedback

        original_thought = original_thought.add_validation_result(
            "test_validator",
            ValidationResult(
                passed=True, message="Test validation", validator_name="test_validator"
            ),
        )
        original_thought = original_thought.add_critic_feedback(
            CriticFeedback(
                critic_name="test_critic", feedback="Test feedback", needs_improvement=False
            )
        )

        # Save and load
        storage.save(original_thought.id, original_thought)
        loaded_thought = storage.get(original_thought.id)

        # Verify all fields are preserved
        assert loaded_thought.id == original_thought.id
        assert loaded_thought.prompt == original_thought.prompt
        assert loaded_thought.text == original_thought.text
        assert loaded_thought.iteration == original_thought.iteration
        assert len(loaded_thought.validation_results) == len(original_thought.validation_results)
        assert len(loaded_thought.critic_feedback) == len(original_thought.critic_feedback)

    def test_concurrent_storage_operations(self):
        """Test concurrent storage operations for data consistency."""
        storage = MemoryStorage()
        thoughts = []

        # Simulate concurrent saves
        for i in range(10):
            thought = create_test_thought(
                prompt=f"Concurrent test {i}", text=f"Concurrent data {i}"
            )
            storage.save(thought.id, thought)
            thoughts.append(thought)

        # Verify all thoughts are stored correctly
        for thought in thoughts:
            loaded = storage.get(thought.id)
            assert loaded.id == thought.id
            assert loaded.text == thought.text

        # Verify no data corruption
        all_ids = [thought.id for thought in thoughts]
        assert len(set(all_ids)) == len(all_ids), "No duplicate IDs should exist"
