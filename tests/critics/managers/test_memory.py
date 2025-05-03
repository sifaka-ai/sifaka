"""
Tests for the critics MemoryManager class.

This module contains comprehensive tests for the MemoryManager class that
is used to store and retrieve reflections and other memory items for critics.

Tests cover all public methods including add_to_memory, get_memory, clear_memory,
and the memory_size property. Both normal and edge cases are tested.
"""

import unittest
from unittest.mock import patch
import pytest

from sifaka.critics.managers.memory import MemoryManager


class TestMemoryManager(unittest.TestCase):
    """Tests for the MemoryManager class."""

    def setUp(self):
        """Set up test fixtures."""
        self.memory_manager = MemoryManager(buffer_size=5)

    def test_initialization(self):
        """Test initialization with different buffer sizes."""
        # Test default initialization
        memory_manager = MemoryManager()
        self.assertEqual(memory_manager._buffer_size, 5)
        self.assertEqual(memory_manager.memory_size, 0)

        # Test initialization with custom buffer size
        memory_manager = MemoryManager(buffer_size=10)
        self.assertEqual(memory_manager._buffer_size, 10)
        self.assertEqual(memory_manager.memory_size, 0)

        # Test initialization with negative buffer size (should convert to 1)
        memory_manager = MemoryManager(buffer_size=-5)
        self.assertEqual(memory_manager._buffer_size, 1)
        self.assertEqual(memory_manager.memory_size, 0)

        # Test initialization with zero buffer size (should convert to 1)
        memory_manager = MemoryManager(buffer_size=0)
        self.assertEqual(memory_manager._buffer_size, 1)
        self.assertEqual(memory_manager.memory_size, 0)

    def test_add_to_memory(self):
        """Test adding items to memory."""
        # Create a fresh memory manager
        memory_manager = MemoryManager(buffer_size=5)

        # Add a single item
        memory_manager.add_to_memory("Item 1")
        self.assertEqual(memory_manager.memory_size, 1)
        self.assertEqual(memory_manager.get_memory(), ["Item 1"])

        # Add multiple items
        memory_manager.add_to_memory("Item 2")
        memory_manager.add_to_memory("Item 3")
        self.assertEqual(memory_manager.memory_size, 3)
        self.assertEqual(memory_manager.get_memory(), ["Item 1", "Item 2", "Item 3"])

    def test_add_to_memory_with_empty_items(self):
        """Test adding empty items to memory."""
        # Create a fresh memory manager
        memory_manager = MemoryManager(buffer_size=5)

        # Add empty string (should be ignored)
        memory_manager.add_to_memory("")
        self.assertEqual(memory_manager.memory_size, 0)

        # Add whitespace string (should be ignored)
        memory_manager.add_to_memory("   ")
        self.assertEqual(memory_manager.memory_size, 0)

    def test_add_to_memory_beyond_capacity(self):
        """Test adding items beyond the buffer capacity."""
        # Create a fresh memory manager
        memory_manager = MemoryManager(buffer_size=5)

        # Fill the buffer to capacity
        for i in range(5):
            memory_manager.add_to_memory(f"Item {i+1}")

        self.assertEqual(memory_manager.memory_size, 5)
        self.assertEqual(
            memory_manager.get_memory(),
            ["Item 1", "Item 2", "Item 3", "Item 4", "Item 5"]
        )

        # Add one more item (should discard the oldest item)
        memory_manager.add_to_memory("Item 6")
        self.assertEqual(memory_manager.memory_size, 5)
        self.assertEqual(
            memory_manager.get_memory(),
            ["Item 2", "Item 3", "Item 4", "Item 5", "Item 6"]
        )

    def test_add_to_memory_with_whitespace_stripping(self):
        """Test that whitespace is stripped from items."""
        # Create a fresh memory manager
        memory_manager = MemoryManager(buffer_size=5)

        memory_manager.add_to_memory("  Item with spaces  ")
        self.assertEqual(memory_manager.get_memory(), ["Item with spaces"])

    def test_get_memory(self):
        """Test retrieving all items from memory."""
        # Create a fresh memory manager
        memory_manager = MemoryManager(buffer_size=5)

        # Add items to memory
        items = ["Item 1", "Item 2", "Item 3"]
        for item in items:
            memory_manager.add_to_memory(item)

        # Get all items
        result = memory_manager.get_memory()
        self.assertEqual(result, items)

    def test_get_memory_with_limited_items(self):
        """Test retrieving a limited number of items from memory."""
        # Create a fresh memory manager
        memory_manager = MemoryManager(buffer_size=5)

        # Add items to memory
        for i in range(1, 6):
            memory_manager.add_to_memory(f"Item {i}")

        # Get the most recent 3 items
        result = memory_manager.get_memory(max_items=3)
        self.assertEqual(result, ["Item 3", "Item 4", "Item 5"])

        # Get the most recent 1 item
        result = memory_manager.get_memory(max_items=1)
        self.assertEqual(result, ["Item 5"])

    def test_get_memory_after_rotation(self):
        """Test get_memory after buffer rotations."""
        # Create a fresh memory manager
        memory_manager = MemoryManager(buffer_size=5)

        # Fill the buffer to capacity
        for i in range(5):
            memory_manager.add_to_memory(f"Item {i+1}")

        # Add one more item to rotate the buffer
        memory_manager.add_to_memory("Item 6")

        # Get all items
        result = memory_manager.get_memory()
        self.assertEqual(result, ["Item 2", "Item 3", "Item 4", "Item 5", "Item 6"])

        # Get the most recent 3 items
        result = memory_manager.get_memory(max_items=3)
        self.assertEqual(result, ["Item 4", "Item 5", "Item 6"])

    def test_clear_memory(self):
        """Test clearing memory."""
        # Create a fresh memory manager
        memory_manager = MemoryManager(buffer_size=5)

        # Add items to memory
        for i in range(3):
            memory_manager.add_to_memory(f"Item {i+1}")

        # Verify items are in memory
        self.assertEqual(memory_manager.memory_size, 3)

        # Clear memory
        memory_manager.clear_memory()

        # Verify memory is empty
        self.assertEqual(memory_manager.memory_size, 0)
        self.assertEqual(memory_manager.get_memory(), [])

    def test_memory_size_property(self):
        """Test the memory_size property."""
        # Create a fresh memory manager
        memory_manager = MemoryManager(buffer_size=5)

        # Empty memory
        self.assertEqual(memory_manager.memory_size, 0)

        # Add items
        memory_manager.add_to_memory("Item 1")
        self.assertEqual(memory_manager.memory_size, 1)

        memory_manager.add_to_memory("Item 2")
        self.assertEqual(memory_manager.memory_size, 2)

        # Clear memory
        memory_manager.clear_memory()
        self.assertEqual(memory_manager.memory_size, 0)

    @patch('sifaka.critics.managers.memory.logger')
    def test_logging(self, mock_logger):
        """Test that logging occurs in add_to_memory and clear_memory."""
        # Create a fresh memory manager
        memory_manager = MemoryManager(buffer_size=5)

        # Add item to memory
        memory_manager.add_to_memory("Item 1")
        mock_logger.debug.assert_called_with("Added item to memory buffer (size: 1)")

        # Clear memory
        memory_manager.clear_memory()
        mock_logger.debug.assert_called_with("Cleared memory buffer")


if __name__ == "__main__":
    unittest.main()