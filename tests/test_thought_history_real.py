#!/usr/bin/env python3
"""Tests for Sifaka thought history functionality.

This test suite covers the ThoughtHistory class which manages
the history of thoughts in a chain execution.
"""

import pytest
from datetime import datetime
from unittest.mock import Mock

from sifaka.core.thought import Thought, ThoughtReference
from sifaka.core.thought.history import ThoughtHistory


class TestThoughtHistory:
    """Test ThoughtHistory functionality."""

    def test_thought_history_creation(self):
        """Test basic ThoughtHistory creation."""
        history = ThoughtHistory()
        
        assert history.max_in_memory == 10  # default
        assert len(history._thoughts) == 0
        assert len(history._references) == 0
        assert len(history._chain_order) == 0

    def test_thought_history_custom_memory_limit(self):
        """Test ThoughtHistory with custom memory limit."""
        history = ThoughtHistory(max_in_memory=5)
        assert history.max_in_memory == 5

    def test_add_thought(self):
        """Test adding thoughts to history."""
        history = ThoughtHistory()
        thought = Thought(prompt="Test prompt", text="Test response")
        
        history.add_thought(thought)
        
        # Check that thought was added
        assert thought.id in history._thoughts
        assert history._thoughts[thought.id] == thought
        
        # Check chain tracking
        chain_id = thought.chain_id or "default"
        assert chain_id in history._references
        assert chain_id in history._chain_order
        assert len(history._chain_order[chain_id]) == 1
        assert len(history._references[chain_id]) == 1

    def test_add_multiple_thoughts(self):
        """Test adding multiple thoughts to history."""
        history = ThoughtHistory()
        
        thoughts = [
            Thought(prompt="Prompt 1", text="Response 1"),
            Thought(prompt="Prompt 2", text="Response 2"),
            Thought(prompt="Prompt 3", text="Response 3"),
        ]
        
        for thought in thoughts:
            history.add_thought(thought)
        
        # Check all thoughts were added
        assert len(history._thoughts) == 3
        
        # Check chain order
        chain_id = "default"
        assert len(history._chain_order[chain_id]) == 3
        assert len(history._references[chain_id]) == 3

    def test_get_thought(self):
        """Test getting thoughts by ID."""
        history = ThoughtHistory()
        thought = Thought(prompt="Test prompt", text="Test response")
        
        history.add_thought(thought)
        
        # Test valid ID
        retrieved = history.get_thought(thought.id)
        assert retrieved == thought
        
        # Test invalid ID
        assert history.get_thought("nonexistent") is None

    def test_get_latest_thought(self):
        """Test getting latest thought for a chain."""
        history = ThoughtHistory()
        
        # Empty chain
        assert history.get_latest_thought("default") is None
        
        # Add thoughts
        thought1 = Thought(prompt="Prompt 1", text="Response 1")
        thought2 = Thought(prompt="Prompt 2", text="Response 2")
        
        history.add_thought(thought1)
        history.add_thought(thought2)
        
        # Latest should be thought2
        latest = history.get_latest_thought("default")
        assert latest == thought2

    def test_get_thought_by_iteration(self):
        """Test getting thought by iteration number."""
        history = ThoughtHistory()
        
        # Create thoughts with specific iterations
        thought1 = Thought(prompt="Prompt 1", text="Response 1", iteration=1)
        thought2 = Thought(prompt="Prompt 2", text="Response 2", iteration=2)
        
        history.add_thought(thought1)
        history.add_thought(thought2)
        
        # Test finding by iteration
        found1 = history.get_thought_by_iteration("default", 1)
        found2 = history.get_thought_by_iteration("default", 2)
        
        assert found1 == thought1
        assert found2 == thought2
        
        # Test non-existent iteration
        assert history.get_thought_by_iteration("default", 99) is None

    def test_get_chain_references(self):
        """Test getting chain references."""
        history = ThoughtHistory()
        
        # Empty chain
        refs = history.get_chain_references("default")
        assert refs == []
        
        # Add thoughts
        thought1 = Thought(prompt="Prompt 1", text="Response 1")
        thought2 = Thought(prompt="Prompt 2", text="Response 2")
        
        history.add_thought(thought1)
        history.add_thought(thought2)
        
        # Get references
        refs = history.get_chain_references("default")
        assert len(refs) == 2
        assert all(isinstance(ref, ThoughtReference) for ref in refs)

    def test_get_chain_thoughts(self):
        """Test getting all thoughts for a chain."""
        history = ThoughtHistory()
        
        # Empty chain
        thoughts = history.get_chain_thoughts("default")
        assert thoughts == []
        
        # Add thoughts
        thought1 = Thought(prompt="Prompt 1", text="Response 1")
        thought2 = Thought(prompt="Prompt 2", text="Response 2")
        
        history.add_thought(thought1)
        history.add_thought(thought2)
        
        # Get all thoughts
        thoughts = history.get_chain_thoughts("default")
        assert len(thoughts) == 2
        assert thought1 in thoughts
        assert thought2 in thoughts

    def test_get_iteration_count(self):
        """Test getting iteration count for a chain."""
        history = ThoughtHistory()
        
        # Empty chain
        assert history.get_iteration_count("default") == 0
        
        # Add thoughts
        for i in range(3):
            thought = Thought(prompt=f"Prompt {i}", text=f"Response {i}")
            history.add_thought(thought)
        
        assert history.get_iteration_count("default") == 3

    def test_remove_thought(self):
        """Test removing thoughts from history."""
        history = ThoughtHistory()
        thought = Thought(prompt="Test prompt", text="Test response")
        
        history.add_thought(thought)
        assert thought.id in history._thoughts
        
        # Remove thought
        success = history.remove_thought(thought.id)
        assert success
        assert thought.id not in history._thoughts
        
        # Try to remove non-existent thought
        success = history.remove_thought("nonexistent")
        assert not success

    def test_clear_chain(self):
        """Test clearing all thoughts for a chain."""
        history = ThoughtHistory()
        
        # Add thoughts
        thoughts = []
        for i in range(3):
            thought = Thought(prompt=f"Prompt {i}", text=f"Response {i}")
            thoughts.append(thought)
            history.add_thought(thought)
        
        assert len(history._thoughts) == 3
        
        # Clear chain
        count = history.clear_chain("default")
        assert count == 3
        assert len(history._thoughts) == 0
        assert "default" not in history._references
        assert "default" not in history._chain_order

    def test_get_memory_usage(self):
        """Test getting memory usage statistics."""
        history = ThoughtHistory()
        
        # Empty history
        stats = history.get_memory_usage()
        assert stats["total_thoughts_in_memory"] == 0
        assert stats["total_chains"] == 0
        assert stats["total_references"] == 0
        assert stats["average_thoughts_per_chain"] == 0
        
        # Add thoughts
        for i in range(3):
            thought = Thought(prompt=f"Prompt {i}", text=f"Response {i}")
            history.add_thought(thought)
        
        stats = history.get_memory_usage()
        assert stats["total_thoughts_in_memory"] == 3
        assert stats["total_chains"] == 1
        assert stats["total_references"] == 3
        assert stats["average_thoughts_per_chain"] == 3

    def test_memory_cleanup(self):
        """Test memory cleanup when exceeding limit."""
        history = ThoughtHistory(max_in_memory=2)
        
        # Add more thoughts than the limit
        thoughts = []
        for i in range(4):
            thought = Thought(prompt=f"Prompt {i}", text=f"Response {i}")
            thoughts.append(thought)
            history.add_thought(thought)
        
        # Should only keep the last 2 thoughts in memory
        assert len(history._thoughts) <= 2
        
        # But references should still exist for all
        assert len(history._references["default"]) == 4

    def test_optimize_memory(self):
        """Test memory optimization."""
        history = ThoughtHistory(max_in_memory=2)
        
        # Add many thoughts
        for i in range(5):
            thought = Thought(prompt=f"Prompt {i}", text=f"Response {i}")
            history.add_thought(thought)
        
        # Optimize memory
        removed_count = history.optimize_memory()
        assert removed_count >= 0
        
        # Should respect memory limit
        assert len(history._thoughts) <= 2

    def test_multiple_chains(self):
        """Test handling multiple chains."""
        history = ThoughtHistory()
        
        # Add thoughts to different chains
        thought1 = Thought(prompt="Prompt 1", text="Response 1", chain_id="chain1")
        thought2 = Thought(prompt="Prompt 2", text="Response 2", chain_id="chain2")
        
        history.add_thought(thought1)
        history.add_thought(thought2)
        
        # Check separate chain tracking
        assert len(history._references) == 2
        assert "chain1" in history._references
        assert "chain2" in history._references
        
        # Check chain-specific operations
        assert history.get_iteration_count("chain1") == 1
        assert history.get_iteration_count("chain2") == 1
        
        latest1 = history.get_latest_thought("chain1")
        latest2 = history.get_latest_thought("chain2")
        
        assert latest1 == thought1
        assert latest2 == thought2
