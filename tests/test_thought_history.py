#!/usr/bin/env python3
"""
Test thought history functionality specifically.

This module focuses on testing the thought iteration and history tracking
features of the persistence layer.
"""

import tempfile
from sifaka.core.thought import Thought
from sifaka.persistence.json import JSONThoughtStorage


def test_thought_iterations():
    """Test that thought iterations create unique IDs and proper parent relationships."""
    print("Testing thought iterations...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        storage = JSONThoughtStorage(storage_dir=temp_dir)
        
        # Create initial thought
        thought1 = Thought(
            prompt="Explain quantum computing",
            text="Quantum computing is a type of computation...",
            chain_id="quantum-chain"
        )
        storage.save_thought(thought1)
        print(f"✓ Thought 1 saved: {thought1.id}")
        
        # Create next iteration
        thought2 = thought1.next_iteration()
        thought2 = thought2.set_text("Quantum computing is a revolutionary approach...")
        storage.save_thought(thought2)
        print(f"✓ Thought 2 saved: {thought2.id}")
        
        # Create third iteration
        thought3 = thought2.next_iteration()
        thought3 = thought3.set_text("Quantum computing represents a paradigm shift...")
        storage.save_thought(thought3)
        print(f"✓ Thought 3 saved: {thought3.id}")
        
        # Verify IDs are different
        assert thought1.id != thought2.id != thought3.id, "Thoughts should have unique IDs"
        print("✓ All thoughts have unique IDs")
        
        # Test parent relationships
        assert thought2.parent_id == thought1.id, f"Thought2 parent should be {thought1.id}, got {thought2.parent_id}"
        assert thought3.parent_id == thought2.id, f"Thought3 parent should be {thought2.id}, got {thought3.parent_id}"
        print("✓ Parent relationships correct")
        
        # Test iteration numbers
        assert thought1.iteration == 0, f"Thought1 iteration should be 0, got {thought1.iteration}"
        assert thought2.iteration == 1, f"Thought2 iteration should be 1, got {thought2.iteration}"
        assert thought3.iteration == 2, f"Thought3 iteration should be 2, got {thought3.iteration}"
        print("✓ Iteration numbers correct")


def test_history_retrieval():
    """Test retrieving complete thought history."""
    print("\nTesting history retrieval...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        storage = JSONThoughtStorage(storage_dir=temp_dir)
        
        # Create a chain of thoughts
        thought1 = Thought(prompt="Original prompt", text="Original text")
        storage.save_thought(thought1)
        
        thought2 = thought1.next_iteration()
        thought2 = thought2.set_text("Improved text")
        storage.save_thought(thought2)
        
        thought3 = thought2.next_iteration()
        thought3 = thought3.set_text("Final text")
        storage.save_thought(thought3)
        
        # Test history retrieval from latest thought
        history = storage.get_thought_history(thought3.id)
        assert len(history) == 3, f"Expected 3 thoughts in history, got {len(history)}"
        
        # Verify chronological order (oldest first)
        assert history[0].id == thought1.id, "First thought in history should be original"
        assert history[1].id == thought2.id, "Second thought in history should be first iteration"
        assert history[2].id == thought3.id, "Third thought in history should be latest"
        print("✓ History retrieved in correct chronological order")
        
        # Test history retrieval from middle thought
        history_from_middle = storage.get_thought_history(thought2.id)
        assert len(history_from_middle) == 2, f"Expected 2 thoughts in history from middle, got {len(history_from_middle)}"
        assert history_from_middle[0].id == thought1.id, "Should include original thought"
        assert history_from_middle[1].id == thought2.id, "Should include middle thought"
        print("✓ History retrieval from middle thought works")
        
        # Test history retrieval from original thought
        history_from_original = storage.get_thought_history(thought1.id)
        assert len(history_from_original) == 1, f"Expected 1 thought in history from original, got {len(history_from_original)}"
        assert history_from_original[0].id == thought1.id, "Should only include original thought"
        print("✓ History retrieval from original thought works")


def test_chain_thoughts():
    """Test retrieving all thoughts in a chain."""
    print("\nTesting chain thoughts retrieval...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        storage = JSONThoughtStorage(storage_dir=temp_dir)
        
        chain_id = "test-chain"
        
        # Create multiple thoughts in the same chain
        thought1 = Thought(prompt="First prompt", text="First text", chain_id=chain_id)
        storage.save_thought(thought1)
        
        thought2 = Thought(prompt="Second prompt", text="Second text", chain_id=chain_id)
        storage.save_thought(thought2)
        
        # Create an iteration of the first thought
        thought1_v2 = thought1.next_iteration()
        thought1_v2 = thought1_v2.set_text("Updated first text")
        storage.save_thought(thought1_v2)
        
        # Create a thought in a different chain
        other_thought = Thought(prompt="Other prompt", text="Other text", chain_id="other-chain")
        storage.save_thought(other_thought)
        
        # Retrieve all thoughts in the test chain
        chain_thoughts = storage.get_chain_thoughts(chain_id)
        assert len(chain_thoughts) == 3, f"Expected 3 thoughts in chain, got {len(chain_thoughts)}"
        
        # Verify all thoughts belong to the correct chain
        for thought in chain_thoughts:
            assert thought.chain_id == chain_id, f"Thought {thought.id} has wrong chain_id: {thought.chain_id}"
        
        print("✓ Chain thoughts retrieved correctly")
        
        # Verify thoughts are sorted by timestamp (ascending)
        timestamps = [thought.timestamp for thought in chain_thoughts]
        assert timestamps == sorted(timestamps), "Chain thoughts should be sorted by timestamp"
        print("✓ Chain thoughts sorted by timestamp")


def main():
    """Run all history-related tests."""
    print("🧪 Testing Sifaka Thought History")
    print("=" * 40)
    
    try:
        test_thought_iterations()
        test_history_retrieval()
        test_chain_thoughts()
        
        print("\n" + "=" * 40)
        print("🎉 All history tests passed!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
