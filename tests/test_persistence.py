#!/usr/bin/env python3
"""
Comprehensive test suite for Sifaka JSON persistence implementation.

This module tests all aspects of the JSON persistence layer including:
- Basic save/retrieve operations
- Thought history and iterations
- Query functionality with filters
- Health checks and maintenance
- Error handling and edge cases
"""

import tempfile
import shutil
from datetime import datetime, timedelta
from pathlib import Path

from sifaka.core.thought import Thought, Document, ValidationResult, CriticFeedback
from sifaka.persistence.json import JSONThoughtStorage
from sifaka.persistence.base import ThoughtQuery


def test_basic_persistence():
    """Test basic save and retrieve operations."""
    print("Testing basic persistence...")
    
    # Create temporary storage directory
    with tempfile.TemporaryDirectory() as temp_dir:
        storage = JSONThoughtStorage(storage_dir=temp_dir)
        
        # Create a test thought
        thought = Thought(
            prompt="Write a short story about a robot learning to paint.",
            text="Once upon a time, there was a robot named Artie who discovered the joy of painting...",
            system_prompt="You are a creative storyteller.",
            chain_id="test-chain-1"
        )
        
        # Add some context
        thought = thought.add_pre_generation_context([
            Document(text="Robots are mechanical beings capable of learning.", score=0.9),
            Document(text="Painting is a form of artistic expression.", score=0.8)
        ])
        
        # Add validation results
        validation_result = ValidationResult(
            passed=True,
            message="Text passes all validation checks",
            score=0.95,
            issues=[],
            suggestions=[]
        )
        thought = thought.add_validation_result("length_validator", validation_result)
        
        # Add critic feedback
        critic_feedback = CriticFeedback(
            critic_name="creativity_critic",
            issues=["Could use more vivid descriptions"],
            suggestions=["Add more sensory details", "Describe the robot's emotions"],
            metadata={"confidence": 0.8}
        )
        thought = thought.add_critic_feedback(critic_feedback)
        
        print(f"Created thought with ID: {thought.id}")
        
        # Save the thought
        storage.save_thought(thought)
        print("✓ Thought saved successfully")
        
        # Retrieve the thought
        retrieved_thought = storage.get_thought(thought.id)
        assert retrieved_thought is not None, "Failed to retrieve thought"
        assert retrieved_thought.id == thought.id, "ID mismatch"
        assert retrieved_thought.prompt == thought.prompt, "Prompt mismatch"
        assert retrieved_thought.text == thought.text, "Text mismatch"
        print("✓ Thought retrieved successfully")
        
        # Check context preservation
        assert len(retrieved_thought.pre_generation_context) == 2, "Context not preserved"
        assert retrieved_thought.pre_generation_context[0].text == "Robots are mechanical beings capable of learning."
        print("✓ Context preserved correctly")
        
        # Check validation results preservation
        assert "length_validator" in retrieved_thought.validation_results, "Validation results not preserved"
        assert retrieved_thought.validation_results["length_validator"].passed == True
        print("✓ Validation results preserved correctly")
        
        # Check critic feedback preservation
        assert len(retrieved_thought.critic_feedback) == 1, "Critic feedback not preserved"
        assert retrieved_thought.critic_feedback[0].critic_name == "creativity_critic"
        print("✓ Critic feedback preserved correctly")


def test_thought_history():
    """Test thought history and iteration tracking."""
    print("\nTesting thought history...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        storage = JSONThoughtStorage(storage_dir=temp_dir)
        
        # Create initial thought
        thought1 = Thought(
            prompt="Explain quantum computing",
            text="Quantum computing is a type of computation...",
            chain_id="quantum-chain"
        )
        storage.save_thought(thought1)
        
        # Create next iteration
        thought2 = thought1.next_iteration()
        thought2 = thought2.set_text("Quantum computing is a revolutionary approach to computation that leverages quantum mechanics...")
        storage.save_thought(thought2)
        
        # Create third iteration
        thought3 = thought2.next_iteration()
        thought3 = thought3.set_text("Quantum computing represents a paradigm shift in computational technology...")
        storage.save_thought(thought3)
        
        print(f"Created thought chain: {thought1.id} -> {thought2.id} -> {thought3.id}")
        
        # Test history retrieval
        history = storage.get_thought_history(thought3.id)
        assert len(history) == 3, f"Expected 3 thoughts in history, got {len(history)}"
        assert history[0].id == thought1.id, "First thought in history incorrect"
        assert history[1].id == thought2.id, "Second thought in history incorrect"
        assert history[2].id == thought3.id, "Third thought in history incorrect"
        print("✓ Thought history retrieved correctly")
        
        # Test chain retrieval
        chain_thoughts = storage.get_chain_thoughts("quantum-chain")
        assert len(chain_thoughts) == 3, f"Expected 3 thoughts in chain, got {len(chain_thoughts)}"
        print("✓ Chain thoughts retrieved correctly")


def test_querying():
    """Test thought querying capabilities."""
    print("\nTesting querying...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        storage = JSONThoughtStorage(storage_dir=temp_dir)
        
        # Create multiple thoughts with different characteristics
        thoughts = []
        
        # Thought 1: AI story
        thought1 = Thought(
            prompt="Write about AI",
            text="Artificial intelligence is transforming our world...",
            chain_id="ai-chain"
        )
        thoughts.append(thought1)
        
        # Thought 2: Robot story
        thought2 = Thought(
            prompt="Write about robots",
            text="Robots are becoming increasingly sophisticated...",
            chain_id="robot-chain"
        )
        thoughts.append(thought2)
        
        # Thought 3: Future story (with validation results)
        thought3 = Thought(
            prompt="Describe the future",
            text="The future holds many exciting possibilities...",
            chain_id="future-chain"
        )
        validation_result = ValidationResult(passed=True, message="Good content")
        thought3 = thought3.add_validation_result("content_validator", validation_result)
        thoughts.append(thought3)
        
        # Save all thoughts
        for thought in thoughts:
            storage.save_thought(thought)
        
        print(f"Created {len(thoughts)} test thoughts")
        
        # Test basic query (all thoughts)
        all_thoughts = storage.query_thoughts()
        assert all_thoughts.total_count == 3, f"Expected 3 thoughts, got {all_thoughts.total_count}"
        print("✓ Basic query works")
        
        # Test text content filter
        ai_query = ThoughtQuery(text_contains="intelligence")
        ai_results = storage.query_thoughts(ai_query)
        assert ai_results.total_count == 1, f"Expected 1 AI thought, got {ai_results.total_count}"
        assert "intelligence" in ai_results.thoughts[0].text
        print("✓ Text content filtering works")
        
        # Test chain ID filter
        chain_query = ThoughtQuery(chain_ids=["ai-chain", "robot-chain"])
        chain_results = storage.query_thoughts(chain_query)
        assert chain_results.total_count == 2, f"Expected 2 chain thoughts, got {chain_results.total_count}"
        print("✓ Chain ID filtering works")
        
        # Test validation results filter
        validation_query = ThoughtQuery(has_validation_results=True)
        validation_results = storage.query_thoughts(validation_query)
        assert validation_results.total_count == 1, f"Expected 1 validated thought, got {validation_results.total_count}"
        print("✓ Validation results filtering works")
        
        # Test pagination
        paginated_query = ThoughtQuery(limit=2, offset=0)
        page1 = storage.query_thoughts(paginated_query)
        assert len(page1.thoughts) == 2, f"Expected 2 thoughts in page 1, got {len(page1.thoughts)}"
        
        paginated_query = ThoughtQuery(limit=2, offset=2)
        page2 = storage.query_thoughts(paginated_query)
        assert len(page2.thoughts) == 1, f"Expected 1 thought in page 2, got {len(page2.thoughts)}"
        print("✓ Pagination works")


def test_health_check():
    """Test storage health check."""
    print("\nTesting health check...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        storage = JSONThoughtStorage(storage_dir=temp_dir)
        
        # Create a test thought
        thought = Thought(prompt="Test health", text="This is a test")
        storage.save_thought(thought)
        
        # Run health check
        health = storage.health_check()
        
        assert health["status"] == "healthy", f"Expected healthy status, got {health['status']}"
        assert health["total_thoughts"] == 1, f"Expected 1 thought, got {health['total_thoughts']}"
        assert health["directories_exist"] == True, "Directories should exist"
        assert health["writable"] == True, "Storage should be writable"
        
        print("✓ Health check passed")
        print(f"  - Status: {health['status']}")
        print(f"  - Total thoughts: {health['total_thoughts']}")
        print(f"  - Storage size: {health['total_size_mb']} MB")


def test_error_handling():
    """Test error handling and edge cases."""
    print("\nTesting error handling...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        storage = JSONThoughtStorage(storage_dir=temp_dir)
        
        # Test retrieving non-existent thought
        non_existent = storage.get_thought("non-existent-id")
        assert non_existent is None, "Should return None for non-existent thought"
        print("✓ Non-existent thought handling works")
        
        # Test deleting non-existent thought
        deleted = storage.delete_thought("non-existent-id")
        assert deleted == False, "Should return False for non-existent thought deletion"
        print("✓ Non-existent thought deletion handling works")
        
        # Test empty query
        empty_results = storage.query_thoughts()
        assert empty_results.total_count == 0, "Should return 0 for empty storage"
        print("✓ Empty storage query works")


def main():
    """Run all tests."""
    print("🧪 Testing Sifaka JSON Persistence Implementation")
    print("=" * 50)
    
    try:
        test_basic_persistence()
        test_thought_history()
        test_querying()
        test_health_check()
        test_error_handling()
        
        print("\n" + "=" * 50)
        print("🎉 All tests passed! JSON persistence is working correctly.")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
