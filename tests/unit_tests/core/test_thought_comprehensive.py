"""Comprehensive unit tests for SifakaThought model.

This module provides extensive testing of the SifakaThought core state container:
- Thought creation and initialization
- State management and transitions
- Audit trail functionality
- Validation and critique tracking
- Conversation linking and history
- Serialization and persistence
- Edge cases and error handling

Tests cover:
- Basic thought operations
- State mutation and immutability
- Audit trail completeness
- Conversation management
- Performance characteristics
- Error scenarios
"""

import uuid
from datetime import datetime, timedelta


from sifaka.core.thought import (
    SifakaThought,
)


class TestSifakaThoughtCreation:
    """Test SifakaThought creation and initialization."""

    def test_thought_creation_minimal(self):
        """Test creating SifakaThought with minimal parameters."""
        thought = SifakaThought(prompt="Test prompt")

        # Verify basic properties
        assert thought.prompt == "Test prompt"
        assert thought.current_text is None
        assert thought.final_text is None
        assert thought.iteration == 0
        assert thought.max_iterations == 3

        # Verify ID generation
        assert thought.id is not None
        assert len(thought.id) > 0

        # Verify audit trail initialization
        assert thought.generations == []
        assert thought.validations == []
        assert thought.critiques == []
        assert thought.tool_calls == []

        # Verify timestamps
        assert isinstance(thought.created_at, datetime)
        assert isinstance(thought.updated_at, datetime)
        assert thought.created_at <= thought.updated_at

    def test_thought_creation_full_parameters(self):
        """Test creating SifakaThought with all parameters."""
        custom_id = str(uuid.uuid4())
        created_time = datetime.now() - timedelta(hours=1)

        thought = SifakaThought(
            id=custom_id,
            prompt="Full test prompt",
            current_text="Current text",
            final_text="Final text",
            iteration=2,
            max_iterations=5,
            created_at=created_time,
        )

        # Verify all parameters are set correctly
        assert thought.id == custom_id
        assert thought.prompt == "Full test prompt"
        assert thought.current_text == "Current text"
        assert thought.final_text == "Final text"
        assert thought.iteration == 2
        assert thought.max_iterations == 5
        assert thought.created_at == created_time

    def test_thought_creation_with_conversation_data(self):
        """Test creating SifakaThought with conversation linking."""
        parent_id = str(uuid.uuid4())
        conversation_id = str(uuid.uuid4())

        thought = SifakaThought(
            prompt="Follow-up prompt", parent_thought_id=parent_id, conversation_id=conversation_id
        )

        assert thought.parent_thought_id == parent_id
        assert thought.conversation_id == conversation_id

    def test_thought_id_uniqueness(self):
        """Test that each thought gets a unique ID."""
        thoughts = [SifakaThought(prompt=f"Prompt {i}") for i in range(100)]
        ids = [thought.id for thought in thoughts]

        # Verify all IDs are unique
        assert len(set(ids)) == 100

    def test_thought_creation_with_techniques(self):
        """Test creating SifakaThought with research techniques."""
        techniques = ["constitutional", "reflexion", "self-refine"]

        thought = SifakaThought(prompt="Research prompt", techniques_applied=techniques)

        assert thought.techniques_applied == techniques


class TestSifakaThoughtStateManagement:
    """Test SifakaThought state management and transitions."""

    def test_add_generation(self):
        """Test adding generation results to thought."""
        thought = SifakaThought(prompt="Test prompt")

        # Add a generation (using correct API)
        thought.add_generation(
            text="Generated text",
            model="openai:gpt-4",
            pydantic_result=None,  # Can be None for testing
        )

        # Verify generation was added
        assert len(thought.generations) == 1
        generation = thought.generations[0]
        assert generation.text == "Generated text"
        assert generation.model == "openai:gpt-4"
        assert generation.iteration == 0  # Should use current iteration
        assert isinstance(generation.conversation_history, list)

    def test_add_validation_result(self):
        """Test adding validation results to thought."""
        thought = SifakaThought(prompt="Test prompt")

        # Add validation result (using correct API)
        thought.add_validation(
            validator="length", passed=True, details={"word_count": 150, "min_length": 100}
        )

        # Verify validation was added
        assert len(thought.validations) == 1
        validation = thought.validations[0]
        assert validation.validator == "length"
        assert validation.passed is True
        assert validation.details["word_count"] == 150

    def test_add_critique_result(self):
        """Test adding critique results to thought."""
        thought = SifakaThought(prompt="Test prompt")

        # Add critique result (using correct API)
        thought.add_critique(
            critic="constitutional",
            feedback="Good structure but needs more detail",
            suggestions=["Add examples", "Expand conclusion"],
            needs_improvement=True,
        )

        # Verify critique was added
        assert len(thought.critiques) == 1
        critique = thought.critiques[0]
        assert critique.critic == "constitutional"
        assert critique.feedback == "Good structure but needs more detail"
        assert len(critique.suggestions) == 2
        assert critique.needs_improvement is True

    def test_add_tool_call(self):
        """Test adding tool call records to thought."""
        thought = SifakaThought(prompt="Test prompt")

        # Add tool call (using correct API)
        thought.add_tool_call(
            tool_name="web_search",
            args={"query": "renewable energy"},
            result={"results": ["Result 1", "Result 2"]},
            execution_time=1.5,
        )

        # Verify tool call was added
        assert len(thought.tool_calls) == 1
        tool_call = thought.tool_calls[0]
        assert tool_call.tool_name == "web_search"
        assert tool_call.args["query"] == "renewable energy"
        assert tool_call.execution_time == 1.5

    def test_iteration_management(self):
        """Test iteration management and state tracking."""
        thought = SifakaThought(prompt="Test prompt", max_iterations=3)

        # Initial state
        assert thought.iteration == 0
        assert thought.max_iterations == 3

        # Add some audit trail entries
        thought.add_validation("length", False, {"reason": "too short"})
        thought.add_critique("constitutional", "Needs work", ["Fix this"], needs_improvement=True)

        # Manually increment iteration (this would be done by the graph)
        thought.iteration += 1
        assert thought.iteration == 1

        # Add more entries for new iteration
        thought.add_validation("length", True, {"word_count": 200})
        thought.add_critique("constitutional", "Good work", [], needs_improvement=False)

        # Verify entries are tracked by iteration
        assert len(thought.validations) == 2
        assert len(thought.critiques) == 2

    def test_validation_passed_aggregation(self):
        """Test validation_passed() aggregation logic."""
        thought = SifakaThought(prompt="Test prompt")

        # No validations - should return False (per actual implementation)
        assert thought.validation_passed() is False

        # All validations pass
        thought.add_validation("length", True, {})
        thought.add_validation("coherence", True, {})
        assert thought.validation_passed() is True

        # Some validations fail
        thought.add_validation("factual", False, {"reason": "inaccurate"})
        assert thought.validation_passed() is False

    def test_critique_tracking(self):
        """Test critique tracking and analysis."""
        thought = SifakaThought(prompt="Test prompt")

        # Add critiques with different improvement needs
        thought.add_critique("constitutional", "Good", [], needs_improvement=False)
        thought.add_critique("reflexion", "Excellent", [], needs_improvement=False)
        thought.add_critique("self-refine", "Could be better", ["Fix this"], needs_improvement=True)

        # Verify critiques are tracked
        assert len(thought.critiques) == 3

        # Check individual critique properties
        improvement_needed = [c.needs_improvement for c in thought.critiques]
        assert improvement_needed == [False, False, True]


class TestSifakaThoughtAuditTrail:
    """Test SifakaThought audit trail functionality."""

    def test_complete_audit_trail(self):
        """Test that audit trail captures complete workflow."""
        thought = SifakaThought(prompt="Test prompt")

        # Simulate complete workflow
        # Generation
        thought.add_generation("First draft", "gpt-4", None)

        # Validation
        thought.add_validation("length", True, {"word_count": 150})
        thought.add_validation("coherence", False, {"score": 0.6})

        # Critique
        thought.add_critique(
            "constitutional", "Needs improvement", ["Add examples"], needs_improvement=True
        )

        # Tool usage
        thought.add_tool_call("web_search", {"query": "examples"}, {"results": []}, 0.5)

        # Second iteration
        thought.iteration = 1
        thought.add_generation("Improved draft", "gpt-4", None)
        thought.add_validation("coherence", True, {"score": 0.8})
        thought.add_critique("constitutional", "Much better", [], needs_improvement=False)

        # Verify complete audit trail
        assert len(thought.generations) == 2
        assert len(thought.validations) == 3  # length + 2 coherence
        assert len(thought.critiques) == 2
        assert len(thought.tool_calls) == 1

        # Verify iteration tracking
        assert thought.generations[0].iteration == 0
        assert thought.generations[1].iteration == 1

    def test_audit_trail_consistency(self):
        """Test that audit trail maintains consistency."""
        thought = SifakaThought(prompt="Test prompt")

        # Add generation
        thought.add_generation("Original text", "gpt-4", None)
        original_generation = thought.generations[0]
        original_text = original_generation.text

        # Verify generation was added correctly
        assert original_text == "Original text"
        assert len(thought.generations) == 1

        # Add another generation
        thought.add_generation("Second text", "gpt-4", None)

        # Verify both generations exist
        assert len(thought.generations) == 2
        assert thought.generations[0].text == "Original text"
        assert thought.generations[1].text == "Second text"

        # Verify generations maintain their order
        assert thought.generations[0].iteration == 0
        assert thought.generations[1].iteration == 0  # Same iteration

    def test_audit_trail_ordering(self):
        """Test that audit trail maintains chronological ordering."""
        thought = SifakaThought(prompt="Test prompt")

        # Add entries with slight delays to ensure different timestamps
        import time

        thought.add_generation("First", "gpt-4", None)
        time.sleep(0.001)
        thought.add_validation("length", True, {})
        time.sleep(0.001)
        thought.add_critique("constitutional", "Good", [], needs_improvement=False)
        time.sleep(0.001)
        thought.add_tool_call("search", {}, {}, 0.1)

        # Verify timestamps are in order
        timestamps = [
            thought.generations[0].timestamp,
            thought.validations[0].timestamp,
            thought.critiques[0].timestamp,
            thought.tool_calls[0].timestamp,
        ]

        # Each timestamp should be >= the previous one
        for i in range(1, len(timestamps)):
            assert timestamps[i] >= timestamps[i - 1]


class TestSifakaThoughtConversationManagement:
    """Test SifakaThought conversation linking and history management."""

    def test_conversation_creation(self):
        """Test creating a new conversation."""
        thought = SifakaThought(prompt="Start conversation")

        # First thought in conversation should use its own ID as conversation_id
        assert thought.conversation_id is None or thought.conversation_id == thought.id
        assert thought.parent_thought_id is None

    def test_conversation_continuation(self):
        """Test continuing an existing conversation."""
        # Create parent thought
        parent = SifakaThought(prompt="Parent prompt")
        parent_id = parent.id

        # Create follow-up thought
        follow_up = SifakaThought(
            prompt="Follow-up prompt", parent_thought_id=parent_id, conversation_id=parent_id
        )

        assert follow_up.parent_thought_id == parent_id
        assert follow_up.conversation_id == parent_id

    def test_conversation_chain(self):
        """Test a chain of conversation thoughts."""
        # Create conversation chain
        thought1 = SifakaThought(prompt="First thought")
        conv_id = thought1.id

        thought2 = SifakaThought(
            prompt="Second thought", parent_thought_id=thought1.id, conversation_id=conv_id
        )

        thought3 = SifakaThought(
            prompt="Third thought", parent_thought_id=thought2.id, conversation_id=conv_id
        )

        # Verify chain structure
        assert thought1.parent_thought_id is None
        assert thought2.parent_thought_id == thought1.id
        assert thought3.parent_thought_id == thought2.id

        # All should share same conversation_id
        assert thought2.conversation_id == conv_id
        assert thought3.conversation_id == conv_id

    def test_conversation_branching(self):
        """Test branching conversations from a single parent."""
        parent = SifakaThought(prompt="Parent thought")
        conv_id = parent.id

        # Create two branches
        branch1 = SifakaThought(
            prompt="Branch 1", parent_thought_id=parent.id, conversation_id=conv_id
        )

        branch2 = SifakaThought(
            prompt="Branch 2", parent_thought_id=parent.id, conversation_id=conv_id
        )

        # Both branches should reference same parent
        assert branch1.parent_thought_id == parent.id
        assert branch2.parent_thought_id == parent.id
        assert branch1.conversation_id == conv_id
        assert branch2.conversation_id == conv_id

        # But should have different IDs
        assert branch1.id != branch2.id


class TestSifakaThoughtEdgeCases:
    """Test SifakaThought edge cases and error scenarios."""

    def test_empty_prompt(self):
        """Test thought creation with empty prompt."""
        thought = SifakaThought(prompt="")
        assert thought.prompt == ""

    def test_very_long_prompt(self):
        """Test thought with extremely long prompt."""
        long_prompt = "A" * 100000  # 100k characters
        thought = SifakaThought(prompt=long_prompt)
        assert len(thought.prompt) == 100000

    def test_unicode_prompt(self):
        """Test thought with Unicode characters."""
        unicode_prompt = "Hello 世界 🌍 Здравствуй мир 🚀"
        thought = SifakaThought(prompt=unicode_prompt)
        assert thought.prompt == unicode_prompt

    def test_special_characters_prompt(self):
        """Test thought with special characters."""
        special_prompt = 'Test\n\t"quotes" & <tags> {json: true} $variables'
        thought = SifakaThought(prompt=special_prompt)
        assert thought.prompt == special_prompt

    def test_zero_max_iterations(self):
        """Test thought with zero max iterations."""
        thought = SifakaThought(prompt="Test", max_iterations=0)
        assert thought.max_iterations == 0
        # Note: should_continue() method doesn't exist in actual implementation

    def test_negative_iteration(self):
        """Test thought with negative iteration (should be handled gracefully)."""
        thought = SifakaThought(prompt="Test", iteration=-1)
        # Should still work, though logically unusual
        assert thought.iteration == -1

    def test_massive_audit_trail(self):
        """Test thought with very large audit trail."""
        thought = SifakaThought(prompt="Test")

        # Add many entries to audit trail
        for i in range(1000):
            thought.add_generation(f"Generation {i}", "gpt-4", None)
            thought.add_validation(f"validator_{i}", i % 2 == 0, {"iteration": i})
            thought.add_critique(f"critic_{i}", f"Feedback {i}", [], needs_improvement=(i % 3 == 0))
            thought.add_tool_call(f"tool_{i}", {"input": i}, {"output": i}, 0.1)

        # Verify all entries were added
        assert len(thought.generations) == 1000
        assert len(thought.validations) == 1000
        assert len(thought.critiques) == 1000
        assert len(thought.tool_calls) == 1000

    def test_concurrent_modifications(self):
        """Test concurrent modifications to thought (thread safety)."""
        import threading
        import time

        thought = SifakaThought(prompt="Concurrent test")
        results = []

        def add_entries(thread_id):
            for i in range(100):
                thought.add_generation(f"Thread {thread_id} Gen {i}", "gpt-4", None)
                time.sleep(0.001)  # Small delay to encourage race conditions
            results.append(thread_id)

        # Start multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=add_entries, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Verify all entries were added (500 total)
        assert len(thought.generations) == 500
        assert len(results) == 5

    def test_memory_efficiency(self):
        """Test memory efficiency with large thoughts."""
        import sys

        thought = SifakaThought(prompt="Memory test")

        # Get initial memory usage
        initial_size = sys.getsizeof(thought)

        # Add moderate amount of data
        for i in range(100):
            thought.add_generation(f"Gen {i}", "gpt-4", None)

        # Memory should scale reasonably
        final_size = sys.getsizeof(thought)
        growth_ratio = final_size / initial_size

        # Should not grow excessively (less than 100x for 100 entries)
        assert growth_ratio < 100


class TestSifakaThoughtSerialization:
    """Test SifakaThought serialization and persistence."""

    def test_json_serialization(self):
        """Test JSON serialization of thought."""
        thought = SifakaThought(prompt="Serialization test")
        thought.add_generation("Generated text", "gpt-4", None)
        thought.add_validation("length", True, {"word_count": 50})

        # Convert to dict (JSON-serializable)
        thought_dict = thought.model_dump()

        # Verify structure
        assert thought_dict["prompt"] == "Serialization test"
        assert len(thought_dict["generations"]) == 1
        assert len(thought_dict["validations"]) == 1
        assert "id" in thought_dict
        assert "created_at" in thought_dict

    def test_json_deserialization(self):
        """Test JSON deserialization of thought."""
        # Create original thought
        original = SifakaThought(prompt="Deserialization test")
        original.add_generation("Generated text", "gpt-4", None)

        # Serialize and deserialize
        thought_dict = original.model_dump()
        restored = SifakaThought.model_validate(thought_dict)

        # Verify restoration
        assert restored.prompt == original.prompt
        assert restored.id == original.id
        assert len(restored.generations) == len(original.generations)
        assert restored.generations[0].text == original.generations[0].text

    def test_partial_serialization(self):
        """Test serialization with exclude/include options."""
        thought = SifakaThought(prompt="Partial test")
        thought.add_generation("Text", "gpt-4", None)

        # Serialize excluding audit trail
        minimal_dict = thought.model_dump(
            exclude={"generations", "validations", "critiques", "tool_calls"}
        )

        # Should only have basic fields
        assert "prompt" in minimal_dict
        assert "id" in minimal_dict
        assert "generations" not in minimal_dict
        assert "validations" not in minimal_dict
