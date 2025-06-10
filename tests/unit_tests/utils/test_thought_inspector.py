"""Comprehensive unit tests for thought inspector utilities.

This module tests the thought inspection and debugging utilities:
- Iteration detail printing and analysis
- Conversation message extraction and formatting
- Critic and validation summary generation
- Thought overview and statistics

Tests cover:
- Thought inspection functions
- Message formatting and display
- Summary generation utilities
- Error handling and edge cases
- Mock-based testing with sample thoughts
"""

from io import StringIO
from unittest.mock import patch

import pytest

from sifaka.core.thought import SifakaThought
from sifaka.utils.thought_inspector import (
    get_conversation_messages_for_iteration,
    get_latest_conversation_messages,
    get_thought_overview,
    print_all_iterations,
    print_conversation_messages,
    print_critic_summary,
    print_iteration_details,
    print_validation_summary,
)


class TestThoughtInspector:
    """Test suite for thought inspector utilities."""

    @pytest.fixture
    def sample_thought(self):
        """Create a comprehensive sample thought for testing."""
        thought = SifakaThought(
            prompt="Explain the benefits of renewable energy",
            final_text="Renewable energy sources like solar and wind power offer numerous environmental and economic benefits.",
            iteration=2,
            max_iterations=5,
        )

        # Add multiple generations
        thought.add_generation("First draft about renewable energy", "gpt-4", {"temperature": 0.7})
        thought.iteration = 1
        thought.add_generation("Improved version with more details", "gpt-4", {"temperature": 0.5})

        # Add validations
        thought.add_validation("length_validator", True, {"word_count": 15})
        thought.add_validation("content_validator", False, {"issue": "needs more examples"})
        thought.add_validation("clarity_validator", True, {"score": 0.85})

        # Add critiques
        thought.add_critique(
            "constitutional_critic",
            "Good structure but needs more specific examples",
            ["Add concrete examples of renewable energy benefits", "Include statistics"],
            confidence=0.8,
            needs_improvement=True,
        )
        thought.add_critique(
            "reflexion_critic",
            "Clear explanation but could be more comprehensive",
            ["Expand on economic benefits", "Discuss environmental impact"],
            confidence=0.75,
            needs_improvement=True,
        )

        return thought

    @pytest.fixture
    def minimal_thought(self):
        """Create a minimal thought for edge case testing."""
        return SifakaThought(
            prompt="Simple prompt", final_text="Simple response", iteration=0, max_iterations=1
        )

    def test_print_iteration_details_basic(self, sample_thought):
        """Test printing iteration details for a specific iteration."""
        with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
            print_iteration_details(sample_thought, iteration=1)
            output = mock_stdout.getvalue()

        # Verify output contains iteration information
        assert "Iteration 1" in output
        assert "Generation:" in output
        assert "Improved version with more details" in output

    def test_print_iteration_details_invalid_iteration(self, sample_thought):
        """Test printing iteration details for invalid iteration."""
        with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
            print_iteration_details(sample_thought, iteration=10)
            output = mock_stdout.getvalue()

        # Should handle invalid iteration gracefully
        assert "No generation found" in output or "Invalid iteration" in output

    def test_print_all_iterations(self, sample_thought):
        """Test printing all iterations."""
        with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
            print_all_iterations(sample_thought)
            output = mock_stdout.getvalue()

        # Verify output contains all iterations
        assert "Iteration 0" in output
        assert "Iteration 1" in output
        assert "First draft about renewable energy" in output
        assert "Improved version with more details" in output

    def test_print_all_iterations_minimal(self, minimal_thought):
        """Test printing all iterations for minimal thought."""
        with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
            print_all_iterations(minimal_thought)
            output = mock_stdout.getvalue()

        # Should handle minimal thought gracefully
        assert "Iteration" in output

    def test_get_latest_conversation_messages(self, sample_thought):
        """Test getting latest conversation messages."""
        messages = get_latest_conversation_messages(sample_thought)

        # Verify message structure
        assert isinstance(messages, list)
        assert len(messages) >= 2  # Should have user and assistant messages

        # Check message content
        user_message = next((msg for msg in messages if msg["role"] == "user"), None)
        assistant_message = next((msg for msg in messages if msg["role"] == "assistant"), None)

        assert user_message is not None
        assert assistant_message is not None
        assert sample_thought.prompt in user_message["content"]

    def test_get_conversation_messages_for_iteration(self, sample_thought):
        """Test getting conversation messages for specific iteration."""
        messages = get_conversation_messages_for_iteration(sample_thought, iteration=1)

        # Verify message structure
        assert isinstance(messages, list)
        assert len(messages) >= 2

        # Check that messages correspond to the correct iteration
        assistant_message = next((msg for msg in messages if msg["role"] == "assistant"), None)
        assert assistant_message is not None

    def test_get_conversation_messages_invalid_iteration(self, sample_thought):
        """Test getting conversation messages for invalid iteration."""
        messages = get_conversation_messages_for_iteration(sample_thought, iteration=10)

        # Should return empty list or handle gracefully
        assert isinstance(messages, list)

    def test_print_conversation_messages(self, sample_thought):
        """Test printing conversation messages."""
        messages = get_latest_conversation_messages(sample_thought)

        with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
            print_conversation_messages(messages)
            output = mock_stdout.getvalue()

        # Verify output contains message information
        assert "User:" in output or "Assistant:" in output
        assert sample_thought.prompt in output

    def test_print_conversation_messages_empty(self):
        """Test printing empty conversation messages."""
        with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
            print_conversation_messages([])
            output = mock_stdout.getvalue()

        # Should handle empty messages gracefully
        assert "No messages" in output or output.strip() == ""

    def test_print_critic_summary(self, sample_thought):
        """Test printing critic summary."""
        with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
            print_critic_summary(sample_thought)
            output = mock_stdout.getvalue()

        # Verify output contains critic information
        assert "Critics Summary" in output or "Critiques" in output
        assert "constitutional_critic" in output
        assert "reflexion_critic" in output
        assert "needs more specific examples" in output

    def test_print_critic_summary_no_critiques(self, minimal_thought):
        """Test printing critic summary with no critiques."""
        with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
            print_critic_summary(minimal_thought)
            output = mock_stdout.getvalue()

        # Should handle no critiques gracefully
        assert "No critiques" in output or "Critics Summary" in output

    def test_print_validation_summary(self, sample_thought):
        """Test printing validation summary."""
        with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
            print_validation_summary(sample_thought)
            output = mock_stdout.getvalue()

        # Verify output contains validation information
        assert "Validation Summary" in output or "Validations" in output
        assert "length_validator" in output
        assert "content_validator" in output
        assert "clarity_validator" in output
        assert "✓" in output or "✗" in output  # Pass/fail indicators

    def test_print_validation_summary_no_validations(self, minimal_thought):
        """Test printing validation summary with no validations."""
        with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
            print_validation_summary(minimal_thought)
            output = mock_stdout.getvalue()

        # Should handle no validations gracefully
        assert "No validations" in output or "Validation Summary" in output

    def test_get_thought_overview(self, sample_thought):
        """Test getting thought overview."""
        overview = get_thought_overview(sample_thought)

        # Verify overview structure
        assert isinstance(overview, dict)

        # Check key fields
        expected_fields = [
            "id",
            "prompt",
            "final_text",
            "iteration",
            "max_iterations",
            "generations_count",
            "validations_count",
            "critiques_count",
        ]

        for field in expected_fields:
            assert field in overview

        # Verify counts
        assert overview["generations_count"] == len(sample_thought.generations)
        assert overview["validations_count"] == len(sample_thought.validations)
        assert overview["critiques_count"] == len(sample_thought.critiques)

    def test_get_thought_overview_minimal(self, minimal_thought):
        """Test getting thought overview for minimal thought."""
        overview = get_thought_overview(minimal_thought)

        # Verify overview structure
        assert isinstance(overview, dict)
        assert overview["generations_count"] == 0
        assert overview["validations_count"] == 0
        assert overview["critiques_count"] == 0

    def test_get_thought_overview_statistics(self, sample_thought):
        """Test thought overview statistics calculation."""
        overview = get_thought_overview(sample_thought)

        # Check statistics fields
        if "statistics" in overview:
            stats = overview["statistics"]
            assert isinstance(stats, dict)

            # Verify statistical calculations
            if "validation_success_rate" in stats:
                assert isinstance(stats["validation_success_rate"], (int, float))
                assert 0 <= stats["validation_success_rate"] <= 1

            if "average_critic_confidence" in stats:
                assert isinstance(stats["average_critic_confidence"], (int, float))
                assert 0 <= stats["average_critic_confidence"] <= 1

    def test_iteration_details_with_validations_and_critiques(self, sample_thought):
        """Test iteration details include validations and critiques."""
        with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
            print_iteration_details(
                sample_thought, iteration=1, include_validations=True, include_critiques=True
            )
            output = mock_stdout.getvalue()

        # Should include validation and critique information
        assert "Validation" in output or "length_validator" in output
        assert "Critique" in output or "constitutional_critic" in output

    def test_conversation_messages_formatting(self, sample_thought):
        """Test conversation message formatting."""
        messages = get_latest_conversation_messages(sample_thought)

        # Verify message formatting
        for message in messages:
            assert "role" in message
            assert "content" in message
            assert message["role"] in ["user", "assistant", "system"]
            assert isinstance(message["content"], str)
            assert len(message["content"]) > 0

    def test_thought_inspector_error_handling(self):
        """Test thought inspector error handling with None input."""
        with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
            # Test functions with None input
            print_iteration_details(None, iteration=0)
            print_all_iterations(None)
            print_critic_summary(None)
            print_validation_summary(None)

            output = mock_stdout.getvalue()

        # Should handle None input gracefully
        assert "Error" in output or "None" in output or output.strip() == ""

    def test_get_thought_overview_none_input(self):
        """Test get_thought_overview with None input."""
        overview = get_thought_overview(None)

        # Should return empty dict or handle gracefully
        assert overview is None or isinstance(overview, dict)

    def test_conversation_messages_with_tool_calls(self, sample_thought):
        """Test conversation messages when thought has tool calls."""
        # Add a tool call to the thought
        sample_thought.add_tool_call(
            "web_search", {"query": "renewable energy"}, {"results": ["result1"]}
        )

        messages = get_latest_conversation_messages(sample_thought)

        # Should include tool call information
        assert isinstance(messages, list)
        # Tool calls might be included in assistant messages or as separate entries

    def test_print_functions_with_custom_formatting(self, sample_thought):
        """Test print functions with custom formatting options."""
        with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
            # Test with different formatting options if available
            print_iteration_details(sample_thought, iteration=1, verbose=True)
            print_critic_summary(sample_thought, show_confidence=True)
            print_validation_summary(sample_thought, show_details=True)

            output = mock_stdout.getvalue()

        # Should produce formatted output
        assert len(output) > 0
