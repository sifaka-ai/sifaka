"""
Tests for the critics module.

This module contains tests for the critics in the Sifaka framework.
"""

from typing import Any, Dict

import pytest

from sifaka.critics.base import Critic
from sifaka.critics.n_critics import NCriticsCritic, create_n_critics_critic
from sifaka.critics.reflexion import ReflexionCritic, create_reflexion_critic
from sifaka.errors import ImproverError


class TestCriticBase:
    """Tests for the Critic base class."""

    def test_init_with_defaults(self, mock_model) -> None:
        """Test initializing a Critic with default parameters."""

        class TestCritic(Critic):
            def _critique(self, text: str) -> Dict[str, Any]:
                return {"needs_improvement": True, "message": "Test critique"}

            def _improve(self, text: str, critique: Dict[str, Any]) -> str:
                return "Improved: " + text

        critic = TestCritic(model=mock_model)
        assert critic.model is mock_model
        assert critic._name == "TestCritic"
        assert critic.options == {}

    def test_init_with_name(self, mock_model) -> None:
        """Test initializing a Critic with a custom name."""

        class TestCritic(Critic):
            def _critique(self, text: str) -> Dict[str, Any]:
                return {"needs_improvement": True, "message": "Test critique"}

            def _improve(self, text: str, critique: Dict[str, Any]) -> str:
                return "Improved: " + text

        critic = TestCritic(model=mock_model, name="CustomName")
        assert critic._name == "CustomName"

    def test_init_with_options(self, mock_model) -> None:
        """Test initializing a Critic with options."""

        class TestCritic(Critic):
            def _critique(self, text: str) -> Dict[str, Any]:
                return {"needs_improvement": True, "message": "Test critique"}

            def _improve(self, text: str, critique: Dict[str, Any]) -> str:
                return "Improved: " + text

        critic = TestCritic(model=mock_model, option1="value1", option2="value2")
        assert critic.options == {"option1": "value1", "option2": "value2"}

    def test_improve_with_needs_improvement(self, mock_model) -> None:
        """Test improving text that needs improvement."""

        class TestCritic(Critic):
            def _critique(self, text: str) -> Dict[str, Any]:
                return {"needs_improvement": True, "message": "Test critique"}

            def _improve(self, text: str, critique: Dict[str, Any]) -> str:
                return "Improved: " + text

        critic = TestCritic(model=mock_model)
        improved_text, result = critic.improve("Test text")

        assert improved_text == "Improved: Test text"
        assert result.original_text == "Test text"
        assert result.improved_text == "Improved: Test text"
        assert result.changes_made is True
        assert "test critique" in result.message.lower()

    def test_improve_with_no_improvement_needed(self, mock_model) -> None:
        """Test improving text that doesn't need improvement."""

        class TestCritic(Critic):
            def _critique(self, text: str) -> Dict[str, Any]:
                return {"needs_improvement": False, "message": "No improvement needed"}

            def _improve(self, text: str, critique: Dict[str, Any]) -> str:
                return "This should not be called"

        critic = TestCritic(model=mock_model)
        improved_text, result = critic.improve("Test text")

        assert improved_text == "Test text"  # Should be unchanged
        assert result.original_text == "Test text"
        assert result.improved_text == "Test text"
        assert result.changes_made is False
        assert "no improvement needed" in result.message.lower()

    def test_improve_with_empty_text(self, mock_model) -> None:
        """Test improving empty text."""

        class TestCritic(Critic):
            def _critique(self, text: str) -> Dict[str, Any]:
                return {"needs_improvement": True, "message": "Test critique"}

            def _improve(self, text: str, critique: Dict[str, Any]) -> str:
                return "Improved: " + text

        critic = TestCritic(model=mock_model)
        improved_text, result = critic.improve("")

        assert improved_text == ""  # Should be unchanged
        assert result.original_text == ""
        assert result.improved_text == ""
        assert result.changes_made is False
        assert "empty" in result.message.lower()

    def test_improve_handles_exceptions_in_critique(self, mock_model) -> None:
        """Test that improve handles exceptions in _critique."""

        class TestCritic(Critic):
            def _critique(self, text: str) -> Dict[str, Any]:
                raise ValueError("Test error in critique")

            def _improve(self, text: str, critique: Dict[str, Any]) -> str:
                return "This should not be called"

        critic = TestCritic(model=mock_model)

        # The error should be caught and re-raised as ImproverError
        with pytest.raises(ImproverError) as excinfo:
            critic.improve("Test text")

        # Check that the error message contains the original error
        assert "Test error in critique" in str(excinfo.value)

    def test_improve_handles_exceptions_in_improve(self, mock_model) -> None:
        """Test that improve handles exceptions in _improve."""

        class TestCritic(Critic):
            def _critique(self, text: str) -> Dict[str, Any]:
                return {"needs_improvement": True, "message": "Test critique"}

            def _improve(self, text: str, critique: Dict[str, Any]) -> str:
                raise ValueError("Test error in improve")

        critic = TestCritic(model=mock_model)

        # The error should be caught and re-raised as ImproverError
        with pytest.raises(ImproverError) as excinfo:
            critic.improve("Test text")

        # Check that the error message contains the original error
        assert "Test error in improve" in str(excinfo.value)


class TestReflexionCritic:
    """Tests for the ReflexionCritic."""

    def test_init_with_defaults(self, mock_model) -> None:
        """Test initializing a ReflexionCritic with default parameters."""
        critic = ReflexionCritic(model=mock_model)
        assert critic.model is mock_model
        assert critic._name == "ReflexionCritic"
        assert critic.options == {}

    def test_init_with_options(self, mock_model) -> None:
        """Test initializing a ReflexionCritic with options."""
        critic = ReflexionCritic(model=mock_model, temperature=0.5)
        assert critic.temperature == 0.5

    def test_create_reflexion_critic(self, mock_model) -> None:
        """Test creating a ReflexionCritic using the factory function."""
        critic = create_reflexion_critic(model=mock_model)
        assert isinstance(critic, ReflexionCritic)
        assert critic.model is mock_model

    def test_create_reflexion_critic_with_options(self, mock_model) -> None:
        """Test creating a ReflexionCritic with options using the factory function."""
        critic = create_reflexion_critic(model=mock_model, temperature=0.5)
        assert isinstance(critic, ReflexionCritic)
        assert critic.temperature == 0.5

    def test_critique_method(self, mock_model) -> None:
        """Test the _critique method of ReflexionCritic."""
        # Set up the mock model to return a specific response with JSON
        mock_model.set_response(
            """
        Here's my analysis:

        {
            "needs_improvement": true,
            "message": "This text could be improved in several ways. It lacks clarity and detail.",
            "issues": [
                "The text is too vague.",
                "There are no specific examples.",
                "The structure is confusing."
            ],
            "suggestions": [
                "Add more specific details.",
                "Include concrete examples.",
                "Improve the structure with clear paragraphs."
            ],
            "reflections": [
                "The text doesn't provide enough context for the reader.",
                "Without specific examples, the points being made are unclear.",
                "A more structured approach would help guide the reader."
            ]
        }
        """
        )

        critic = ReflexionCritic(model=mock_model)
        critique = critic._critique("Test text")

        assert critique["needs_improvement"] is True
        assert "message" in critique
        assert "issues" in critique
        assert "suggestions" in critique
        assert "reflections" in critique
        assert len(critique["issues"]) == 3
        assert len(critique["suggestions"]) == 3
        assert len(critique["reflections"]) == 3
        assert len(mock_model.generate_calls) == 1

    def test_improve_method(self, mock_model) -> None:
        """Test the _improve method of ReflexionCritic."""
        # Set up the mock model to return a specific response
        mock_model.set_response("This is the improved text.")

        critic = ReflexionCritic(model=mock_model)
        critique = {
            "needs_improvement": True,
            "reflection": "This text could be improved.",
            "issues": ["Issue 1", "Issue 2"],
            "suggestions": ["Suggestion 1", "Suggestion 2"],
        }

        improved_text = critic._improve("Test text", critique)

        assert improved_text == "This is the improved text."
        assert len(mock_model.generate_calls) == 1
        assert "issue" in mock_model.generate_calls[0][0].lower()
        assert "suggestion" in mock_model.generate_calls[0][0].lower()

    def test_end_to_end_improvement(self, mock_model) -> None:
        """Test the end-to-end improvement process of ReflexionCritic."""
        # Create a mock improvement result

        original_text = "Test text"
        improved_text = "This is the improved text with more details and examples."

        # Create a mock _improve method that returns the improved text
        def mock_improve(self, text, critique):  # pylint: disable=unused-argument
            return improved_text

        # Create a mock critique dictionary
        critique = {
            "needs_improvement": True,
            "message": "This text could be improved in several ways.",
            "issues": ["The text is too vague.", "There are no specific examples."],
            "suggestions": ["Add more specific details.", "Include concrete examples."],
            "reflections": [
                "The text doesn't provide enough context for the reader.",
                "Without specific examples, the points being made are unclear.",
            ],
            "processing_time_ms": 100.0,
        }

        # Create a mock _critique method that returns the critique dictionary
        def mock_critique(self, text):  # pylint: disable=unused-argument
            return critique

        # Patch both methods
        from unittest.mock import patch

        with (
            patch.object(ReflexionCritic, "_critique", mock_critique),
            patch.object(ReflexionCritic, "_improve", mock_improve),
        ):

            critic = ReflexionCritic(model=mock_model)
            result_text, result = critic.improve(original_text)

            # Check the results
            assert result_text == improved_text
            assert result.original_text == original_text
            assert result.improved_text == improved_text
            assert result.changes_made is True


class TestNCriticsCritic:
    """Tests for the NCriticsCritic."""

    def test_init_with_defaults(self, mock_model) -> None:
        """Test initializing an NCriticsCritic with default parameters."""
        critic = NCriticsCritic(model=mock_model)
        assert critic.model is mock_model
        assert critic._name == "NCriticsCritic"
        assert critic.options == {}

    def test_init_with_options(self, mock_model) -> None:
        """Test initializing an NCriticsCritic with options."""
        critic = NCriticsCritic(model=mock_model, temperature=0.5)
        assert critic.temperature == 0.5

    def test_create_n_critics_critic(self, mock_model) -> None:
        """Test creating an NCriticsCritic using the factory function."""
        critic = create_n_critics_critic(model=mock_model)
        assert isinstance(critic, NCriticsCritic)
        assert critic.model is mock_model

    def test_create_n_critics_critic_with_options(self, mock_model) -> None:
        """Test creating an NCriticsCritic with options using the factory function."""
        critic = create_n_critics_critic(model=mock_model, temperature=0.5, num_critics=3)
        assert isinstance(critic, NCriticsCritic)
        assert critic.temperature == 0.5
        assert critic.num_critics == 3

    def test_end_to_end_improvement(self, mock_model) -> None:
        """Test the end-to-end improvement process of NCriticsCritic."""
        # Create a mock improvement result
        original_text = "Test text"
        improved_text = "This is the improved text with more details and examples."

        # Create a mock _improve method that returns the improved text
        def mock_improve(self, text, critique):  # pylint: disable=unused-argument
            return improved_text

        # Create a mock critique dictionary
        critique = {
            "needs_improvement": True,
            "message": "This text could be improved in several ways.",
            "issues": ["The text is too vague.", "There are no specific examples."],
            "suggestions": ["Add more specific details.", "Include concrete examples."],
            "critic_responses": [
                "Critic 1: The text is too vague.",
                "Critic 2: There are no specific examples.",
                "Critic 3: The structure is confusing.",
            ],
            "processing_time_ms": 100.0,
        }

        # Create a mock _critique method that returns the critique dictionary
        def mock_critique(self, text):  # pylint: disable=unused-argument
            return critique

        # Patch both methods
        from unittest.mock import patch

        with (
            patch.object(NCriticsCritic, "_critique", mock_critique),
            patch.object(NCriticsCritic, "_improve", mock_improve),
        ):

            critic = NCriticsCritic(model=mock_model)
            result_text, result = critic.improve(original_text)

            # Check the results
            assert result_text == improved_text
            assert result.original_text == original_text
            assert result.improved_text == improved_text
            assert result.changes_made is True
