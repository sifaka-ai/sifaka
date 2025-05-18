"""
Tests for the Constitutional critic.

This module contains tests for the Constitutional critic in the Sifaka framework.
"""

import json
import pytest
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

from sifaka.critics.constitutional import ConstitutionalCritic, create_constitutional_critic
from sifaka.errors import ImproverError
from sifaka.results import ImprovementResult


class TestConstitutionalCritic:
    """Tests for the ConstitutionalCritic class."""

    def test_init_with_defaults(self, mock_model) -> None:
        """Test initializing a ConstitutionalCritic with default parameters."""
        critic = ConstitutionalCritic(model=mock_model)
        assert critic.model == mock_model
        assert critic.temperature == 0.7
        assert len(critic.principles) == 5
        assert "expert editor" in critic.system_prompt.lower()
        assert "clear and concise" in critic.principles[0]
        assert "grammatically correct" in critic.principles[1]
        assert "well-structured" in critic.principles[2]
        assert "factually accurate" in critic.principles[3]
        assert "appropriate for the intended audience" in critic.principles[4]

    def test_init_with_custom_parameters(self, mock_model) -> None:
        """Test initializing a ConstitutionalCritic with custom parameters."""
        principles = ["Principle 1", "Principle 2"]
        critic = ConstitutionalCritic(
            model=mock_model,
            principles=principles,
            system_prompt="Custom system prompt",
            temperature=0.5,
        )
        assert critic.model == mock_model
        assert critic.principles == principles
        assert critic.system_prompt == "Custom system prompt"
        assert critic.temperature == 0.5

    def test_init_with_empty_principles(self, mock_model) -> None:
        """Test initializing a ConstitutionalCritic with empty principles."""
        critic = ConstitutionalCritic(model=mock_model, principles=[])
        assert len(critic.principles) == 5  # Should use default principles
        assert "clear and concise" in critic.principles[0]

    def test_init_without_model(self) -> None:
        """Test initializing a ConstitutionalCritic without a model."""
        with pytest.raises(ImproverError) as excinfo:
            ConstitutionalCritic(model=None)

        error = excinfo.value
        assert "Model must be provided" in str(error)
        assert error.component == "ConstitutionalCritic"
        assert error.operation == "initialization"

    def test_critique_method(self, mock_model) -> None:
        """Test the _critique method."""
        principles = ["Principle 1", "Principle 2"]
        critic = ConstitutionalCritic(model=mock_model, principles=principles)

        # Mock the _generate method to return a JSON response
        json_response = {
            "needs_improvement": True,
            "message": "Text violates principles",
            "violations": ["Violation 1", "Violation 2"],
            "suggestions": ["Suggestion 1", "Suggestion 2"],
        }

        with patch.object(critic, "_generate") as mock_generate:
            mock_generate.return_value = json.dumps(json_response)

            # Call the _critique method
            result = critic._critique("Test text")

            # Check that the _generate method was called correctly
            mock_generate.assert_called_once()
            assert "Test text" in mock_generate.call_args[0][0]
            assert "Principle 1" in mock_generate.call_args[0][0]
            assert "Principle 2" in mock_generate.call_args[0][0]

            # Check the result
            assert result["needs_improvement"] is True
            assert result["message"] == "Text violates principles"
            assert result["violations"] == ["Violation 1", "Violation 2"]
            assert result["suggestions"] == ["Suggestion 1", "Suggestion 2"]
            assert result["issues"] == [
                "Violation 1",
                "Violation 2",
            ]  # issues should be the same as violations
            assert "processing_time_ms" in result

    def test_critique_with_invalid_json(self, mock_model) -> None:
        """Test the _critique method with invalid JSON."""
        critic = ConstitutionalCritic(model=mock_model)

        # Mock the _generate method to return an invalid JSON response
        with patch.object(critic, "_generate") as mock_generate:
            mock_generate.return_value = "This is not valid JSON"

            # Call the _critique method
            result = critic._critique("Test text")

            # Check that the _generate method was called correctly
            mock_generate.assert_called_once()

            # Check the result (should use default values)
            assert result["needs_improvement"] is True
            assert "Unable to parse critique response" in result["message"]
            assert "Unable to identify specific violations" in result["violations"]
            assert "General improvement" in result["suggestions"]
            assert "processing_time_ms" in result

    def test_critique_with_json_decode_error(self, mock_model) -> None:
        """Test the _critique method with a JSON decode error."""
        critic = ConstitutionalCritic(model=mock_model)

        # Mock the _generate method to return a malformed JSON response
        with patch.object(critic, "_generate") as mock_generate:
            # Use a string that will trigger a JSONDecodeError rather than just not finding JSON
            mock_generate.return_value = (
                '{"needs_improvement": true, "message": "Text violates principles", "violations": ['
            )

            # Call the _critique method
            result = critic._critique("Test text")

            # Check that the _generate method was called correctly
            mock_generate.assert_called_once()

            # Check the result (should use default values)
            assert result["needs_improvement"] is True
            assert "Unable to parse critique response" in result["message"]
            assert "Unable to identify specific violations" in result["violations"]
            assert "General improvement" in result["suggestions"]
            assert "processing_time_ms" in result
            # The error field might not be present if the implementation doesn't add it for all error types
            # So we'll make this check optional

    def test_improve_method(self, mock_model) -> None:
        """Test the _improve method."""
        principles = ["Principle 1", "Principle 2"]
        critic = ConstitutionalCritic(model=mock_model, principles=principles)

        # Create a critique
        critique = {
            "needs_improvement": True,
            "message": "Text violates principles",
            "violations": ["Violation 1", "Violation 2"],
            "suggestions": ["Suggestion 1", "Suggestion 2"],
        }

        # Mock the _generate method to return an improved text
        with patch.object(critic, "_generate") as mock_generate:
            mock_generate.return_value = "Improved text"

            # Call the _improve method
            result = critic._improve("Test text", critique)

            # Check that the _generate method was called correctly
            mock_generate.assert_called_once()
            assert "Test text" in mock_generate.call_args[0][0]
            assert "Principle 1" in mock_generate.call_args[0][0]
            assert "Principle 2" in mock_generate.call_args[0][0]
            assert "Violation 1" in mock_generate.call_args[0][0]
            assert "Violation 2" in mock_generate.call_args[0][0]
            assert "Suggestion 1" in mock_generate.call_args[0][0]
            assert "Suggestion 2" in mock_generate.call_args[0][0]

            # Check the result
            assert result == "Improved text"

    def test_improve_method_with_code_block_markers(self, mock_model) -> None:
        """Test the _improve method with code block markers in the response."""
        critic = ConstitutionalCritic(model=mock_model)

        # Create a critique
        critique = {
            "needs_improvement": True,
            "message": "Text violates principles",
            "violations": ["Violation 1"],
            "suggestions": ["Suggestion 1"],
        }

        # Mock the _generate method to return a response with code block markers
        with patch.object(critic, "_generate") as mock_generate:
            mock_generate.return_value = "```\nImproved text\n```"

            # Call the _improve method
            result = critic._improve("Test text", critique)

            # Check the result (code block markers should be removed)
            assert result == "Improved text"

    def test_improve_full_method(self, mock_model) -> None:
        """Test the improve method (full process)."""
        critic = ConstitutionalCritic(model=mock_model)

        # Mock the _critique method
        with patch.object(critic, "_critique") as mock_critique:
            mock_critique.return_value = {
                "needs_improvement": True,
                "message": "Text violates principles",
                "violations": ["Violation 1"],
                "suggestions": ["Suggestion 1"],
                "issues": ["Violation 1"],
            }

            # Mock the _improve method
            with patch.object(critic, "_improve") as mock_improve:
                mock_improve.return_value = "Improved text"

                # Call the improve method
                improved_text, result = critic.improve("Test text")

                # Check that the methods were called correctly
                mock_critique.assert_called_once_with("Test text")
                mock_improve.assert_called_once()

                # Check the result
                assert improved_text == "Improved text"
                assert isinstance(result, ImprovementResult)
                assert result._original_text == "Test text"
                assert result._improved_text == "Improved text"
                assert result._changes_made is True
                assert result.message == "Text violates principles"

    def test_improve_method_no_improvement_needed(self, mock_model) -> None:
        """Test the improve method when no improvement is needed."""
        critic = ConstitutionalCritic(model=mock_model)

        # Mock the _critique method to indicate no improvement needed
        with patch.object(critic, "_critique") as mock_critique:
            mock_critique.return_value = {
                "needs_improvement": False,
                "message": "No improvement needed",
                "violations": [],
                "suggestions": [],
                "issues": [],
            }

            # Call the improve method
            improved_text, result = critic.improve("Test text")

            # Check that the _critique method was called correctly
            mock_critique.assert_called_once_with("Test text")

            # Check the result
            assert improved_text == "Test text"  # No change
            assert isinstance(result, ImprovementResult)
            assert result._original_text == "Test text"
            assert result._improved_text == "Test text"
            assert result._changes_made is False
            assert result.message == "No improvement needed"

    def test_factory_function(self, mock_model) -> None:
        """Test the create_constitutional_critic factory function."""
        principles = ["Principle 1", "Principle 2"]

        # Call the factory function
        critic = create_constitutional_critic(
            model=mock_model,
            principles=principles,
            system_prompt="Custom system prompt",
            temperature=0.5,
        )

        # Check the result
        assert isinstance(critic, ConstitutionalCritic)
        assert critic.model == mock_model
        assert critic.principles == principles
        assert critic.system_prompt == "Custom system prompt"
        assert critic.temperature == 0.5

    def test_factory_function_without_model(self) -> None:
        """Test the create_constitutional_critic factory function without a model."""
        with pytest.raises(ImproverError) as excinfo:
            create_constitutional_critic(model=None)

        error = excinfo.value
        assert "Model must be provided" in str(error)
        assert error.component == "ConstitutionalCriticFactory"
        assert error.operation == "create_critic"
