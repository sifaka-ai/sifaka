"""
Tests for the N-Critics critic.

This module contains tests for the N-Critics critic in the Sifaka framework.
"""

import json
from unittest.mock import patch

import pytest

from sifaka.critics.n_critics import NCriticsCritic
from sifaka.errors import ImproverError
from sifaka.results import ImprovementResult


class TestNCriticsCritic:
    """Tests for the NCriticsCritic class."""

    def test_init_with_defaults(self, mock_model) -> None:
        """Test initializing an NCriticsCritic with default parameters."""
        critic = NCriticsCritic(model=mock_model)
        assert critic.model == mock_model
        assert critic.temperature == 0.7
        assert critic.num_critics == 3
        assert len(critic.critic_roles) == 3

    def test_init_with_custom_parameters(self, mock_model) -> None:
        """Test initializing an NCriticsCritic with custom parameters."""
        critic = NCriticsCritic(
            model=mock_model,
            system_prompt="Custom system prompt",
            temperature=0.5,
            num_critics=2,
        )
        assert critic.model == mock_model
        assert critic.system_prompt == "Custom system prompt"
        assert critic.temperature == 0.5
        assert critic.num_critics == 2
        assert len(critic.critic_roles) == 2

    def test_init_with_invalid_parameters(self, mock_model) -> None:
        """Test initializing an NCriticsCritic with invalid parameters."""
        # Test with num_critics out of range (should be clamped)
        critic = NCriticsCritic(model=mock_model, num_critics=10)
        assert critic.num_critics == 5  # Clamped to maximum of 5

        critic = NCriticsCritic(model=mock_model, num_critics=0)
        assert critic.num_critics == 1  # Clamped to minimum of 1

        # The max_refinement_iterations parameter has been removed from NCriticsCritic

    def test_init_without_model(self) -> None:
        """Test initializing an NCriticsCritic without a model."""
        with pytest.raises(ImproverError) as excinfo:
            NCriticsCritic(model=None)

        error = excinfo.value
        assert "Model must be provided" in str(error)
        assert error.component == "NCriticsCritic"
        assert error.operation == "initialization"

    def test_critique_method(self, mock_model) -> None:
        """Test the _critique method."""
        # Mock the _generate_critic_critique method
        critic = NCriticsCritic(model=mock_model, num_critics=2)

        # Create mock critiques
        mock_critiques = [
            {
                "role": "Factual Accuracy Critic",
                "needs_improvement": True,
                "score": 6,
                "issues": ["Issue 1", "Issue 2"],
                "suggestions": ["Suggestion 1", "Suggestion 2"],
                "explanation": "Explanation 1",
            },
            {
                "role": "Coherence and Clarity Critic",
                "needs_improvement": False,
                "score": 8,
                "issues": ["Issue 3"],
                "suggestions": ["Suggestion 3"],
                "explanation": "Explanation 2",
            },
        ]

        # Mock the _generate_critic_critique method
        with patch.object(critic, "_generate_critic_critique") as mock_generate:
            mock_generate.side_effect = mock_critiques

            # Mock the _aggregate_critiques method
            with patch.object(critic, "_aggregate_critiques") as mock_aggregate:
                mock_aggregate.return_value = {
                    "summary": "Aggregated summary",
                    "issues": ["Issue 1", "Issue 2", "Issue 3"],
                    "suggestions": ["Suggestion 1", "Suggestion 2", "Suggestion 3"],
                    "average_score": 7.0,
                }

                # Call the _critique method
                result = critic._critique("Test text")

                # Check that the methods were called correctly
                assert mock_generate.call_count == 2
                mock_aggregate.assert_called_once_with(mock_critiques)

                # Check the result
                assert result["needs_improvement"] is True
                assert result["message"] == "Aggregated summary"
                assert len(result["critic_critiques"]) == 2
                assert result["aggregated_critique"]["summary"] == "Aggregated summary"
                assert "processing_time_ms" in result

    def test_generate_critic_critique(self, mock_model) -> None:
        """Test the _generate_critic_critique method."""
        critic = NCriticsCritic(model=mock_model)

        # Mock the _generate method to return a JSON response
        json_response = {
            "role": "Test Critic",
            "needs_improvement": True,
            "score": 7,
            "issues": ["Issue 1", "Issue 2"],
            "suggestions": ["Suggestion 1", "Suggestion 2"],
            "explanation": "Test explanation",
        }

        with patch.object(critic, "_generate") as mock_generate:
            mock_generate.return_value = json.dumps(json_response)

            # Call the _generate_critic_critique method
            result = critic._generate_critic_critique("Test text", "Test Critic")

            # Check that the _generate method was called correctly
            mock_generate.assert_called_once()
            assert "Test text" in mock_generate.call_args[0][0]
            assert "Test Critic" in mock_generate.call_args[0][0]

            # Check the result
            assert result["role"] == "Test Critic"
            assert result["needs_improvement"] is True
            assert result["score"] == 7
            assert result["issues"] == ["Issue 1", "Issue 2"]
            assert result["suggestions"] == ["Suggestion 1", "Suggestion 2"]
            assert result["explanation"] == "Test explanation"
            assert "processing_time_ms" in result

    def test_generate_critic_critique_with_invalid_json(self, mock_model) -> None:
        """Test the _generate_critic_critique method with invalid JSON."""
        critic = NCriticsCritic(model=mock_model)

        # Mock the _generate method to return an invalid JSON response
        with patch.object(critic, "_generate") as mock_generate:
            mock_generate.return_value = "This is not valid JSON"

            # Call the _generate_critic_critique method
            result = critic._generate_critic_critique("Test text", "Test Critic")

            # Check that the _generate method was called correctly
            mock_generate.assert_called_once()

            # Check the result (should use default values)
            assert result["role"] == "Test Critic"
            assert result["needs_improvement"] is True
            assert result["score"] == 5
            assert "Unable to parse critique response" in result["issues"]
            assert "General improvement needed" in result["suggestions"]
            assert "processing_time_ms" in result

    def test_aggregate_critiques(self, mock_model) -> None:
        """Test the _aggregate_critiques method."""
        critic = NCriticsCritic(model=mock_model)

        # Create test critiques
        critiques = [
            {
                "role": "Critic 1",
                "needs_improvement": True,
                "score": 6,
                "issues": ["Issue 1", "Issue 2"],
                "suggestions": ["Suggestion 1", "Suggestion 2"],
                "explanation": "Explanation 1",
            },
            {
                "role": "Critic 2",
                "needs_improvement": False,
                "score": 8,
                "issues": ["Issue 3"],
                "suggestions": ["Suggestion 3"],
                "explanation": "Explanation 2",
            },
        ]

        # Call the _aggregate_critiques method
        result = critic._aggregate_critiques(critiques)

        # Check the result
        assert "summary" in result
        assert "issues" in result
        assert "suggestions" in result
        assert "average_score" in result
        assert result["issues"] == ["Issue 1", "Issue 2", "Issue 3"]
        assert result["suggestions"] == ["Suggestion 1", "Suggestion 2", "Suggestion 3"]
        assert result["average_score"] == 7.0
        assert "processing_time_ms" in result

    def test_improve_method(self, mock_model) -> None:
        """Test the improve method."""
        critic = NCriticsCritic(model=mock_model, num_critics=1)

        # Mock the _critique method
        with patch.object(critic, "_critique") as mock_critique:
            mock_critique.return_value = {
                "needs_improvement": True,
                "message": "Test message",
                "issues": ["Issue 1"],
                "suggestions": ["Suggestion 1"],
                "critic_critiques": [
                    {
                        "role": "Test Critic",
                        "needs_improvement": True,
                        "score": 7,
                        "issues": ["Issue 1"],
                        "suggestions": ["Suggestion 1"],
                        "explanation": "Test explanation",
                    }
                ],
                "aggregated_critique": {
                    "summary": "Test summary",
                    "issues": ["Issue 1"],
                    "suggestions": ["Suggestion 1"],
                    "average_score": 7.0,
                },
                "processing_time_ms": 100.0,
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
                assert result.message == "Test message"

    def test_improve_method_no_improvement_needed(self, mock_model) -> None:
        """Test the improve method when no improvement is needed."""
        critic = NCriticsCritic(model=mock_model)

        # Mock the _critique method to indicate no improvement needed
        with patch.object(critic, "_critique") as mock_critique:
            mock_critique.return_value = {
                "needs_improvement": False,
                "message": "No improvement needed",
                "issues": [],
                "suggestions": [],
                "critic_critiques": [
                    {
                        "role": "Test Critic",
                        "needs_improvement": False,
                        "score": 9,
                        "issues": [],
                        "suggestions": [],
                        "explanation": "Text is already good",
                    }
                ],
                "aggregated_critique": {
                    "summary": "No improvement needed",
                    "issues": [],
                    "suggestions": [],
                    "average_score": 9.0,
                },
                "processing_time_ms": 100.0,
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
