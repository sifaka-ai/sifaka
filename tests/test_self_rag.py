"""
Tests for the Self-RAG critic.

This module contains tests for the Self-RAG critic in the Sifaka framework.
"""

import json
from typing import List, Optional
from unittest.mock import MagicMock, patch

import pytest

from sifaka.critics.self_rag import SelfRAGCritic, create_self_rag_critic
from sifaka.errors import ImproverError, RetrieverError
from sifaka.results import ImprovementResult


class MockRetriever:
    """Mock retriever for testing."""

    def __init__(self, passages: Optional[List[str]] = None):
        """Initialize the mock retriever."""
        self.passages = passages or ["Passage 1", "Passage 2", "Passage 3"]
        self.retrieve_calls = []

    def retrieve(self, query: str) -> List[str]:
        """Retrieve passages for a query."""
        self.retrieve_calls.append(query)
        return self.passages


class TestSelfRAGCritic:
    """Tests for the SelfRAGCritic class."""

    def test_init_with_defaults(self, mock_model) -> None:
        """Test initializing a SelfRAGCritic with default parameters."""
        retriever = MockRetriever()
        critic = SelfRAGCritic(model=mock_model, retriever=retriever)

        assert critic.model == mock_model
        assert critic.retriever == retriever
        assert critic.reflection_enabled is True
        assert critic.max_passages == 5
        assert critic.temperature == 0.7
        assert critic.name == "SelfRAGCritic"
        assert "expert editor" in critic.system_prompt.lower()

    def test_init_with_custom_parameters(self, mock_model) -> None:
        """Test initializing a SelfRAGCritic with custom parameters."""
        retriever = MockRetriever()
        critic = SelfRAGCritic(
            model=mock_model,
            retriever=retriever,
            system_prompt="Custom system prompt",
            temperature=0.5,
            reflection_enabled=False,
            max_passages=3,
            name="CustomName",
        )

        assert critic.model == mock_model
        assert critic.retriever == retriever
        assert critic.reflection_enabled is False
        assert critic.max_passages == 3
        assert critic.temperature == 0.5
        assert critic.name == "CustomName"
        assert critic.system_prompt == "Custom system prompt"

    def test_init_without_model(self) -> None:
        """Test initializing a SelfRAGCritic without a model."""
        retriever = MockRetriever()

        with pytest.raises(ImproverError) as excinfo:
            SelfRAGCritic(model=None, retriever=retriever)

        assert "model not provided" in str(excinfo.value).lower()

    def test_init_without_retriever(self, mock_model) -> None:
        """Test initializing a SelfRAGCritic without a retriever."""
        with pytest.raises(ImproverError) as excinfo:
            SelfRAGCritic(model=mock_model, retriever=None)

        assert "retriever not provided" in str(excinfo.value).lower()

    def test_format_passages(self, mock_model) -> None:
        """Test the _format_passages method."""
        retriever = MockRetriever()
        critic = SelfRAGCritic(model=mock_model, retriever=retriever)

        passages = ["First passage", "Second passage", "Third passage"]
        formatted = critic._format_passages(passages)

        assert "Passage 1:\nFirst passage" in formatted
        assert "Passage 2:\nSecond passage" in formatted
        assert "Passage 3:\nThird passage" in formatted

    def test_critique_with_valid_json_response(self, mock_model) -> None:
        """Test the _critique method with a valid JSON response."""
        retriever = MockRetriever(["Relevant information 1", "Relevant information 2"])
        critic = SelfRAGCritic(model=mock_model, retriever=retriever)

        # Disable reflection to simplify the test
        critic.reflection_enabled = False

        # Mock the _generate method to return a valid JSON response
        with patch.object(critic, "_generate") as mock_generate:
            mock_generate.return_value = json.dumps(
                {
                    "needs_improvement": True,
                    "message": "Text needs improvement",
                    "queries": ["Query 1", "Query 2"],
                    "areas_for_improvement": ["Area 1", "Area 2"],
                }
            )

            # Call the _critique method
            result = critic._critique("Test text")

            # Check that the _generate method was called at least once
            assert mock_generate.call_count >= 1

            # Check the result
            assert result["needs_improvement"] is True
            assert result["message"] == "Text needs improvement"
            assert "queries" in result
            assert "areas_for_improvement" in result
            assert "retrieved_passages" in result
            assert "issues" in result  # This should be set from areas_for_improvement
            assert "suggestions" in result
            assert len(result["suggestions"]) > 0
            assert "processing_time_ms" in result

    def test_critique_with_invalid_json_response(self, mock_model) -> None:
        """Test the _critique method with an invalid JSON response."""
        retriever = MockRetriever()
        critic = SelfRAGCritic(model=mock_model, retriever=retriever)

        # Disable reflection to simplify the test
        critic.reflection_enabled = False

        # Mock the _generate method to return an invalid JSON response
        with patch.object(critic, "_generate") as mock_generate:
            # Use a string that looks like JSON but doesn't have the expected fields
            mock_generate.return_value = '{"foo": "bar"}'

            # Call the _critique method
            result = critic._critique("Test text")

            # Check that the _generate method was called at least once
            assert mock_generate.call_count >= 1

            # Check the result (should use default values)
            assert result["needs_improvement"] is True
            assert "queries" in result
            assert "areas_for_improvement" in result
            assert "issues" in result  # This should be set from areas_for_improvement
            assert "suggestions" in result
            assert "processing_time_ms" in result

    def test_critique_with_json_decode_error(self, mock_model) -> None:
        """Test the _critique method with a JSON decode error."""
        retriever = MockRetriever()
        critic = SelfRAGCritic(model=mock_model, retriever=retriever)

        # Disable reflection to simplify the test
        critic.reflection_enabled = False

        # Mock the _generate method to return a malformed JSON response
        with patch.object(critic, "_generate") as mock_generate:
            # Use a string that will trigger a JSONDecodeError
            mock_generate.return_value = (
                '{"needs_improvement": true, "message": "Text needs improvement", "queries": ['
            )

            # Call the _critique method
            result = critic._critique("Test text")

            # Check that the _generate method was called at least once
            assert mock_generate.call_count >= 1

            # Check the result (should use default values)
            assert result["needs_improvement"] is True
            assert "queries" in result
            assert "areas_for_improvement" in result
            # Note: The implementation doesn't add "issues" or "suggestions" for JSON decode errors
            # in the default response, so we don't check for them
            assert "processing_time_ms" in result

    def test_critique_with_retriever_error(self, mock_model) -> None:
        """Test the _critique method with a retriever error."""
        # Create a retriever that raises an error
        retriever = MagicMock()
        retriever.retrieve.side_effect = RetrieverError("Retriever error")

        critic = SelfRAGCritic(model=mock_model, retriever=retriever)

        # Disable reflection to simplify the test
        critic.reflection_enabled = False

        # Mock the _generate method to return a valid JSON response
        with patch.object(critic, "_generate") as mock_generate:
            mock_generate.return_value = json.dumps(
                {
                    "needs_improvement": True,
                    "message": "Text needs improvement",
                    "queries": ["Query 1", "Query 2"],
                    "areas_for_improvement": ["Area 1", "Area 2"],
                }
            )

            # Call the _critique method (should not raise an error because the implementation
            # catches RetrieverError and continues with empty passages)
            result = critic._critique("Test text")

            # Check that the _generate method was called at least once
            assert mock_generate.call_count >= 1

            # Check the result
            assert result["needs_improvement"] is True
            assert "queries" in result
            assert "areas_for_improvement" in result
            assert "retrieved_passages" in result
            assert len(result["retrieved_passages"]) == 0  # No passages retrieved due to error

    def test_improve_with_retrieved_passages(self, mock_model) -> None:
        """Test the _improve method with retrieved passages."""
        retriever = MockRetriever()
        critic = SelfRAGCritic(model=mock_model, retriever=retriever)

        # Disable reflection to simplify the test
        critic.reflection_enabled = False

        # Create a critique with retrieved passages
        critique = {
            "needs_improvement": True,
            "message": "Text needs improvement",
            "areas_for_improvement": ["Area 1", "Area 2"],
            "retrieved_passages": ["Passage 1", "Passage 2"],
            "reflection": "Reflection on the passages",
        }

        # Mock the _generate method to return an improved text
        with patch.object(critic, "_generate") as mock_generate:
            mock_generate.return_value = "Improved text"

            # Call the _improve method
            result = critic._improve("Test text", critique)

            # Check that the _generate method was called at least once
            assert mock_generate.call_count >= 1

            # Check the result
            assert result == "Improved text"

    def test_improve_without_retrieved_passages(self, mock_model) -> None:
        """Test the _improve method without retrieved passages."""
        retriever = MockRetriever()
        critic = SelfRAGCritic(model=mock_model, retriever=retriever)

        # Create a critique without retrieved passages
        critique = {
            "needs_improvement": True,
            "message": "Text needs improvement",
            "areas_for_improvement": ["Area 1", "Area 2"],
            "retrieved_passages": [],
        }

        # Call the _improve method
        result = critic._improve("Test text", critique)

        # Check the result (should return the original text)
        assert result == "Test text"

    def test_improve_with_code_block_markers(self, mock_model) -> None:
        """Test the _improve method with code block markers in the response."""
        retriever = MockRetriever()
        critic = SelfRAGCritic(model=mock_model, retriever=retriever)

        # Create a critique with retrieved passages
        critique = {
            "needs_improvement": True,
            "message": "Text needs improvement",
            "areas_for_improvement": ["Area 1", "Area 2"],
            "retrieved_passages": ["Passage 1", "Passage 2"],
        }

        # Mock the _generate method to return a response with code block markers
        with patch.object(critic, "_generate") as mock_generate:
            mock_generate.return_value = "```\nImproved text\n```"

            # Call the _improve method
            result = critic._improve("Test text", critique)

            # Check the result (code block markers should be removed)
            assert result == "Improved text"

    def test_improve_with_reflection_enabled(self, mock_model) -> None:
        """Test the _improve method with reflection enabled."""
        retriever = MockRetriever()
        critic = SelfRAGCritic(model=mock_model, retriever=retriever, reflection_enabled=True)

        # Create a critique with retrieved passages
        critique = {
            "needs_improvement": True,
            "message": "Text needs improvement",
            "areas_for_improvement": ["Area 1", "Area 2"],
            "retrieved_passages": ["Passage 1", "Passage 2"],
        }

        # Mock the _generate method to return different responses for improvement and reflection
        with patch.object(critic, "_generate") as mock_generate:
            mock_generate.side_effect = ["Improved text", "Reflection on improvement"]

            # Call the _improve method
            result = critic._improve("Test text", critique)

            # Check that the _generate method was called twice (once for improvement, once for reflection)
            assert mock_generate.call_count == 2

            # Check the result
            assert result == "Improved text"

    def test_improve_with_reflection_error(self, mock_model) -> None:
        """Test the _improve method when reflection generation fails."""
        retriever = MockRetriever()
        critic = SelfRAGCritic(model=mock_model, retriever=retriever, reflection_enabled=True)

        # Create a critique with retrieved passages
        critique = {
            "needs_improvement": True,
            "message": "Text needs improvement",
            "areas_for_improvement": ["Area 1", "Area 2"],
            "retrieved_passages": ["Passage 1", "Passage 2"],
        }

        # Mock the _generate method to return a response for improvement but raise an error for reflection
        with patch.object(critic, "_generate") as mock_generate:
            mock_generate.side_effect = ["Improved text", Exception("Reflection error")]

            # Call the _improve method (should not raise an error)
            result = critic._improve("Test text", critique)

            # Check that the _generate method was called twice (once for improvement, once for reflection)
            assert mock_generate.call_count == 2

            # Check the result (should still return the improved text)
            assert result == "Improved text"

    def test_improve_with_error(self, mock_model) -> None:
        """Test the _improve method when an error occurs."""
        retriever = MockRetriever()
        critic = SelfRAGCritic(model=mock_model, retriever=retriever)

        # Create a critique with retrieved passages
        critique = {
            "needs_improvement": True,
            "message": "Text needs improvement",
            "areas_for_improvement": ["Area 1", "Area 2"],
            "retrieved_passages": ["Passage 1", "Passage 2"],
        }

        # Mock the _generate method to raise an error
        with patch.object(critic, "_generate") as mock_generate:
            mock_generate.side_effect = Exception("Improvement error")

            # Call the _improve method (should raise an error)
            with pytest.raises(ImproverError) as excinfo:
                critic._improve("Test text", critique)

            # Check the error
            assert "Error improving text" in str(excinfo.value)
            assert excinfo.value.component == "SelfRAGCritic"
            assert excinfo.value.operation == "improvement"

    def test_improve_full_method(self, mock_model) -> None:
        """Test the improve method (full process)."""
        retriever = MockRetriever()
        critic = SelfRAGCritic(model=mock_model, retriever=retriever)

        # Mock the _critique method
        with patch.object(critic, "_critique") as mock_critique:
            mock_critique.return_value = {
                "needs_improvement": True,
                "message": "Text needs improvement",
                "areas_for_improvement": ["Area 1", "Area 2"],
                "retrieved_passages": ["Passage 1", "Passage 2"],
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
                assert result.message == "Text needs improvement"

    def test_factory_function(self, mock_model) -> None:
        """Test the create_self_rag_critic factory function."""
        retriever = MockRetriever()

        # Call the factory function
        critic = create_self_rag_critic(
            model=mock_model,
            retriever=retriever,
            system_prompt="Custom system prompt",
            temperature=0.5,
            reflection_enabled=False,
            max_passages=3,
        )

        # Check the result
        assert isinstance(critic, SelfRAGCritic)
        assert critic.model == mock_model
        assert critic.retriever == retriever
        assert critic.reflection_enabled is False
        assert critic.max_passages == 3
        assert critic.temperature == 0.5
        assert critic.system_prompt == "Custom system prompt"

    def test_factory_function_with_error(self, mock_model) -> None:
        """Test the create_self_rag_critic factory function when an error occurs."""
        # Call the factory function without a retriever (should raise an error)
        with pytest.raises(ImproverError) as excinfo:
            create_self_rag_critic(model=mock_model, retriever=None)

        # Check the error
        assert "Failed to create SelfRAGCritic" in str(excinfo.value)
        assert excinfo.value.component == "SelfRAGCriticFactory"
        assert excinfo.value.operation == "create_critic"
