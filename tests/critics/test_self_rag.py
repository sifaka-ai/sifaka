"""
Tests for the SelfRAGCritic.

This module contains tests for the SelfRAGCritic implementation.
"""

import pytest
from unittest.mock import MagicMock, patch

from sifaka.critics.self_rag import (
    SelfRAGCritic,
    SelfRAGCriticConfig,
    create_self_rag_critic,
)
from sifaka.retrieval import SimpleRetriever


class TestSelfRAGCritic:
    """Tests for the SelfRAGCritic class."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock model provider."""
        model = MagicMock()
        model.generate.side_effect = [
            "search for health insurance claim steps",  # retrieval query
            "To file a health insurance claim, you need to follow these steps...",  # response
            "The response accurately addresses the query using the retrieved information...",  # reflection
        ]
        return model

    @pytest.fixture
    def mock_retriever(self):
        """Create a mock retriever."""
        retriever = MagicMock()
        retriever.retrieve.return_value = "To file a claim for health reimbursement, follow these steps: 1. Complete the claim form..."
        return retriever

    @pytest.fixture
    def critic(self, mock_model, mock_retriever):
        """Create a SelfRAGCritic instance with mock components."""
        config = SelfRAGCriticConfig(
            name="test_critic",
            description="Test critic",
            system_prompt="You are a test critic.",
            temperature=0.7,
            max_tokens=1000,
            retrieval_threshold=0.5,
        )
        return SelfRAGCritic(
            config=config,
            llm_provider=mock_model,
            retriever=mock_retriever,
        )

    def test_initialization(self, mock_model, mock_retriever):
        """Test that the critic initializes correctly."""
        config = SelfRAGCriticConfig(
            name="test_critic",
            description="Test critic",
        )
        critic = SelfRAGCritic(
            config=config,
            llm_provider=mock_model,
            retriever=mock_retriever,
        )
        assert critic._state.model == mock_model
        assert critic._state.cache.get("retriever") == mock_retriever
        assert critic._state.initialized is True
        assert "system_prompt" in critic._state.cache
        assert "retrieval_threshold" in critic._state.cache

    def test_run(self, critic, mock_model, mock_retriever):
        """Test the run method."""
        result = critic.run("What are the steps to file a health insurance claim?")

        # Check that the model was called for retrieval query, response, and reflection
        assert mock_model.generate.call_count == 3

        # Check that the retriever was called with the retrieval query
        mock_retriever.retrieve.assert_called_once()

        # Check the result structure
        assert "response" in result
        assert "retrieval_query" in result
        assert "retrieved_context" in result
        assert "reflection" in result

        # Check the result values
        assert result["retrieval_query"] == "search for health insurance claim steps"
        assert "health insurance claim" in result["response"]
        assert "accurately addresses" in result["reflection"]

    def test_validate(self, critic):
        """Test the validate method."""
        # SelfRAGCritic.validate always returns True
        assert critic.validate("Any text") is True

    def test_critique(self, critic, mock_model):
        """Test the critique method."""
        # Reset mock to ensure we get predictable behavior
        mock_model.generate.reset_mock()
        mock_model.generate.side_effect = [
            "search for health insurance claim steps",  # retrieval query
            "To file a health insurance claim, you need to follow these steps...",  # response
            "The response is good but could be more detailed. \n- Missing information about online submission.",  # reflection with issues
        ]

        # Mock the run method to return a response with issues
        critic.run = MagicMock(
            return_value={
                "response": "To file a health insurance claim, you need to follow these steps...",
                "retrieval_query": "search for health insurance claim steps",
                "retrieved_context": "context",
                "reflection": "The response is good but could be more detailed. \n- Missing information about online submission.",
            }
        )

        result = critic.critique("This is a health insurance claim.")

        # Check the result structure
        assert "score" in result
        assert "feedback" in result
        assert "issues" in result
        assert "suggestions" in result

        # Check that issues were extracted from the reflection
        assert len(result["issues"]) > 0
        assert "Missing information" in result["issues"][0]

    def test_improve(self, critic, mock_model):
        """Test the improve method."""
        # Reset mock to ensure we get predictable behavior
        mock_model.generate.reset_mock()
        mock_model.generate.side_effect = [
            "search for health insurance claim steps",  # retrieval query
            "Improved response with detailed steps for filing a health insurance claim...",  # response
            "The response now includes all necessary information...",  # reflection
        ]

        # Mock the run method to return a response with "detailed steps"
        critic.run = MagicMock(
            return_value={
                "response": "Improved response with detailed steps for filing a health insurance claim...",
                "retrieval_query": "search for health insurance claim steps",
                "retrieved_context": "context",
                "reflection": "reflection",
            }
        )

        improved_text = critic.improve("Basic health insurance claim info.")

        # Check that the improved text is returned
        assert "detailed steps" in improved_text

    def test_improve_with_feedback(self, critic, mock_model):
        """Test the improve_with_feedback method."""
        # Reset mock and set a specific return value
        mock_model.generate.reset_mock()
        mock_model.generate.side_effect = None  # Clear any previous side effects
        mock_model.generate.return_value = "Improved text based on feedback."

        # Create a new critic with the mock model to avoid side effects from other tests
        config = SelfRAGCriticConfig(
            name="test_critic",
            description="Test critic",
        )
        test_critic = SelfRAGCritic(
            config=config,
            llm_provider=mock_model,
            retriever=MagicMock(),
        )

        improved_text = test_critic.improve_with_feedback(
            "Original text.", "Add more details about the claim process."
        )

        # Check that the model was called with the feedback
        assert mock_model.generate.call_count == 1
        args = mock_model.generate.call_args[0]
        assert "feedback" in args[0].lower()
        assert "original text" in args[0].lower()

        # Check the result
        assert improved_text == "Improved text based on feedback."

    def test_factory_function(self, mock_model, mock_retriever):
        """Test the create_self_rag_critic factory function."""
        critic = create_self_rag_critic(
            llm_provider=mock_model,
            retriever=mock_retriever,
            name="factory_critic",
            description="Created with factory function",
            system_prompt="Custom system prompt",
            temperature=0.8,
            max_tokens=1500,
        )

        # Check that the critic was created with the correct parameters
        assert critic._state.model == mock_model
        assert critic._state.cache.get("retriever") == mock_retriever
        assert critic._state.cache["system_prompt"] == "Custom system prompt"
        assert critic._state.cache["temperature"] == 0.8
        assert critic._state.cache["max_tokens"] == 1500

    def test_empty_input_validation(self, critic):
        """Test validation of empty inputs."""
        with pytest.raises(ValueError):
            critic.run("")

        with pytest.raises(ValueError):
            critic.improve("")

        with pytest.raises(ValueError):
            critic.critique("")

        with pytest.raises(ValueError):
            critic.improve_with_feedback("Valid text", "")


class TestSimpleRetriever:
    """Tests for the SimpleRetriever class."""

    def test_initialization(self):
        """Test that the retriever initializes correctly."""
        documents = {
            "doc1": "This is document 1",
            "doc2": "This is document 2",
        }
        retriever = SimpleRetriever(documents=documents)
        assert retriever.documents == documents

    def test_retrieve(self):
        """Test the retrieve method."""
        documents = {
            "health": "Health insurance information",
            "travel": "Travel insurance details",
        }
        retriever = SimpleRetriever(documents=documents)
        result = retriever.retrieve("Tell me about health insurance")

        # Check that the result contains the health document
        assert "Health insurance information" in result

    def test_empty_query(self):
        """Test handling of empty queries."""
        retriever = SimpleRetriever(documents={"doc": "content"})
        with pytest.raises(ValueError):
            retriever.retrieve("")

    def test_no_matching_documents(self):
        """Test handling of queries with no matching documents."""
        retriever = SimpleRetriever(documents={"doc": "content"})
        result = retriever.retrieve("something completely unrelated")
        assert "No relevant documents found" in result
