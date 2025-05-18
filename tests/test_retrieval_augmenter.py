"""
Tests for the RetrievalAugmenter.

This module contains tests for the RetrievalAugmenter in the Sifaka framework.
"""

import json
import pytest
from typing import List, Optional
from unittest.mock import MagicMock, patch

from sifaka.retrievers.augmenter import RetrievalAugmenter
from sifaka.errors import RetrieverError


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


class TestRetrievalAugmenter:
    """Tests for the RetrievalAugmenter class."""

    def test_init_with_defaults(self) -> None:
        """Test initializing a RetrievalAugmenter with default parameters."""
        retriever = MockRetriever()
        augmenter = RetrievalAugmenter(retriever=retriever)

        assert augmenter.retriever == retriever
        assert augmenter.model is None
        assert augmenter.max_passages == 5
        assert augmenter.max_queries == 3
        assert augmenter.query_temperature == 0.3
        assert augmenter.include_query_context is True

    def test_init_with_custom_parameters(self, mock_model) -> None:
        """Test initializing a RetrievalAugmenter with custom parameters."""
        retriever = MockRetriever()
        augmenter = RetrievalAugmenter(
            retriever=retriever,
            model=mock_model,
            max_passages=10,
            max_queries=5,
            query_temperature=0.5,
            include_query_context=False,
        )

        assert augmenter.retriever == retriever
        assert augmenter.model == mock_model
        assert augmenter.max_passages == 10
        assert augmenter.max_queries == 5
        assert augmenter.query_temperature == 0.5
        assert augmenter.include_query_context is False

    def test_init_without_retriever(self) -> None:
        """Test initializing a RetrievalAugmenter without a retriever."""
        with pytest.raises(RetrieverError) as excinfo:
            RetrievalAugmenter(retriever=None)

        assert "Retriever not provided" in str(excinfo.value)

    def test_retrieve_with_retriever_object(self) -> None:
        """Test retrieving passages with a Retriever object."""
        retriever = MockRetriever(["Passage A", "Passage B", "Passage C"])
        augmenter = RetrievalAugmenter(retriever=retriever)

        # Test retrieving with a single query
        result = augmenter.retrieve("Test text", custom_queries=["Test query"])

        # Check that the retriever was called correctly
        assert len(retriever.retrieve_calls) == 1
        assert retriever.retrieve_calls[0] == "Test query"

        # Check the result
        assert len(result) == 3
        assert "Query: Test query" in result[0]
        assert "Passage: Passage A" in result[0]
        assert "Query: Test query" in result[1]
        assert "Passage: Passage B" in result[1]
        assert "Query: Test query" in result[2]
        assert "Passage: Passage C" in result[2]

    def test_retrieve_with_function_retriever(self) -> None:
        """Test retrieving passages with a function-based retriever."""
        # Create a mock retriever class with a retrieve method
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = ["Passage X", "Passage Y", "Passage Z"]

        augmenter = RetrievalAugmenter(retriever=mock_retriever)

        # Test retrieving with a single query
        result = augmenter.retrieve("Test text", custom_queries=["Test query"])

        # Check that the retriever function was called correctly
        mock_retriever.retrieve.assert_called_once_with("Test query")

        # Check the result
        assert len(result) == 3
        assert "Query: Test query" in result[0]
        assert "Passage: Passage X" in result[0]
        assert "Query: Test query" in result[1]
        assert "Passage: Passage Y" in result[1]
        assert "Query: Test query" in result[2]
        assert "Passage: Passage Z" in result[2]

    def test_retrieve_without_query_context(self) -> None:
        """Test retrieving passages without including query context."""
        retriever = MockRetriever(["Passage A", "Passage B", "Passage C"])
        augmenter = RetrievalAugmenter(retriever=retriever, include_query_context=False)

        # Test retrieving with a single query
        result = augmenter.retrieve("Test text", custom_queries=["Test query"])

        # Check the result
        assert len(result) == 3
        assert result[0] == "Passage A"
        assert result[1] == "Passage B"
        assert result[2] == "Passage C"

    def test_retrieve_with_multiple_queries(self) -> None:
        """Test retrieving passages with multiple queries."""
        # Create a retriever that returns different passages for different queries
        retriever = MagicMock()
        retriever.retrieve.side_effect = [
            ["Passage A1", "Passage A2"],
            ["Passage B1", "Passage B2"],
            ["Passage C1", "Passage C2"],
        ]

        augmenter = RetrievalAugmenter(retriever=retriever)

        # Test retrieving with multiple queries
        result = augmenter.retrieve("Test text", custom_queries=["Query A", "Query B", "Query C"])

        # Check that the retriever was called correctly
        assert retriever.retrieve.call_count == 3
        retriever.retrieve.assert_any_call("Query A")
        retriever.retrieve.assert_any_call("Query B")
        retriever.retrieve.assert_any_call("Query C")

        # Check the result - the actual number of results may vary based on implementation
        # Just check that we have results for each query
        assert len(result) >= 3  # At least one passage per query

        # Check that we have passages from all queries
        query_a_found = False
        query_b_found = False
        query_c_found = False

        for passage in result:
            if "Query: Query A" in passage and "Passage: Passage A" in passage:
                query_a_found = True
            if "Query: Query B" in passage and "Passage: Passage B" in passage:
                query_b_found = True
            if "Query: Query C" in passage and "Passage: Passage C" in passage:
                query_c_found = True

        assert query_a_found, "No passages found for Query A"
        assert query_b_found, "No passages found for Query B"
        assert query_c_found, "No passages found for Query C"

    def test_retrieve_with_max_passages(self) -> None:
        """Test retrieving passages with a maximum number of passages."""
        # Create a retriever that returns many passages
        retriever = MockRetriever(["Passage 1", "Passage 2", "Passage 3", "Passage 4", "Passage 5"])
        augmenter = RetrievalAugmenter(retriever=retriever, max_passages=3)

        # Test retrieving with a single query
        result = augmenter.retrieve("Test text", custom_queries=["Test query"])

        # Check the result
        assert len(result) == 3  # Limited to max_passages
        assert "Passage 1" in result[0]
        assert "Passage 2" in result[1]
        assert "Passage 3" in result[2]

    def test_retrieve_with_max_queries(self) -> None:
        """Test retrieving passages with a maximum number of queries."""
        retriever = MockRetriever()
        augmenter = RetrievalAugmenter(retriever=retriever, max_queries=2)

        # Test retrieving with more queries than max_queries
        augmenter.retrieve("Test text", custom_queries=["Query A", "Query B", "Query C", "Query D"])

        # Check that only the first max_queries queries were used
        assert len(retriever.retrieve_calls) == 2
        assert retriever.retrieve_calls[0] == "Query A"
        assert retriever.retrieve_calls[1] == "Query B"

    def test_retrieve_with_retriever_error(self) -> None:
        """Test retrieving passages when the retriever raises an error."""
        # Create a retriever that raises an error for a specific query
        retriever = MagicMock()
        retriever.retrieve.side_effect = [
            ["Passage A1", "Passage A2"],
            Exception("Retriever error"),
            ["Passage C1", "Passage C2"],
        ]

        augmenter = RetrievalAugmenter(retriever=retriever)

        # Test retrieving with multiple queries
        result = augmenter.retrieve("Test text", custom_queries=["Query A", "Query B", "Query C"])

        # Check that the retriever was called for all queries
        assert retriever.retrieve.call_count == 3

        # Check the result (should only include passages from successful queries)
        assert len(result) == 4  # 2 passages from Query A, 2 from Query C
        assert "Query: Query A" in result[0]
        assert "Passage: Passage A1" in result[0]
        assert "Query: Query A" in result[1]
        assert "Passage: Passage A2" in result[1]
        assert "Query: Query C" in result[2]
        assert "Passage: Passage C1" in result[2]
        assert "Query: Query C" in result[3]
        assert "Passage: Passage C2" in result[3]

    def test_deduplicate_passages(self) -> None:
        """Test deduplicating passages."""
        retriever = MockRetriever()
        augmenter = RetrievalAugmenter(retriever=retriever)

        # Test deduplicating passages
        passages = [
            "Duplicate passage",
            "Unique passage 1",
            "Duplicate passage",  # Duplicate
            "Unique passage 2",
            "Duplicate passage",  # Duplicate
        ]

        result = augmenter._deduplicate_passages(passages)

        # Check the result
        assert len(result) == 3
        assert result[0] == "Duplicate passage"
        assert result[1] == "Unique passage 1"
        assert result[2] == "Unique passage 2"

    def test_format_passages(self) -> None:
        """Test formatting passages."""
        retriever = MockRetriever()
        augmenter = RetrievalAugmenter(retriever=retriever)

        # Test formatting passages
        passages = ["Passage A", "Passage B", "Passage C"]

        result = augmenter.format_passages(passages)

        # Check the result
        assert "Passage 1:\nPassage A" in result
        assert "Passage 2:\nPassage B" in result
        assert "Passage 3:\nPassage C" in result

    def test_generate_queries_with_model(self) -> None:
        """Test generating queries with a model."""
        retriever = MockRetriever()

        # Create a mock model with a generate method that returns a JSON response
        model_mock = MagicMock()
        model_mock.generate.return_value = json.dumps(
            {"queries": ["Generated query 1", "Generated query 2", "Generated query 3"]}
        )

        augmenter = RetrievalAugmenter(retriever=retriever, model=model_mock)

        # Test generating queries
        result = augmenter._generate_queries("Test text")

        # Check that the model was called correctly
        model_mock.generate.assert_called_once()

        # Check the result
        assert len(result) == 3
        assert result[0] == "Generated query 1"
        assert result[1] == "Generated query 2"
        assert result[2] == "Generated query 3"

    def test_generate_queries_with_model_invalid_json(self) -> None:
        """Test generating queries with a model that returns invalid JSON."""
        retriever = MockRetriever()

        # Create a mock model with a generate method that returns invalid JSON
        model_mock = MagicMock()
        model_mock.generate.return_value = "This is not valid JSON"

        augmenter = RetrievalAugmenter(retriever=retriever, model=model_mock)

        # Mock the _generate_heuristic_queries method
        with patch.object(augmenter, "_generate_heuristic_queries") as mock_heuristic:
            mock_heuristic.return_value = ["Heuristic query 1", "Heuristic query 2"]

            # Test generating queries
            result = augmenter._generate_queries("Test text")

            # Check that the model was called and then fell back to heuristic
            model_mock.generate.assert_called_once()
            mock_heuristic.assert_called_once_with("Test text")

            # Check the result
            assert len(result) == 2
            assert result[0] == "Heuristic query 1"
            assert result[1] == "Heuristic query 2"

    def test_generate_queries_with_model_empty_queries(self) -> None:
        """Test generating queries with a model that returns empty queries."""
        retriever = MockRetriever()

        # Create a mock model with a generate method that returns empty queries
        model_mock = MagicMock()
        model_mock.generate.return_value = json.dumps({"queries": []})

        augmenter = RetrievalAugmenter(retriever=retriever, model=model_mock)

        # Mock the _generate_heuristic_queries method
        with patch.object(augmenter, "_generate_heuristic_queries") as mock_heuristic:
            mock_heuristic.return_value = ["Heuristic query 1", "Heuristic query 2"]

            # Test generating queries
            result = augmenter._generate_queries("Test text")

            # Check that the model was called and then fell back to heuristic
            model_mock.generate.assert_called_once()
            mock_heuristic.assert_called_once_with("Test text")

            # Check the result
            assert len(result) == 2
            assert result[0] == "Heuristic query 1"
            assert result[1] == "Heuristic query 2"

    def test_generate_queries_with_model_error(self) -> None:
        """Test generating queries with a model that raises an error."""
        retriever = MockRetriever()

        # Create a mock model with a generate method that raises an error
        model_mock = MagicMock()
        model_mock.generate.side_effect = Exception("Model error")

        augmenter = RetrievalAugmenter(retriever=retriever, model=model_mock)

        # Mock the _generate_heuristic_queries method
        with patch.object(augmenter, "_generate_heuristic_queries") as mock_heuristic:
            mock_heuristic.return_value = ["Heuristic query 1", "Heuristic query 2"]

            # Test generating queries
            result = augmenter._generate_queries("Test text")

            # Check that the model was called and then fell back to heuristic
            model_mock.generate.assert_called_once()
            mock_heuristic.assert_called_once_with("Test text")

            # Check the result
            assert len(result) == 2
            assert result[0] == "Heuristic query 1"
            assert result[1] == "Heuristic query 2"

    def test_generate_heuristic_queries(self) -> None:
        """Test generating queries using the heuristic approach."""
        retriever = MockRetriever()
        augmenter = RetrievalAugmenter(retriever=retriever)

        # Test generating queries for a short text
        short_text = "This is a short test text."
        result = augmenter._generate_heuristic_queries(short_text)

        # Check the result
        assert len(result) >= 2
        assert "This is a short test text" in result  # First sentence
        assert short_text.strip() in result  # Entire text

        # Test generating queries for a long text
        long_text = "This is a long test text. " * 20  # More than 200 characters
        result = augmenter._generate_heuristic_queries(long_text)

        # Check the result
        assert len(result) >= 2
        assert "This is a long test text" in result[0]  # First sentence
        assert long_text[:200].strip() in result  # First 200 characters

    def test_get_retrieval_context(self) -> None:
        """Test getting retrieval context."""
        retriever = MockRetriever(["Passage A", "Passage B", "Passage C"])
        augmenter = RetrievalAugmenter(retriever=retriever)

        # Mock the _generate_queries method
        with patch.object(augmenter, "_generate_queries") as mock_generate_queries:
            mock_generate_queries.return_value = ["Generated query"]

            # Test getting retrieval context
            result = augmenter.get_retrieval_context("Test text")

            # Check that the methods were called correctly
            mock_generate_queries.assert_called_once_with("Test text")

            # Check the result
            assert "queries" in result
            assert result["queries"] == ["Generated query"]
            assert "passages" in result
            assert len(result["passages"]) == 3
            assert "Query: Generated query" in result["passages"][0]
            assert "Passage: Passage A" in result["passages"][0]
            assert "formatted_passages" in result
            assert "Passage 1:" in result["formatted_passages"]
            assert "passage_count" in result
            assert result["passage_count"] == 3

    def test_get_retrieval_context_with_custom_queries(self) -> None:
        """Test getting retrieval context with custom queries."""
        retriever = MockRetriever(["Passage A", "Passage B", "Passage C"])
        augmenter = RetrievalAugmenter(retriever=retriever)

        # Test getting retrieval context with custom queries
        result = augmenter.get_retrieval_context("Test text", custom_queries=["Custom query"])

        # Check the result
        assert "queries" in result
        assert result["queries"] == ["Custom query"]
        assert "passages" in result
        assert len(result["passages"]) == 3
        assert "Query: Custom query" in result["passages"][0]
        assert "Passage: Passage A" in result["passages"][0]
        assert "formatted_passages" in result
        assert "Passage 1:" in result["formatted_passages"]
        assert "passage_count" in result
        assert result["passage_count"] == 3

    def test_get_retrieval_context_with_error(self) -> None:
        """Test getting retrieval context when an error occurs."""
        retriever = MockRetriever()
        augmenter = RetrievalAugmenter(retriever=retriever)

        # Mock the retrieve method to raise an error
        with patch.object(augmenter, "retrieve") as mock_retrieve:
            mock_retrieve.side_effect = Exception("Retrieval error")

            # Test getting retrieval context
            result = augmenter.get_retrieval_context("Test text")

            # Check the result
            assert "queries" in result
            assert result["queries"] == []
            assert "passages" in result
            assert result["passages"] == []
            assert "formatted_passages" in result
            assert result["formatted_passages"] == ""
            assert "passage_count" in result
            assert result["passage_count"] == 0
            assert "error" in result
            assert "Retrieval error" in result["error"]
