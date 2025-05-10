"""
Tests for the retrieval component.

This module contains tests for the retrieval component in the Sifaka framework.
"""

import pytest

from sifaka.retrieval import (
    create_simple_retriever,
    create_threshold_retriever,
    RetrieverCore,
    RetrieverConfig,
    StringRetrievalResult,
)
from sifaka.retrieval.managers.query import QueryManager
from sifaka.retrieval.strategies.ranking import SimpleRankingStrategy, ScoreThresholdRankingStrategy


def test_retriever_core_initialization():
    """Test that RetrieverCore initializes correctly."""
    # Create a retriever core
    retriever = RetrieverCore()

    # Check that it has the expected properties
    assert retriever.name == "RetrieverCore"
    assert retriever.description == "Core retriever implementation for Sifaka"
    assert isinstance(retriever.config, RetrieverConfig)

    # Check that state is initialized
    assert retriever._state_manager.get("initialized") is False
    assert retriever._state_manager.get("execution_count") == 0

    # Initialize the retriever
    retriever.initialize()

    # Check that state is updated
    assert retriever._state_manager.get("initialized") is True


def test_simple_retriever():
    """Test that SimpleRetriever works correctly."""
    # Create a simple retriever with some documents
    documents = {
        "doc1": "This is a document about machine learning.",
        "doc2": "This document discusses natural language processing.",
        "doc3": "This is a document about artificial intelligence.",
    }
    retriever = create_simple_retriever(documents=documents, max_results=2)

    # Check that it has the expected properties
    assert retriever.name == "SimpleRetriever"
    assert len(retriever.documents) == 3

    # Retrieve information
    result = retriever.retrieve("machine learning")

    # Check the result
    assert isinstance(result, StringRetrievalResult)
    assert len(result.documents) <= 2  # max_results=2
    assert result.query == "machine learning"
    assert result.total_results <= 2

    # Check that state is updated
    assert retriever._state_manager.get("execution_count") == 1
    assert retriever._state_manager.get("last_query") == "machine learning"

    # Get statistics
    stats = retriever.get_statistics()
    assert stats["name"] == "SimpleRetriever"
    assert stats["execution_count"] == 1
    assert "avg_execution_time_ms" in stats


def test_threshold_retriever():
    """Test that ScoreThresholdRankingStrategy works correctly."""
    # Create documents
    documents = {
        "doc1": "This is a document about machine learning.",
        "doc2": "This document discusses natural language processing.",
        "doc3": "This is a document about artificial intelligence.",
        "doc4": "This document has nothing to do with the topic.",
    }

    # Create a threshold retriever
    retriever = create_threshold_retriever(
        documents=documents,
        max_results=3,
        threshold=0.5,
    )

    # Retrieve information
    result = retriever.retrieve("machine learning")

    # Check the result
    assert isinstance(result, StringRetrievalResult)
    assert result.query == "machine learning"

    # All documents should have scores >= 0.5
    for doc in result.documents:
        assert doc.score is not None
        assert doc.score >= 0.5

    # Get statistics
    stats = retriever.get_statistics()
    assert stats["name"] == "ThresholdRetriever"
    assert stats["execution_count"] == 1


def test_query_manager():
    """Test that QueryManager works correctly."""
    # Create a query manager
    query_manager = QueryManager()

    # Process a query
    processed_query = query_manager.process_query("How does Machine Learning work?")

    # Check that the query is processed
    assert processed_query.lower() == processed_query

    # Process another query
    _ = query_manager.process_query("How does Machine Learning work?")

    # Check that the cache is used
    assert query_manager._state_manager.get_metadata("cache_hit") is True

    # Get statistics
    stats = query_manager.get_statistics()
    assert stats["query_count"] == 2
    assert stats["cache_size"] == 1


def test_ranking_strategies():
    """Test that ranking strategies work correctly."""
    # Create documents
    documents = [
        {"content": "This is a document about machine learning."},
        {"content": "This document discusses natural language processing."},
        {"content": "This is a document about artificial intelligence."},
        {"content": "This document has nothing to do with the topic."},
    ]

    # Create a simple ranking strategy
    simple_strategy = SimpleRankingStrategy()

    # Rank documents
    ranked_docs = simple_strategy.rank("machine learning", documents)

    # Check that documents are ranked
    assert len(ranked_docs) > 0
    assert all("score" in doc for doc in ranked_docs)

    # Create a threshold strategy
    threshold_strategy = ScoreThresholdRankingStrategy(
        base_strategy=simple_strategy,
        threshold=0.5,
    )

    # Rank documents with threshold
    filtered_docs = threshold_strategy.rank("machine learning", documents)

    # Check that documents are filtered
    assert all(doc["score"] >= 0.5 for doc in filtered_docs)

    # Get statistics
    stats = threshold_strategy.get_statistics()
    assert stats["ranking_count"] == 1
    assert "filter_ratio" in stats


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
