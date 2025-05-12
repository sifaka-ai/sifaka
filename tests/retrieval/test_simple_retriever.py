"""
Tests for the SimpleRetriever.
"""

import pytest
from sifaka.retrieval.implementations.simple import SimpleRetriever
from sifaka.retrieval.factories import create_simple_retriever, create_threshold_retriever
from sifaka.utils.config import RetrieverConfig


def test_simple_retriever_initialization():
    """Test that the SimpleRetriever can be initialized."""
    documents = {
        "doc1": "This is document 1.",
        "doc2": "This is document 2.",
    }

    retriever = SimpleRetriever(
        name="test_simple_retriever",
        description="Test simple retriever",
        documents=documents,
        config=RetrieverConfig(
            max_results=3,
            min_score=0.1,
        ),
    )

    assert retriever is not None
    assert retriever.name == "test_simple_retriever"
    assert retriever.description == "Test simple retriever"
    assert retriever._state_manager.get("documents") == documents
    assert retriever._state_manager.get("max_results") == 3
    assert retriever._state_manager.get("min_score") == 0.1


def test_simple_retriever_factory():
    """Test that the simple retriever factory works."""
    documents = {
        "doc1": "This is document 1.",
        "doc2": "This is document 2.",
    }

    retriever = create_simple_retriever(
        documents=documents,
        max_results=3,
        min_score=0.1,
        name="test_simple_retriever",
        description="Test simple retriever",
    )

    assert retriever is not None
    assert retriever.name == "test_simple_retriever"
    assert retriever.description == "Test simple retriever"
    assert retriever._state_manager.get("documents") == documents
    assert retriever._state_manager.get("max_results") == 3
    assert retriever._state_manager.get("min_score") == 0.1


def test_simple_retriever_retrieve():
    """Test that the SimpleRetriever retrieves documents correctly."""
    documents = {
        "doc1": "This is a document about cats.",
        "doc2": "This is a document about dogs.",
        "doc3": "This is a document about birds.",
    }

    retriever = create_simple_retriever(
        documents=documents,
        max_results=2,
    )

    result = retriever.retrieve("cats")

    assert result is not None
    assert len(result.documents) <= 2  # Should respect max_results

    # The document about cats should be first
    assert "cats" in result.documents[0].content.lower()

    # Check that the result has the expected structure
    assert hasattr(result, "query")
    assert result.query == "cats"
    assert hasattr(result, "documents")
    assert len(result.documents) > 0
    assert hasattr(result.documents[0], "metadata")
    assert hasattr(result.documents[0], "content")
    assert hasattr(result.documents[0], "score")


def test_simple_retriever_empty_query():
    """Test that the SimpleRetriever handles empty queries."""
    documents = {
        "doc1": "This is document 1.",
        "doc2": "This is document 2.",
    }

    retriever = create_simple_retriever(
        documents=documents,
        max_results=3,
    )

    result = retriever.retrieve("")

    assert result is not None
    assert len(result.documents) == 0


def test_simple_retriever_no_results():
    """Test that the SimpleRetriever handles queries with no results."""
    documents = {
        "doc1": "This is a document about cats.",
        "doc2": "This is a document about dogs.",
    }

    # Use create_threshold_retriever instead of create_simple_retriever
    retriever = create_threshold_retriever(
        documents=documents,
        max_results=3,
        threshold=0.9,  # High threshold to ensure no results
    )

    result = retriever.retrieve("zebras")

    assert result is not None
    assert len(result.documents) == 0
