"""
Pytest fixtures for retrieval tests.
"""

import pytest
from sifaka.utils.config.retrieval import RetrieverConfig


@pytest.fixture
def retriever_config():
    """Fixture for a retriever configuration."""
    config = RetrieverConfig(
        max_results=3,
        min_score=0.1,
    )
    return config


@pytest.fixture
def test_documents():
    """Fixture for test documents."""
    return {
        "doc1": "This is a document about cats.",
        "doc2": "This is a document about dogs.",
        "doc3": "This is a document about birds.",
    }


@pytest.fixture
def test_query():
    """Fixture for a test query."""
    return "cats"


@pytest.fixture
def empty_query():
    """Fixture for an empty query."""
    return ""


@pytest.fixture
def no_results_query():
    """Fixture for a query that should return no results."""
    return "zebras"
