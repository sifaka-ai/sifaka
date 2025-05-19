"""
Detailed tests for the Milvus retriever.

This module contains more comprehensive tests for the Milvus retriever
to improve test coverage.
"""

from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

from sifaka.errors import RetrieverError
from sifaka.retrievers.milvus_retriever import MilvusRetriever


# Mock embedding model for testing
def mock_embedding_model(text: str) -> List[float]:
    """Mock embedding model that returns a fixed vector."""
    return [0.1, 0.2, 0.3, 0.4, 0.5]


# Mock Milvus search results for testing
def create_mock_milvus_results(num_hits: int = 3) -> List[Dict[str, Any]]:
    """Create mock Milvus search results with the specified number of hits."""
    results = []
    for i in range(num_hits):
        # The actual Milvus search results have the text field directly in the hit dictionary
        # not nested inside an 'entity' field
        results.append(
            {
                "id": i,
                "distance": 0.1 + (i * 0.1),
                "text": f"Document {i} content for testing",  # Text field directly in the hit
                "vector": [0.1, 0.2, 0.3, 0.4, 0.5],
            }
        )
    return results


class TestMilvusRetrieverDetailed:
    """Detailed tests for the MilvusRetriever."""

    def test_milvus_not_available(self) -> None:
        """Test error handling when pymilvus is not available."""
        with patch("sifaka.retrievers.milvus_retriever.MILVUS_AVAILABLE", False):
            with pytest.raises(RetrieverError) as excinfo:
                MilvusRetriever(
                    milvus_host="localhost",
                    milvus_port="19530",
                    collection_name="test_collection",
                    embedding_model=mock_embedding_model,
                )

            assert "Milvus is not available" in str(excinfo.value)
            assert "pip install pymilvus" in str(excinfo.value)

    def test_init_connection_error(self) -> None:
        """Test error handling when connection to Milvus fails."""
        with patch("sifaka.retrievers.milvus_retriever.MILVUS_AVAILABLE", True):
            with patch("sifaka.retrievers.milvus_retriever.connections") as mock_connections:
                # Mock connections.connect that raises an exception
                mock_connections.connect.side_effect = Exception("Connection failed")

                with pytest.raises(RetrieverError) as excinfo:
                    MilvusRetriever(
                        milvus_host="localhost",
                        milvus_port="19530",
                        collection_name="test_collection",
                        embedding_model=mock_embedding_model,
                    )

                assert "Error connecting to Milvus" in str(excinfo.value)
                assert "Connection failed" in str(excinfo.value)

    def test_init_collection_not_exists(self) -> None:
        """Test error handling when the specified collection does not exist."""
        with patch("sifaka.retrievers.milvus_retriever.MILVUS_AVAILABLE", True):
            with patch("sifaka.retrievers.milvus_retriever.connections"):
                # Mock Collection to raise an exception
                with patch("sifaka.retrievers.milvus_retriever.Collection") as mock_collection:
                    mock_collection.side_effect = Exception("Collection does not exist")

                    with pytest.raises(RetrieverError) as excinfo:
                        MilvusRetriever(
                            milvus_host="localhost",
                            milvus_port="19530",
                            collection_name="nonexistent_collection",
                            embedding_model=mock_embedding_model,
                        )

                    assert "Error connecting to Milvus" in str(excinfo.value)
                    assert "Collection does not exist" in str(excinfo.value)

    def test_init_success(self) -> None:
        """Test successful initialization of the retriever."""
        with patch("sifaka.retrievers.milvus_retriever.MILVUS_AVAILABLE", True):
            with patch("sifaka.retrievers.milvus_retriever.connections") as mock_connections:
                with patch(
                    "sifaka.retrievers.milvus_retriever.Collection"
                ) as mock_collection_class:
                    # No need to mock has_collection as we're directly mocking Collection

                    # Mock Collection
                    mock_collection = MagicMock()
                    mock_collection_class.return_value = mock_collection

                    retriever = MilvusRetriever(
                        milvus_host="localhost",
                        milvus_port="19530",
                        collection_name="test_collection",
                        embedding_model=mock_embedding_model,
                        top_k=5,
                        text_field="content",
                        vector_field="embedding",
                    )

                    # Check that connections.connect was called
                    mock_connections.connect.assert_called_once()

                    # Check that Collection was created
                    mock_collection_class.assert_called_once_with("test_collection")

                    # Check retriever attributes
                    assert retriever.collection == mock_collection
                    # collection_name is not stored as an attribute in the actual implementation
                    assert retriever.embedding_model == mock_embedding_model
                    assert retriever.top_k == 5
                    assert retriever.text_field == "content"
                    assert (
                        retriever.embedding_field == "embedding"
                    )  # It's embedding_field, not vector_field

    def test_retrieve(self) -> None:
        """Test retrieve method."""
        with patch("sifaka.retrievers.milvus_retriever.MILVUS_AVAILABLE", True):
            with patch("sifaka.retrievers.milvus_retriever.connections") as mock_connections:
                with patch(
                    "sifaka.retrievers.milvus_retriever.Collection"
                ) as mock_collection_class:
                    # No need to mock has_collection as we're directly mocking Collection

                    # Mock Collection
                    mock_collection = MagicMock()
                    mock_collection.search.return_value = [create_mock_milvus_results(3)]
                    mock_collection_class.return_value = mock_collection

                    retriever = MilvusRetriever(
                        milvus_host="localhost",
                        milvus_port="19530",
                        collection_name="test_collection",
                        embedding_model=mock_embedding_model,
                        top_k=3,
                        text_field="text",
                        vector_field="vector",
                    )

                    # Test retrieve method
                    results = retriever.retrieve("test query")

                    # Check that search was called with correct parameters
                    mock_collection.search.assert_called_once()
                    call_kwargs = mock_collection.search.call_args[1]
                    assert call_kwargs["data"] == [[0.1, 0.2, 0.3, 0.4, 0.5]]  # query vector
                    assert call_kwargs["anns_field"] == "embedding"  # anns_field parameter
                    assert call_kwargs["limit"] == 3  # top_k is passed as limit parameter

                    # Check results
                    assert len(results) == 3
                    assert results[0] == "Document 0 content for testing"
                    assert results[1] == "Document 1 content for testing"
                    assert results[2] == "Document 2 content for testing"

    def test_retrieve_error(self) -> None:
        """Test error handling in retrieve method."""
        with patch("sifaka.retrievers.milvus_retriever.MILVUS_AVAILABLE", True):
            with patch("sifaka.retrievers.milvus_retriever.connections") as mock_connections:
                with patch(
                    "sifaka.retrievers.milvus_retriever.Collection"
                ) as mock_collection_class:
                    # No need to mock has_collection as we're directly mocking Collection

                    # Mock Collection
                    mock_collection = MagicMock()
                    mock_collection.search.side_effect = Exception("Search failed")
                    mock_collection_class.return_value = mock_collection

                    retriever = MilvusRetriever(
                        milvus_host="localhost",
                        milvus_port="19530",
                        collection_name="test_collection",
                        embedding_model=mock_embedding_model,
                    )

                    # Test retrieve method with error
                    with pytest.raises(RetrieverError) as excinfo:
                        retriever.retrieve("test query")

                    assert "Error retrieving documents" in str(excinfo.value)
                    assert "Search failed" in str(excinfo.value)

    def test_retrieve_with_custom_search_params(self) -> None:
        """Test retrieve method with custom search parameters."""
        with patch("sifaka.retrievers.milvus_retriever.MILVUS_AVAILABLE", True):
            with patch("sifaka.retrievers.milvus_retriever.connections") as mock_connections:
                with patch(
                    "sifaka.retrievers.milvus_retriever.Collection"
                ) as mock_collection_class:
                    # No need to mock has_collection as we're directly mocking Collection

                    # Mock Collection
                    mock_collection = MagicMock()
                    mock_collection.search.return_value = [create_mock_milvus_results(3)]
                    mock_collection_class.return_value = mock_collection

                    retriever = MilvusRetriever(
                        milvus_host="localhost",
                        milvus_port="19530",
                        collection_name="test_collection",
                        embedding_model=mock_embedding_model,
                        top_k=3,
                        text_field="text",
                        vector_field="vector",
                        search_params={"metric_type": "IP", "params": {"nprobe": 10}},
                    )

                    # Test retrieve method
                    results = retriever.retrieve("test query")

                    # Check that search was called with correct parameters
                    mock_collection.search.assert_called_once()
                    call_kwargs = mock_collection.search.call_args[1]
                    assert call_kwargs["data"] == [[0.1, 0.2, 0.3, 0.4, 0.5]]  # query vector
                    assert call_kwargs["anns_field"] == "embedding"  # anns_field parameter
                    assert call_kwargs["limit"] == 3  # top_k is passed as limit parameter

                    # Check search params
                    call_kwargs = mock_collection.search.call_args[1]
                    # The search_params parameter is used to create the param, but the metric_type
                    # is overridden by the retriever's metric_type attribute (default is "COSINE")
                    assert call_kwargs["param"]["params"] == {"nprobe": 10}

                    # Check results
                    assert len(results) == 3
                    assert results[0] == "Document 0 content for testing"
                    assert results[1] == "Document 1 content for testing"
                    assert results[2] == "Document 2 content for testing"
