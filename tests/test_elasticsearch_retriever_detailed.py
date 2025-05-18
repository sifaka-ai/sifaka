"""
Detailed tests for the Elasticsearch retriever.

This module contains more comprehensive tests for the Elasticsearch retriever
to improve test coverage.
"""

from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

from sifaka.errors import RetrieverError
from sifaka.retrievers.elasticsearch_retriever import ElasticsearchRetriever


# Mock embedding model for testing
def mock_embedding_model(text: str) -> List[float]:
    """Mock embedding model that returns a fixed vector."""
    return [0.1, 0.2, 0.3, 0.4, 0.5]


# Mock Elasticsearch response for testing
def create_mock_es_response(num_hits: int = 3) -> Dict[str, Any]:
    """Create a mock Elasticsearch response with the specified number of hits."""
    hits = []
    for i in range(num_hits):
        hits.append(
            {
                "_id": f"doc_{i}",
                "_score": 0.9 - (i * 0.1),
                "_source": {
                    "text": f"Document {i} content for testing",
                    "embedding": [0.1, 0.2, 0.3, 0.4, 0.5],
                },
            }
        )

    return {
        "took": 5,
        "timed_out": False,
        "_shards": {"total": 1, "successful": 1, "skipped": 0, "failed": 0},
        "hits": {
            "total": {"value": num_hits, "relation": "eq"},
            "max_score": 0.9,
            "hits": hits,
        },
    }


class TestElasticsearchRetrieverDetailed:
    """Detailed tests for the ElasticsearchRetriever."""

    def test_elasticsearch_not_available(self) -> None:
        """Test error handling when Elasticsearch is not available."""
        with patch("sifaka.retrievers.elasticsearch_retriever.ELASTICSEARCH_AVAILABLE", False):
            with pytest.raises(RetrieverError) as excinfo:
                ElasticsearchRetriever(
                    es_host="http://localhost:9200",
                    es_index="test_index",
                    embedding_model=mock_embedding_model,
                )

            assert "Elasticsearch is not available" in str(excinfo.value)
            assert "pip install elasticsearch" in str(excinfo.value)

    def test_init_connection_error(self) -> None:
        """Test error handling when connection to Elasticsearch fails."""
        with patch("sifaka.retrievers.elasticsearch_retriever.ELASTICSEARCH_AVAILABLE", True):
            with patch("sifaka.retrievers.elasticsearch_retriever.Elasticsearch") as mock_es:
                # Mock Elasticsearch client that fails to ping
                mock_client = MagicMock()
                mock_client.ping.return_value = False
                mock_es.return_value = mock_client

                with pytest.raises(RetrieverError) as excinfo:
                    ElasticsearchRetriever(
                        es_host="http://localhost:9200",
                        es_index="test_index",
                        embedding_model=mock_embedding_model,
                    )

                assert "Failed to connect to Elasticsearch" in str(excinfo.value)

    def test_init_exception(self) -> None:
        """Test error handling when Elasticsearch client initialization raises an exception."""
        with patch("sifaka.retrievers.elasticsearch_retriever.ELASTICSEARCH_AVAILABLE", True):
            with patch("sifaka.retrievers.elasticsearch_retriever.Elasticsearch") as mock_es:
                # Mock Elasticsearch client that raises an exception
                mock_es.side_effect = Exception("Connection refused")

                with pytest.raises(RetrieverError) as excinfo:
                    ElasticsearchRetriever(
                        es_host="http://localhost:9200",
                        es_index="test_index",
                        embedding_model=mock_embedding_model,
                    )

                assert "Error connecting to Elasticsearch" in str(excinfo.value)
                assert "Connection refused" in str(excinfo.value)

    def test_init_index_not_exists(self) -> None:
        """Test error handling when the specified index does not exist."""
        with patch("sifaka.retrievers.elasticsearch_retriever.ELASTICSEARCH_AVAILABLE", True):
            with patch("sifaka.retrievers.elasticsearch_retriever.Elasticsearch") as mock_es:
                # Mock Elasticsearch client
                mock_client = MagicMock()
                mock_client.ping.return_value = True
                mock_client.indices.exists.return_value = False
                mock_es.return_value = mock_client

                with pytest.raises(RetrieverError) as excinfo:
                    ElasticsearchRetriever(
                        es_host="http://localhost:9200",
                        es_index="nonexistent_index",
                        embedding_model=mock_embedding_model,
                    )

                assert "Elasticsearch index 'nonexistent_index' does not exist" in str(
                    excinfo.value
                )

    def test_init_success(self) -> None:
        """Test successful initialization of the retriever."""
        with patch("sifaka.retrievers.elasticsearch_retriever.ELASTICSEARCH_AVAILABLE", True):
            with patch("sifaka.retrievers.elasticsearch_retriever.Elasticsearch") as mock_es:
                # Mock Elasticsearch client
                mock_client = MagicMock()
                mock_client.ping.return_value = True
                mock_client.indices.exists.return_value = True
                mock_es.return_value = mock_client

                retriever = ElasticsearchRetriever(
                    es_host="http://localhost:9200",
                    es_index="test_index",
                    embedding_model=mock_embedding_model,
                    hybrid_search=True,
                    top_k=5,
                    text_field="content",
                    embedding_field="vector",
                )

                assert retriever.es_client == mock_client
                assert retriever.es_index == "test_index"
                assert retriever.embedding_model == mock_embedding_model
                assert retriever.hybrid_search is True
                assert retriever.top_k == 5
                assert retriever.text_field == "content"
                assert retriever.embedding_field == "vector"

    def test_semantic_search(self) -> None:
        """Test semantic search functionality."""
        with patch("sifaka.retrievers.elasticsearch_retriever.ELASTICSEARCH_AVAILABLE", True):
            with patch("sifaka.retrievers.elasticsearch_retriever.Elasticsearch") as mock_es:
                # Mock Elasticsearch client
                mock_client = MagicMock()
                mock_client.ping.return_value = True
                mock_client.indices.exists.return_value = True
                mock_client.search.return_value = create_mock_es_response(3)
                mock_es.return_value = mock_client

                retriever = ElasticsearchRetriever(
                    es_host="http://localhost:9200",
                    es_index="test_index",
                    embedding_model=mock_embedding_model,
                    hybrid_search=False,
                    top_k=3,
                    text_field="text",
                    embedding_field="embedding",
                )

                # Test semantic search
                results = retriever._semantic_search("test query")

                # Check that search was called with correct parameters
                mock_client.search.assert_called_once()
                call_args = mock_client.search.call_args[1]
                assert call_args["index"] == "test_index"
                assert "knn" in call_args["body"]
                assert call_args["body"]["knn"]["field"] == "embedding"
                assert call_args["body"]["knn"]["k"] == 3

                # Check results
                assert len(results) == 3
                assert results[0] == "Document 0 content for testing"
                assert results[1] == "Document 1 content for testing"
                assert results[2] == "Document 2 content for testing"

    def test_hybrid_search(self) -> None:
        """Test hybrid search functionality."""
        with patch("sifaka.retrievers.elasticsearch_retriever.ELASTICSEARCH_AVAILABLE", True):
            with patch("sifaka.retrievers.elasticsearch_retriever.Elasticsearch") as mock_es:
                # Mock Elasticsearch client
                mock_client = MagicMock()
                mock_client.ping.return_value = True
                mock_client.indices.exists.return_value = True
                mock_client.search.return_value = create_mock_es_response(3)
                mock_es.return_value = mock_client

                retriever = ElasticsearchRetriever(
                    es_host="http://localhost:9200",
                    es_index="test_index",
                    embedding_model=mock_embedding_model,
                    hybrid_search=True,
                    top_k=3,
                    text_field="text",
                    embedding_field="embedding",
                )

                # Test hybrid search
                results = retriever._hybrid_search("test query")

                # Check that search was called with correct parameters
                mock_client.search.assert_called_once()
                call_args = mock_client.search.call_args[1]
                assert call_args["index"] == "test_index"
                assert "query" in call_args["body"]
                assert "bool" in call_args["body"]["query"]
                assert "knn" in call_args["body"]
                assert call_args["body"]["knn"]["field"] == "embedding"
                assert call_args["body"]["knn"]["k"] == 3
                assert call_args["body"]["size"] == 3

                # Check results
                assert len(results) == 3
                assert results[0] == "Document 0 content for testing"
                assert results[1] == "Document 1 content for testing"
                assert results[2] == "Document 2 content for testing"

    def test_retrieve_with_semantic_search(self) -> None:
        """Test retrieve method with semantic search."""
        with patch("sifaka.retrievers.elasticsearch_retriever.ELASTICSEARCH_AVAILABLE", True):
            with patch("sifaka.retrievers.elasticsearch_retriever.Elasticsearch") as mock_es:
                # Mock Elasticsearch client
                mock_client = MagicMock()
                mock_client.ping.return_value = True
                mock_client.indices.exists.return_value = True
                mock_client.search.return_value = create_mock_es_response(3)
                mock_es.return_value = mock_client

                retriever = ElasticsearchRetriever(
                    es_host="http://localhost:9200",
                    es_index="test_index",
                    embedding_model=mock_embedding_model,
                    hybrid_search=False,
                    top_k=3,
                )

                # Mock _semantic_search method
                retriever._semantic_search = MagicMock(
                    return_value=[
                        "Document 0 content for testing",
                        "Document 1 content for testing",
                        "Document 2 content for testing",
                    ]
                )

                # Test retrieve method
                results = retriever.retrieve("test query")

                # Check that _semantic_search was called
                retriever._semantic_search.assert_called_once_with("test query")

                # Check results
                assert len(results) == 3
                assert results[0] == "Document 0 content for testing"
                assert results[1] == "Document 1 content for testing"
                assert results[2] == "Document 2 content for testing"

    def test_retrieve_with_hybrid_search(self) -> None:
        """Test retrieve method with hybrid search."""
        with patch("sifaka.retrievers.elasticsearch_retriever.ELASTICSEARCH_AVAILABLE", True):
            with patch("sifaka.retrievers.elasticsearch_retriever.Elasticsearch") as mock_es:
                # Mock Elasticsearch client
                mock_client = MagicMock()
                mock_client.ping.return_value = True
                mock_client.indices.exists.return_value = True
                mock_client.search.return_value = create_mock_es_response(3)
                mock_es.return_value = mock_client

                retriever = ElasticsearchRetriever(
                    es_host="http://localhost:9200",
                    es_index="test_index",
                    embedding_model=mock_embedding_model,
                    hybrid_search=True,
                    top_k=3,
                )

                # Mock _hybrid_search method
                retriever._hybrid_search = MagicMock(
                    return_value=[
                        "Document 0 content for testing",
                        "Document 1 content for testing",
                        "Document 2 content for testing",
                    ]
                )

                # Test retrieve method
                results = retriever.retrieve("test query")

                # Check that _hybrid_search was called
                retriever._hybrid_search.assert_called_once_with("test query")

                # Check results
                assert len(results) == 3
                assert results[0] == "Document 0 content for testing"
                assert results[1] == "Document 1 content for testing"
                assert results[2] == "Document 2 content for testing"

    def test_retrieve_error(self) -> None:
        """Test error handling in retrieve method."""
        with patch("sifaka.retrievers.elasticsearch_retriever.ELASTICSEARCH_AVAILABLE", True):
            with patch("sifaka.retrievers.elasticsearch_retriever.Elasticsearch") as mock_es:
                # Mock Elasticsearch client
                mock_client = MagicMock()
                mock_client.ping.return_value = True
                mock_client.indices.exists.return_value = True
                mock_es.return_value = mock_client

                retriever = ElasticsearchRetriever(
                    es_host="http://localhost:9200",
                    es_index="test_index",
                    embedding_model=mock_embedding_model,
                )

                # We need to mock both _semantic_search and _hybrid_search methods
                # since the retriever might use either one depending on the hybrid_search flag
                retriever._semantic_search = MagicMock(side_effect=Exception("Search failed"))
                retriever._hybrid_search = MagicMock(side_effect=Exception("Search failed"))

                # Test retrieve method with error
                with pytest.raises(RetrieverError) as excinfo:
                    retriever.retrieve("test query")

                assert "Error retrieving documents" in str(excinfo.value)
                assert "Search failed" in str(excinfo.value)
