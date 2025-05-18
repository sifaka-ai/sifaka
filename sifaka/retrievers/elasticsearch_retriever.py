"""
Elasticsearch retriever for Sifaka.

This module provides a retriever that uses Elasticsearch for document retrieval.
"""

from typing import List, Dict, Any, Optional, Callable, Union
import logging

# Import Elasticsearch conditionally to avoid hard dependency
try:
    from elasticsearch import Elasticsearch
    ELASTICSEARCH_AVAILABLE = True
except ImportError:
    ELASTICSEARCH_AVAILABLE = False

from sifaka.retrievers.base import Retriever
from sifaka.errors import RetrieverError

logger = logging.getLogger(__name__)


class ElasticsearchRetriever(Retriever):
    """Retriever that uses Elasticsearch for document retrieval.

    This retriever connects to an Elasticsearch instance and performs
    vector search or hybrid search to retrieve relevant documents.

    Attributes:
        es_client: The Elasticsearch client.
        es_index: The Elasticsearch index to search.
        embedding_model: Model or function to generate embeddings for queries.
        hybrid_search: Whether to use hybrid search (combining keyword and semantic).
        top_k: Number of documents to retrieve.
        text_field: The field containing document text.
        embedding_field: The field containing document embeddings.
    """

    def __init__(
        self,
        es_host: str,
        es_index: str,
        embedding_model: Callable[[str], List[float]],
        hybrid_search: bool = True,
        top_k: int = 5,
        text_field: str = "text",
        embedding_field: str = "embedding",
        **kwargs: Any,
    ):
        """Initialize the Elasticsearch retriever.

        Args:
            es_host: The Elasticsearch host URL.
            es_index: The Elasticsearch index to search.
            embedding_model: Model or function to generate embeddings for queries.
            hybrid_search: Whether to use hybrid search (combining keyword and semantic).
            top_k: Number of documents to retrieve.
            text_field: The field containing document text.
            embedding_field: The field containing document embeddings.
            **kwargs: Additional arguments to pass to the Elasticsearch client.

        Raises:
            RetrieverError: If Elasticsearch is not available or connection fails.
        """
        if not ELASTICSEARCH_AVAILABLE:
            raise RetrieverError(
                "Elasticsearch is not available. Please install it with: pip install elasticsearch"
            )

        super().__init__()

        try:
            self.es_client = Elasticsearch(es_host, **kwargs)
            # Check connection
            if not self.es_client.ping():
                raise RetrieverError(f"Failed to connect to Elasticsearch at {es_host}")
        except Exception as e:
            raise RetrieverError(f"Error connecting to Elasticsearch: {str(e)}")

        self.es_index = es_index
        self.embedding_model = embedding_model
        self.hybrid_search = hybrid_search
        self.top_k = top_k
        self.text_field = text_field
        self.embedding_field = embedding_field

        # Verify index exists
        if not self.es_client.indices.exists(index=self.es_index):
            raise RetrieverError(f"Elasticsearch index '{self.es_index}' does not exist")

    def retrieve(self, query: str) -> List[str]:
        """Retrieve relevant documents for a query.

        Args:
            query: The query to retrieve documents for.

        Returns:
            A list of relevant document texts.

        Raises:
            RetrieverError: If retrieval fails.
        """
        try:
            if self.hybrid_search:
                return self._hybrid_search(query)
            else:
                return self._semantic_search(query)
        except Exception as e:
            logger.error(f"Error retrieving documents: {str(e)}")
            raise RetrieverError(f"Error retrieving documents: {str(e)}")

    def _semantic_search(self, query: str) -> List[str]:
        """Perform semantic search using embeddings.

        Args:
            query: The query to search for.

        Returns:
            A list of relevant document texts.
        """
        # Generate embedding for the query
        query_embedding = self.embedding_model(query)
        
        # Perform vector search in Elasticsearch
        search_query = {
            "knn": {
                "field": self.embedding_field,
                "query_vector": query_embedding,
                "k": self.top_k,
                "num_candidates": self.top_k * 2
            },
            "_source": [self.text_field]
        }
        
        response = self.es_client.search(
            index=self.es_index,
            body=search_query
        )
        
        # Extract document texts from the response
        documents = [hit["_source"][self.text_field] for hit in response["hits"]["hits"]]
        return documents

    def _hybrid_search(self, query: str) -> List[str]:
        """Perform hybrid search combining keyword and semantic search.

        Args:
            query: The query to search for.

        Returns:
            A list of relevant document texts.
        """
        # Generate embedding for the query
        query_embedding = self.embedding_model(query)
        
        # Perform hybrid search in Elasticsearch
        search_query = {
            "query": {
                "bool": {
                    "should": [
                        {
                            "match": {
                                self.text_field: {
                                    "query": query,
                                    "boost": 0.3
                                }
                            }
                        }
                    ]
                }
            },
            "knn": {
                "field": self.embedding_field,
                "query_vector": query_embedding,
                "k": self.top_k,
                "num_candidates": self.top_k * 2,
                "boost": 0.7
            },
            "_source": [self.text_field],
            "size": self.top_k
        }
        
        response = self.es_client.search(
            index=self.es_index,
            body=search_query
        )
        
        # Extract document texts from the response
        documents = [hit["_source"][self.text_field] for hit in response["hits"]["hits"]]
        return documents
