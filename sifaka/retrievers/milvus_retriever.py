"""
Milvus retriever for Sifaka.

This module provides a retriever that uses Milvus for document retrieval.
"""

import logging
from typing import TYPE_CHECKING, Any, Callable, Dict, List

# Import Milvus conditionally to avoid hard dependency
try:
    from pymilvus import Collection, connections  # type: ignore

    MILVUS_AVAILABLE = True
except ImportError:
    MILVUS_AVAILABLE = False
    # Define stub types for type checking when pymilvus is not available
    if TYPE_CHECKING:

        class Collection:  # type: ignore
            def __init__(self, name: str) -> None: ...
            def load(self) -> None: ...
            def search(
                self,
                data: List[List[float]],
                anns_field: str,
                param: Dict[str, Any],
                limit: int,
                output_fields: List[str],
            ) -> List[List[Dict[str, Any]]]: ...


from sifaka.errors import RetrieverError
from sifaka.retrievers.base import Retriever

logger = logging.getLogger(__name__)


class MilvusRetriever(Retriever):
    """Retriever that uses Milvus for document retrieval.

    This retriever connects to a Milvus instance and performs
    vector search to retrieve relevant documents.

    Attributes:
        collection: The Milvus collection to search.
        embedding_model: Model or function to generate embeddings for queries.
        text_field: The field containing document text.
        embedding_field: The field containing document embeddings.
        top_k: Number of documents to retrieve.
        metric_type: The metric type to use for similarity search.
    """

    # Type annotations for instance variables
    collection: Any  # Collection from pymilvus
    embedding_model: Callable[[str], List[float]]
    text_field: str
    embedding_field: str
    top_k: int
    metric_type: str

    def __init__(
        self,
        milvus_host: str,
        milvus_port: str,
        collection_name: str,
        embedding_model: Callable[[str], List[float]],
        text_field: str = "text",
        embedding_field: str = "embedding",
        top_k: int = 5,
        metric_type: str = "COSINE",
        **kwargs: Any,
    ):
        """Initialize the Milvus retriever.

        Args:
            milvus_host: The Milvus host.
            milvus_port: The Milvus port.
            collection_name: The Milvus collection to search.
            embedding_model: Model or function to generate embeddings for queries.
            text_field: The field containing document text.
            embedding_field: The field containing document embeddings.
            top_k: Number of documents to retrieve.
            metric_type: The metric type to use for similarity search.
            **kwargs: Additional arguments to pass to the Milvus connection.

        Raises:
            RetrieverError: If Milvus is not available or connection fails.
        """
        if not MILVUS_AVAILABLE:
            raise RetrieverError(
                "Milvus is not available. Please install it with: pip install pymilvus"
            )

        super().__init__()

        try:
            connections.connect(host=milvus_host, port=milvus_port, **kwargs)
            self.collection = Collection(collection_name)
        except Exception as e:
            raise RetrieverError(f"Error connecting to Milvus: {str(e)}")

        self.embedding_model = embedding_model
        self.text_field = text_field
        self.embedding_field = embedding_field
        self.top_k = top_k
        self.metric_type = metric_type

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
            # Generate embedding for the query
            query_embedding = self.embedding_model(query)

            # Perform vector search in Milvus
            self.collection.load()
            search_params = {
                "metric_type": self.metric_type,
                "params": {"nprobe": 10},
            }

            results = self.collection.search(
                data=[query_embedding],
                anns_field=self.embedding_field,
                param=search_params,
                limit=self.top_k,
                output_fields=[self.text_field],
            )

            # Extract document texts from the results
            documents = [hit[self.text_field] for hit in results[0]]
            return documents
        except Exception as e:
            logger.error(f"Error retrieving documents: {str(e)}")
            raise RetrieverError(f"Error retrieving documents: {str(e)}")
