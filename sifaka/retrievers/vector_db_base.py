"""Base vector database retriever interface and factory.

This module provides the base interface for vector database retrievers and
a factory function to create specific implementations. This allows Sifaka
to support multiple vector database providers (Milvus, Pinecone, Weaviate, etc.)
with a consistent interface.

Example:
    ```python
    from sifaka.retrievers.vector_db_base import create_vector_db_retriever

    # Create a Milvus retriever
    retriever = create_vector_db_retriever(
        provider="milvus",
        collection_name="documents",
        embedding_model="BAAI/bge-m3"
    )

    # Add documents
    retriever.add_document("doc1", "This is about AI.")

    # Retrieve similar documents
    results = retriever.retrieve("Tell me about artificial intelligence")
    ```
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from sifaka.core.thought import Thought


@runtime_checkable
class VectorDBRetriever(Protocol):
    """Protocol defining the interface for vector database retrievers.

    This protocol defines the minimum interface that all vector database
    retriever implementations must follow. It extends the basic Retriever
    protocol with vector database specific functionality.

    All vector database retrievers in Sifaka must implement this protocol.
    """

    @abstractmethod
    def add_document(
        self,
        doc_id: str,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add a document to the vector database.

        Args:
            doc_id: Unique identifier for the document.
            text: The document text.
            metadata: Optional metadata for the document.
        """
        ...

    @abstractmethod
    def retrieve(self, query: str) -> List[str]:
        """Retrieve relevant documents for a query.

        Args:
            query: The query to retrieve documents for.

        Returns:
            A list of relevant document texts.
        """
        ...

    @abstractmethod
    def retrieve_for_thought(self, thought: Thought, is_pre_generation: bool = True) -> Thought:
        """Retrieve documents and add them to a thought.

        Args:
            thought: The thought to add documents to.
            is_pre_generation: Whether this is pre-generation retrieval.

        Returns:
            The thought with retrieved documents added.
        """
        ...

    @abstractmethod
    def clear_collection(self) -> None:
        """Clear all documents from the collection."""
        ...

    @abstractmethod
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection.

        Returns:
            Dictionary with collection statistics.
        """
        ...

    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect from the vector database."""
        ...


class BaseVectorDBRetriever(ABC):
    """Abstract base class for vector database retrievers.

    This class provides common functionality that can be shared across
    different vector database implementations. Concrete implementations
    should inherit from this class and implement the abstract methods.

    Attributes:
        collection_name: Name of the collection/index.
        embedding_model: Name or path of the embedding model.
        dimension: Dimension of the embedding vectors.
        max_results: Maximum number of documents to return.
    """

    def __init__(
        self,
        collection_name: str,
        embedding_model: str,
        dimension: int,
        max_results: int = 3,
        **kwargs: Any,  # noqa: ARG002
    ):
        """Initialize the base vector DB retriever.

        Args:
            collection_name: Name of the collection/index.
            embedding_model: Name or path of the embedding model.
            dimension: Dimension of the embedding vectors.
            max_results: Maximum number of documents to return.
            **kwargs: Additional provider-specific parameters.
        """
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.dimension = dimension
        self.max_results = max_results

    @abstractmethod
    def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text.

        Args:
            text: The text to generate embedding for.

        Returns:
            The embedding vector as a list of floats.
        """
        ...

    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect from the vector database."""
        ...

    def __del__(self) -> None:
        """Cleanup when object is destroyed."""
        try:
            self.disconnect()
        except Exception:
            pass  # Ignore errors during cleanup


def create_vector_db_retriever(
    provider: str,
    **kwargs: Any,
) -> VectorDBRetriever:
    """Factory function to create vector database retrievers.

    Args:
        provider: The vector database provider ("milvus", "pinecone", "weaviate").
        **kwargs: Provider-specific configuration parameters.

    Returns:
        A vector database retriever instance.

    Raises:
        ValueError: If the provider is not supported.
        ImportError: If the required dependencies are not installed.

    Example:
        ```python
        # Create Milvus retriever
        milvus_retriever = create_vector_db_retriever(
            provider="milvus",
            collection_name="documents",
            embedding_model="BAAI/bge-m3"
        )

        # Create Pinecone retriever (future)
        pinecone_retriever = create_vector_db_retriever(
            provider="pinecone",
            index_name="documents",
            api_key="your-api-key"
        )
        ```
    """
    provider = provider.lower()

    if provider == "milvus":
        try:
            from sifaka.retrievers.milvus import MilvusRetriever

            return MilvusRetriever(**kwargs)
        except ImportError as e:
            raise ImportError(
                f"Milvus dependencies not available: {e}. "
                "Install with: pip install 'pymilvus[model]'"
            ) from e

    elif provider == "pinecone":
        # Future implementation
        raise NotImplementedError(
            "Pinecone retriever not yet implemented. " "Currently supported providers: milvus"
        )

    elif provider == "weaviate":
        # Future implementation
        raise NotImplementedError(
            "Weaviate retriever not yet implemented. " "Currently supported providers: milvus"
        )

    else:
        supported_providers = ["milvus", "pinecone", "weaviate"]
        raise ValueError(
            f"Unsupported vector database provider: {provider}. "
            f"Supported providers: {', '.join(supported_providers)}"
        )


# Convenience aliases for common configurations
def create_milvus_retriever(**kwargs: Any) -> VectorDBRetriever:
    """Create a Milvus vector database retriever.

    Args:
        **kwargs: Milvus-specific configuration parameters.

    Returns:
        A Milvus retriever instance.
    """
    return create_vector_db_retriever(provider="milvus", **kwargs)


# Future convenience functions
def create_pinecone_retriever(**kwargs: Any) -> VectorDBRetriever:
    """Create a Pinecone vector database retriever (not yet implemented).

    Args:
        **kwargs: Pinecone-specific configuration parameters.

    Returns:
        A Pinecone retriever instance.
    """
    return create_vector_db_retriever(provider="pinecone", **kwargs)


def create_weaviate_retriever(**kwargs: Any) -> VectorDBRetriever:
    """Create a Weaviate vector database retriever (not yet implemented).

    Args:
        **kwargs: Weaviate-specific configuration parameters.

    Returns:
        A Weaviate retriever instance.
    """
    return create_vector_db_retriever(provider="weaviate", **kwargs)
