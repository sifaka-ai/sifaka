"""Retrievers for Sifaka.

This package provides retriever implementations for finding relevant documents
and context for text generation and improvement. Retrievers are used by the
Chain to provide context to models and critics.

Available retrievers:
- MockRetriever: Returns predefined documents for testing
- InMemoryRetriever: Simple keyword-based retrieval from in-memory documents
- RedisRetriever: Redis-based caching retriever with optional base retriever

Vector Database Retrievers:
- MilvusRetriever: Milvus-based semantic search using vector embeddings
- create_vector_db_retriever: Factory function for vector database retrievers

Example:
    ```python
    from sifaka.retrievers import MockRetriever, InMemoryRetriever, RedisRetriever
    from sifaka.retrievers.vector_db_base import create_vector_db_retriever
    from sifaka.chain import Chain
    from sifaka.models.base import create_model

    # Create retrievers
    mock_retriever = MockRetriever()
    memory_retriever = InMemoryRetriever()
    memory_retriever.add_document("doc1", "This is about AI.")

    # Redis retriever as cache wrapper
    redis_retriever = RedisRetriever(base_retriever=memory_retriever)

    # Use in chain
    model = create_model("mock:default")
    chain = Chain(model=model, prompt="Tell me about AI", retriever=redis_retriever)
    ```
"""

# Import base retrievers
from sifaka.retrievers.base import MockRetriever, InMemoryRetriever

# Import Redis retriever with error handling
__all__ = ["MockRetriever", "InMemoryRetriever"]

try:
    from sifaka.retrievers.redis import RedisRetriever, create_redis_retriever

    __all__.extend(["RedisRetriever", "create_redis_retriever"])
except ImportError:
    # Redis not available
    pass

# Import Vector Database retrievers with error handling
try:
    from sifaka.retrievers.vector_db_base import (
        create_vector_db_retriever,
        create_milvus_retriever,
        create_pinecone_retriever,
        create_weaviate_retriever,
    )
    from sifaka.retrievers.milvus import MilvusRetriever

    __all__.extend(
        [
            "MilvusRetriever",
            "create_vector_db_retriever",
            "create_milvus_retriever",
            "create_pinecone_retriever",
            "create_weaviate_retriever",
        ]
    )
except ImportError:
    # Vector DB dependencies not available
    pass

# Import specialized retrievers if they exist
try:
    from sifaka.retrievers.specialized import *

    # Note: specialized module may not exist yet
except ImportError:
    pass
