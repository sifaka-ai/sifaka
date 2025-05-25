"""Simple retriever implementations for Sifaka.

This module provides basic retriever implementations that don't require external
dependencies. These are primarily used for testing and as base components for
the unified storage system.

Available retrievers:
- InMemoryRetriever: Simple in-memory document storage and retrieval
- MockRetriever: Mock retriever for testing
- Retriever: Protocol defining the retriever interface

For advanced storage capabilities including caching and vector search,
use the unified storage system in `sifaka.storage`.

Example:
    ```python
    from sifaka.retrievers import InMemoryRetriever

    # Create a simple in-memory retriever
    retriever = InMemoryRetriever()

    # Add documents
    retriever.add_document("doc1", "This is about AI")
    retriever.add_document("doc2", "This is about ML")

    # Retrieve relevant documents
    results = retriever.retrieve("artificial intelligence")
    ```

For caching, vector search, and persistence, use the unified storage system:
    ```python
    from sifaka.storage import SifakaStorage
    from sifaka.retrievers import InMemoryRetriever

    # Create storage manager
    storage = SifakaStorage(redis_config=..., milvus_config=...)

    # Wrap retriever with caching
    base_retriever = InMemoryRetriever()
    cached_retriever = storage.get_retriever_cache(base_retriever)
    ```
"""

from sifaka.retrievers.simple import (
    InMemoryRetriever,
    MockRetriever,
    Retriever,
)

__all__ = [
    "Retriever",
    "InMemoryRetriever",
    "MockRetriever",
]
