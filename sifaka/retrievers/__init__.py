"""Simplified retriever implementations for Sifaka.

This module provides a clean, minimal retriever interface and implementation
designed for the PydanticAI-first architecture.

Available components:
- Retriever: Protocol defining the simple retriever interface
- InMemoryRetriever: Lightweight in-memory document storage and retrieval

The retriever system has been simplified to focus on core functionality:
- Simple text-based retrieval
- Minimal interface for easy implementation
- Compatible with PydanticAI tools
- No complex thought integration

Example:
    ```python
    from sifaka.retrievers import InMemoryRetriever

    # Create a simple in-memory retriever
    retriever = InMemoryRetriever()

    # Add documents
    retriever.add_document("doc1", "This is about AI")
    retriever.add_document("doc2", "This is about ML")

    # Retrieve relevant documents
    results = retriever.retrieve("artificial intelligence", limit=5)
    ```

For advanced storage capabilities including caching and vector search,
use the storage system in `sifaka.storage`.
"""

from sifaka.retrievers.memory import InMemoryRetriever
from sifaka.retrievers.protocol import Retriever

__all__ = [
    "Retriever",
    "InMemoryRetriever",
]
