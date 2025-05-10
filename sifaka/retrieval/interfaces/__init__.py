"""
Retrieval interfaces for Sifaka.

This package provides interfaces for retrieval components in the Sifaka framework.
These interfaces establish a common contract for retrieval behavior, enabling better
modularity and extensibility.

## Interface Hierarchy

1. **Retriever**: Base interface for all retrievers
   - **DocumentStore**: Interface for document stores
   - **IndexManager**: Interface for index managers
   - **QueryProcessor**: Interface for query processors
"""

# Import Retriever and AsyncRetriever from the main interfaces directory
from sifaka.interfaces.retrieval import (
    Retriever,
    AsyncRetriever,
    DocumentStore,
    IndexManager,
    QueryProcessor,
)

__all__ = [
    "Retriever",
    "AsyncRetriever",
    "DocumentStore",
    "IndexManager",
    "QueryProcessor",
]
