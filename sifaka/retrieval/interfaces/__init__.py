"""
Retrieval interfaces for Sifaka.

This module defines the interfaces for retrieval components in the Sifaka framework.
These interfaces establish a common contract for retrieval behavior, enabling better
modularity and extensibility.

## Interface Hierarchy

1. **Retriever**: Base interface for all retrievers
   - **DocumentStore**: Interface for document stores
   - **IndexManager**: Interface for index managers
   - **QueryProcessor**: Interface for query processors

## Usage

These interfaces are defined using Python's Protocol class from typing,
which enables structural subtyping. This means that classes don't need to
explicitly inherit from these interfaces; they just need to implement the
required methods and properties.
"""

from .retriever import (
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
