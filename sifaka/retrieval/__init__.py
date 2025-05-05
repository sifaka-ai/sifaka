"""
Retrieval module for Sifaka.

This module provides retrieval capabilities for retrieving relevant information
from various sources based on queries. It includes a base Retriever interface
and implementations for different retrieval strategies.

## Component Overview

1. **Base Interface**
   - `Retriever`: Abstract base class for all retrievers

2. **Implementations**
   - `SimpleRetriever`: A basic retriever for in-memory document collections

## Usage Examples

```python
from sifaka.retrieval import SimpleRetriever

# Create a simple retriever with a document collection
documents = {
    "quantum computing": "Quantum computing uses quantum bits or qubits...",
    "machine learning": "Machine learning is a subset of AI that enables systems to learn..."
}
retriever = SimpleRetriever(documents=documents)

# Retrieve information based on a query
result = retriever.retrieve("How does quantum computing work?")
print(result)
```
"""

from .base import Retriever
from .simple import SimpleRetriever

__all__ = [
    "Retriever",
    "SimpleRetriever",
]
