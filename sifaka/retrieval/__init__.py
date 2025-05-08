"""
Retrieval module for Sifaka.

This module provides retrieval capabilities for retrieving relevant information
from various sources based on queries. It includes a Retriever class that follows
the composition over inheritance pattern and implementations for different retrieval strategies.

## Component Overview

1. **Base Components**
   - `Retriever`: Main class that delegates to implementations
   - `RetrieverConfig`: Configuration for retrievers
   - `RetrieverImplementation`: Protocol for retriever implementations

2. **Factory Functions**
   - `create_simple_retriever`: Creates a simple retriever for in-memory document collections

## Usage Examples

```python
from sifaka.retrieval import create_simple_retriever

# Create a simple retriever with a document collection
documents = {
    "quantum computing": "Quantum computing uses quantum bits or qubits...",
    "machine learning": "Machine learning is a subset of AI that enables systems to learn..."
}
retriever = create_simple_retriever(documents=documents)

# Retrieve information based on a query
result = retriever.retrieve("How does quantum computing work?")
print(result)
```
"""

from .base import Retriever, RetrieverConfig, RetrieverImplementation
from .simple import create_simple_retriever

__all__ = [
    "Retriever",
    "RetrieverConfig",
    "RetrieverImplementation",
    "create_simple_retriever",
]
