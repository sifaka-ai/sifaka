from typing import Any, List

"""
Retrieval module for Sifaka.

This module provides retrieval capabilities for retrieving relevant information
from various sources based on queries. It includes interfaces, implementations,
and utilities for different retrieval strategies.

## Component Overview

1. **Core Components**
   - `RetrieverCore`: Core implementation of retriever functionality
   - `RetrieverConfig`: Configuration for retrievers
   - `RetrievalResult`: Result models for retrieval operations

2. **Interfaces**
   - `RetrieverProtocol`: Interface for retrievers
   - `DocumentStoreProtocol`: Interface for document stores
   - `IndexManagerProtocol`: Interface for index managers
   - `QueryProcessorProtocol`: Interface for query processors

3. **Implementations**
   - `SimpleRetriever`: A basic retriever for in-memory document collections

4. **Managers**
   - `QueryManager`: Manager for query processing

5. **Strategies**
   - `RankingStrategy`: Abstract base class for ranking strategies
   - `SimpleRankingStrategy`: Simple ranking strategy based on keyword matching
   - `ScoreThresholdRankingStrategy`: Ranking strategy with score thresholding

6. **Factory Functions**
   - `create_simple_retriever`: Create a simple retriever
   - `create_threshold_retriever`: Create a retriever with score thresholding

## Usage Examples

```python
from sifaka.retrieval import create_simple_retriever

# Create a simple retriever with a document collection
documents = {
    "quantum computing": "Quantum computing uses quantum bits or qubits...",
    "machine learning": "Machine learning is a subset of AI that enables systems to learn..."
}
retriever = create_simple_retriever(documents=documents, max_results=3)

# Retrieve information based on a query
result = retriever.retrieve("How does quantum computing work?") if retriever else ""
print(result.get_formatted_results() if result else "")
```
"""
from sifaka.interfaces import (
    RetrieverProtocol,
    DocumentStoreProtocol,
    IndexManagerProtocol,
    QueryProcessorProtocol,
)
from .core import RetrieverCore
from sifaka.utils.config.retrieval import RetrieverConfig
from ..core.results import RetrievalResult
from .result import StringRetrievalResult, RetrievedDocument
from .implementations.simple import SimpleRetriever
from .managers.query import QueryManager
from .strategies.ranking import (
    RankingStrategy,
    SimpleRankingStrategy,
    ScoreThresholdRankingStrategy,
)
from .factories import create_simple_retriever, create_threshold_retriever

__all__: List[Any] = [
    "RetrieverProtocol",
    "DocumentStoreProtocol",
    "IndexManagerProtocol",
    "QueryProcessorProtocol",
    "RetrieverCore",
    "RetrieverConfig",
    "RetrievalResult",
    "StringRetrievalResult",
    "RetrievedDocument",
    "SimpleRetriever",
    "QueryManager",
    "RankingStrategy",
    "SimpleRankingStrategy",
    "ScoreThresholdRankingStrategy",
    "create_simple_retriever",
    "create_threshold_retriever",
]
