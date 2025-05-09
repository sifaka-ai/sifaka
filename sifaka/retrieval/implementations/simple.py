"""
Simple retriever implementation for Sifaka.

This module provides a basic retriever implementation that works with
in-memory document collections. It's primarily intended for testing and
demonstration purposes.

## Component Lifecycle

1. **Initialization**
   - Configure document collection
   - Set up similarity function
   - Initialize resources

2. **Operation**
   - Process queries
   - Find relevant documents
   - Format results

3. **Cleanup**
   - Release resources
   - Clean up temporary data

## Error Handling

1. **Query Processing Errors**
   - Empty queries
   - Invalid query format
   - Malformed queries

2. **Retrieval Errors**
   - Empty document collection
   - No matching documents
   - Similarity calculation errors

## Examples

```python
from sifaka.retrieval.implementations import SimpleRetriever

# Create a simple retriever with a document collection
documents = {
    "quantum computing": "Quantum computing uses quantum bits or qubits...",
    "machine learning": "Machine learning is a subset of AI that enables systems to learn..."
}
retriever = SimpleRetriever(documents=documents)

# Retrieve information based on a query
result = retriever.retrieve("How does quantum computing work?")
print(result.get_formatted_results())
```
"""

import time
from typing import Any, Dict, List, Optional, Set, Union

from ..core import RetrieverCore
from ..config import RetrieverConfig
from ..result import StringRetrievalResult
from ..strategies.ranking import SimpleRankingStrategy


class SimpleRetriever(RetrieverCore):
    """
    A simple retriever implementation for in-memory document collections.

    This retriever works with a dictionary of documents and uses simple
    keyword matching to find relevant documents for a query.

    ## Lifecycle Management

    1. **Initialization**
       - Configure document collection
       - Set up similarity function
       - Initialize resources

    2. **Operation**
       - Process queries
       - Find relevant documents
       - Format results

    3. **Cleanup**
       - Release resources
       - Clean up temporary data

    ## Error Handling

    1. **Query Processing Errors**
       - Empty queries
       - Invalid query format
       - Malformed queries

    2. **Retrieval Errors**
       - Empty document collection
       - No matching documents
       - Similarity calculation errors
    """

    def __init__(
        self,
        documents: Optional[Dict[str, str]] = None,
        corpus: Optional[str] = None,
        config: Optional[RetrieverConfig] = None,
    ):
        """
        Initialize the simple retriever.

        Args:
            documents: Dictionary mapping document keys to content
            corpus: Path to a text file containing documents (one per line)
            config: The retriever configuration

        Raises:
            ValueError: If both documents and corpus are None
            FileNotFoundError: If corpus file doesn't exist
        """
        super().__init__(config)
        self._name = "SimpleRetriever"
        self._description = "Simple retriever for in-memory document collections"
        self.documents = {}
        self.ranking_strategy = SimpleRankingStrategy(self.config.ranking)

        if documents is not None:
            self.documents = documents
        elif corpus is not None:
            try:
                with open(corpus, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                    for i, line in enumerate(lines):
                        self.documents[f"doc_{i}"] = line.strip()
            except FileNotFoundError:
                raise FileNotFoundError(f"Corpus file not found: {corpus}")
        else:
            # Initialize with empty dict, but warn
            import warnings
            warnings.warn("Initializing SimpleRetriever with empty document collection")

    def retrieve(self, query: str, **kwargs: Any) -> StringRetrievalResult:
        """
        Retrieve information based on a query.

        This method finds the most relevant documents for the query
        using simple keyword matching and returns them as a formatted string.

        Args:
            query: The query to retrieve information for
            **kwargs: Additional retrieval parameters

        Returns:
            A StringRetrievalResult object

        Raises:
            ValueError: If query is empty
            RuntimeError: If retrieval fails
        """
        start_time = time.time()

        if not query or not isinstance(query, str):
            raise ValueError("Query must be a non-empty string")

        if not self.documents:
            return self.create_result(
                query=query,
                processed_query=query,
                documents=[],
                execution_time_ms=0,
            )

        # Process the query
        processed_query = self.process_query(query)

        # Convert documents to the format expected by the ranking strategy
        doc_list = [
            {
                "content": content,
                "metadata": {"document_id": doc_id},
            }
            for doc_id, content in self.documents.items()
        ]

        # Rank the documents
        ranked_docs = self.ranking_strategy.rank(processed_query, doc_list)

        end_time = time.time()
        execution_time_ms = (end_time - start_time) * 1000

        return self.create_result(
            query=query,
            processed_query=processed_query,
            documents=ranked_docs,
            execution_time_ms=execution_time_ms,
        )
