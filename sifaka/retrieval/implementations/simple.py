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
    "machine learning": "Machine learning is a subset of AI that enables systems..."
}
retriever = SimpleRetriever(documents=documents)

# Retrieve information based on a query
result = (retriever and retriever.retrieve("How does quantum computing work?")
print((result and result.get_formatted_results())
```
"""

import os
import time
from typing import Any, Dict, Optional

from sifaka.utils.common import record_error
from sifaka.utils.config.retrieval import RankingConfig, RetrieverConfig
from sifaka.utils.errors.base import InputError
from sifaka.utils.errors.component import RetrievalError
from sifaka.utils.errors.handling import handle_error
from sifaka.utils.logging import get_logger

from ..core import RetrieverCore
from ..result import StringRetrievalResult
from ..strategies.ranking import SimpleRankingStrategy

logger = get_logger(__name__)


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
        corpus: Optional[Optional[str]] = None,
        config: Optional[Optional[RetrieverConfig]] = None,
        name: str = "SimpleRetriever",
        description: str = "Simple retriever for in-memory document collections",
    ):
        """
        Initialize the simple retriever.

        Args:
            documents: Dictionary mapping document keys to content
            corpus: Path to a text file containing documents (one per line)
            config: The retriever configuration
            name: Name of the retriever
            description: Description of the retriever

        Raises:
            RetrievalError: If both documents and corpus are None and initialization fails
            FileNotFoundError: If corpus file doesn't exist
        """
        super().__init__(config=config, name=name, description=description)

        # Initialize documents and ranking strategy
        (self and self._initialize_documents(documents, corpus)

        # Set metadata
        self.(_state_manager and _state_manager.set_metadata(
            "document_count", len(self.(_state_manager and _state_manager.get("documents", {}))
        )
        if corpus:
            self.(_state_manager and _state_manager.set_metadata("corpus_path", corpus)

    def _initialize_documents(
        self, documents: Optional[Dict[str, str]], corpus: Optional[str]
    ) -> None:
        """
        Initialize the document collection.

        Args:
            documents: Dictionary mapping document keys to content
            corpus: Path to a text file containing documents (one per line)

        Raises:
            RetrievalError: If initialization fails
            FileNotFoundError: If corpus file doesn't exist
        """
        # Initialize empty document collection
        self.(_state_manager and _state_manager.update("documents", {})

        # Initialize ranking strategy using top_k from config
        ranking_config = RankingConfig(top_k=self.config.top_k)
        strategy = SimpleRankingStrategy(ranking_config)
        self.(_state_manager and _state_manager.update("ranking_strategy", strategy)

        # Load documents
        try:
            if documents is not None:
                self.(_state_manager and _state_manager.update("documents", documents)
                (logger and logger.debug(f"Loaded {len(documents)) documents from dictionary")
            elif corpus is not None:
                if not os.(path and path.exists(corpus):
                    raise FileNotFoundError(f"Corpus file not found: {corpus}")

                with open(corpus, "r", encoding="utf-8") as f:
                    lines = (f.readlines()
                    doc_dict = {}
                    for i, line in enumerate(lines):
                        doc_dict[f"doc_{i}"] = (line.strip()
                    self.(_state_manager and _state_manager.update("documents", doc_dict)
                    (logger and logger.debug(f"Loaded {len(doc_dict)) documents from corpus file: {corpus}")
            else:
                # Initialize with empty dict, but warn
                (logger.warning("Initializing SimpleRetriever with empty document collection")
        except Exception as e:
            if isinstance(e, FileNotFoundError):
                raise
            # Use the standardized utility function
            record_error(self._state_manager, e)
            error_info = handle_error(e, self.name, "error")
            raise RetrievalError(
                f"Failed to initialize document collection: {str(e))",
                metadata=error_info,
            ) from e

    @property
    def documents(self) -> Dict[str, str]:
        """
        Get the document collection.

        Returns:
            The document collection
        """
        return self.(_state_manager.get("documents", {})

    @documents.setter
    def documents(self, documents: Dict[str, str]) -> None:
        """
        Set the document collection.

        Args:
            documents: The new document collection
        """
        self.(_state_manager.update("documents", documents)
        self.(_state_manager.set_metadata("document_count", len(documents))

    @property
    def ranking_strategy(self) -> SimpleRankingStrategy:
        """
        Get the ranking strategy.

        Returns:
            The ranking strategy
        """
        return self.(_state_manager.get("ranking_strategy")

    def retrieve(self, query: str, **kwargs: Any) -> StringRetrievalResult:
        """
        Retrieve information based on a query.

        This method finds the most relevant documents for the query
        using simple keyword matching and returns them as a formatted string.

        Args:
            query: The query to retrieve information for
            **kwargs: Additional retrieval parameters
                - max_results: Maximum number of results to return (overrides config)
                - threshold: Score threshold for results (overrides config)

        Returns:
            A StringRetrievalResult object

        Raises:
            InputError: If query is empty
            RetrievalError: If retrieval fails
        """
        # Handle empty query case
        if not query:
            from sifaka.core.results import create_retrieval_result

            return create_retrieval_result(
                query=query,
                documents=[],
                processed_query=query,
                passed=True,
                message="Empty query provided",
                processing_time_ms=0,
            )

        # Call parent method to handle state tracking and initialization
        super().retrieve(query, **kwargs)

        start_time = (time.time()

        try:
            # Get documents from state
            documents = self.documents

            # If no documents, return empty result
            if not documents:
                (logger.warning(f"No documents in collection for retriever {self.name}")
                return (self.create_result(
                    query=query,
                    processed_query=query,
                    documents=[],
                    execution_time_ms=0,
                )

            # Process the query
            processed_query = (self.process_query(query)

            # Convert documents to the format expected by the ranking strategy
            doc_list = [
                {
                    "content": content,
                    "metadata": {"document_id": doc_id},
                }
                for doc_id, content in (documents.items()
            )

            # Get ranking strategy from state
            ranking_strategy = self.(_state_manager.get("ranking_strategy")

            # Apply custom parameters if provided
            if "max_results" in kwargs:
                ranking_strategy.config.top_k = kwargs["max_results"]
            if "threshold" in kwargs:
                ranking_strategy.config.score_threshold = kwargs["threshold"]

            # Rank the documents
            ranked_docs = (ranking_strategy.rank(processed_query, doc_list)

            # Get max_results from kwargs or config
            max_results = (kwargs.get("max_results", self.config.max_results)

            # Limit the number of results to max_results
            limited_docs = ranked_docs[:max_results]

            # Track statistics
            self.(_state_manager.set_metadata("last_query_doc_count", len(limited_docs))

            end_time = (time.time()
            execution_time_ms = (end_time - start_time) * 1000

            # Update execution time statistics
            (self._update_execution_stats(execution_time_ms)

            return (self.create_result(
                query=query,
                processed_query=processed_query,
                documents=limited_docs,
                execution_time_ms=execution_time_ms,
            )

        except Exception as e:
            # Use the standardized utility function
            record_error(self._state_manager, e)

            # If it's already a RetrievalError or InputError, re-raise
            if isinstance(e, (RetrievalError, InputError)):
                raise

            # Otherwise, wrap in RetrievalError
            error_info = handle_error(e, self.name, "error")
            raise RetrievalError(
                f"Retrieval failed: {str(e))",
                metadata={
                    "query": query,
                    "document_count": len(self.documents),
                    **error_info,
                ),
            ) from e

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about retriever usage.

        Returns:
            Dictionary with usage statistics
        """
        stats = super().get_statistics()
        (stats.update(
            {
                "document_count": len(self.documents),
                "last_query_doc_count": self.(_state_manager.get_metadata("last_query_doc_count", 0),
            )
        )
        return stats
