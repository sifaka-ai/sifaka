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

import re
from typing import Dict, List, Optional, Set, Union

from .base import Retriever


class SimpleRetriever(Retriever):
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
        max_results: int = 3,
    ):
        """
        Initialize the simple retriever.

        Args:
            documents: Dictionary mapping document keys to content
            corpus: Path to a text file containing documents (one per line)
            max_results: Maximum number of results to return

        Raises:
            ValueError: If both documents and corpus are None
            FileNotFoundError: If corpus file doesn't exist
        """
        self.max_results = max_results
        self.documents = {}

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

    def retrieve(self, query: str) -> str:
        """
        Retrieve information based on a query.

        This method finds the most relevant documents for the query
        using simple keyword matching and returns them as a formatted string.

        Args:
            query: The query to retrieve information for

        Returns:
            Retrieved information as a string

        Raises:
            ValueError: If query is empty
            RuntimeError: If retrieval fails
        """
        if not query or not isinstance(query, str):
            raise ValueError("Query must be a non-empty string")

        if not self.documents:
            return "No documents available for retrieval."

        # Extract keywords from query
        keywords = self._extract_keywords(query)
        if not keywords:
            return "Could not extract meaningful keywords from query."

        # Find relevant documents
        relevant_docs = self._find_relevant_documents(keywords)
        if not relevant_docs:
            return "No relevant documents found for the query."

        # Format results
        return self._format_results(relevant_docs)

    def _extract_keywords(self, query: str) -> Set[str]:
        """
        Extract keywords from a query.

        Args:
            query: The query to extract keywords from

        Returns:
            Set of keywords
        """
        # Remove punctuation and convert to lowercase
        query = re.sub(r'[^\w\s]', '', query.lower())
        
        # Split into words
        words = query.split()
        
        # Remove common stop words
        stop_words = {
            "a", "an", "the", "and", "or", "but", "is", "are", "was", "were",
            "be", "been", "being", "in", "on", "at", "to", "for", "with",
            "about", "against", "between", "into", "through", "during",
            "before", "after", "above", "below", "from", "up", "down", "of",
            "off", "over", "under", "again", "further", "then", "once", "here",
            "there", "when", "where", "why", "how", "all", "any", "both",
            "each", "few", "more", "most", "other", "some", "such", "no",
            "nor", "not", "only", "own", "same", "so", "than", "too", "very",
            "s", "t", "can", "will", "just", "don", "should", "now"
        }
        
        return {word for word in words if word not in stop_words}

    def _find_relevant_documents(self, keywords: Set[str]) -> List[str]:
        """
        Find relevant documents based on keywords.

        Args:
            keywords: Set of keywords to match

        Returns:
            List of relevant document contents
        """
        # Calculate relevance scores
        scores = {}
        for doc_id, content in self.documents.items():
            score = 0
            content_lower = content.lower()
            for keyword in keywords:
                if keyword.lower() in content_lower:
                    score += 1
            if score > 0:
                scores[doc_id] = score
        
        # Sort by relevance score
        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        # Get top results
        top_docs = [self.documents[doc_id] for doc_id, _ in sorted_docs[:self.max_results]]
        
        return top_docs

    def _format_results(self, documents: List[str]) -> str:
        """
        Format retrieval results.

        Args:
            documents: List of document contents

        Returns:
            Formatted results as a string
        """
        if not documents:
            return ""
        
        result = "Retrieved information:\n\n"
        for i, doc in enumerate(documents, 1):
            result += f"Document {i}:\n{doc}\n\n"
        
        return result.strip()
