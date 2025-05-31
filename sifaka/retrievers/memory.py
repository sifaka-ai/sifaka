"""In-memory retriever implementation for Sifaka.

This module provides a simple in-memory retriever that stores documents in memory
and performs keyword-based retrieval. It's designed to be lightweight and suitable
for testing, development, and small-scale applications.

Example:
    ```python
    from sifaka.retrievers.memory import InMemoryRetriever

    # Create retriever
    retriever = InMemoryRetriever()

    # Add documents
    retriever.add_document("ai_doc", "Artificial intelligence is transforming technology")
    retriever.add_document("ml_doc", "Machine learning is a subset of AI")

    # Retrieve relevant documents
    results = retriever.retrieve("artificial intelligence", limit=5)
    print(results)  # ['Artificial intelligence is transforming technology', ...]
    ```
"""

from typing import Dict, List

from sifaka.utils.logging import get_logger

logger = get_logger(__name__)


class InMemoryRetriever:
    """Simple in-memory retriever with keyword-based search.

    This retriever stores documents in memory and performs simple keyword matching
    to find relevant documents. It's designed to be lightweight and easy to use.

    Features:
    - Simple keyword-based matching
    - Relevance scoring using Jaccard similarity
    - Configurable result limits
    - No external dependencies
    """

    def __init__(self, max_results: int = 10):
        """Initialize the in-memory retriever.

        Args:
            max_results: Default maximum number of results to return.
        """
        self.documents: Dict[str, str] = {}
        self.max_results = max_results

        logger.debug(f"Initialized InMemoryRetriever with max_results={max_results}")

    def add_document(self, doc_id: str, text: str) -> None:
        """Add a document to the retriever.

        Args:
            doc_id: Unique identifier for the document.
            text: The document text content.
        """
        self.documents[doc_id] = text
        logger.debug(f"Added document {doc_id} ({len(text)} characters)")

    def retrieve(self, query: str, limit: int = None) -> List[str]:
        """Retrieve relevant documents for a query.

        Args:
            query: The query to retrieve documents for.
            limit: Maximum number of documents to return. Uses default if None.

        Returns:
            A list of relevant document texts, ordered by relevance score.
        """
        if limit is None:
            limit = self.max_results

        logger.debug(f"Retrieving documents for query: '{query[:50]}...' (limit={limit})")

        if not query.strip():
            logger.debug("Empty query, returning empty results")
            return []

        if not self.documents:
            logger.debug("No documents available, returning empty results")
            return []

        # Simple keyword matching with relevance scoring
        query_terms = set(query.lower().split())
        results = []

        for doc_id, text in self.documents.items():
            score = self._calculate_relevance_score(query_terms, text)
            if score > 0:
                results.append((doc_id, text, score))

        # Sort by relevance score (descending)
        results.sort(key=lambda x: x[2], reverse=True)

        # Return the top results (text only)
        top_results = [text for _, text, _ in results[:limit]]

        logger.debug(f"Retrieved {len(top_results)} documents")
        return top_results

    def retrieve_with_metadata(self, query: str, limit: int = None) -> List[dict]:
        """Retrieve relevant documents with metadata and scores.

        Args:
            query: The query to retrieve documents for.
            limit: Maximum number of documents to return. Uses default if None.

        Returns:
            A list of dictionaries with 'text', 'metadata', and 'score' fields.
        """
        if limit is None:
            limit = self.max_results

        logger.debug(
            f"Retrieving documents with metadata for query: '{query[:50]}...' (limit={limit})"
        )

        if not query.strip():
            logger.debug("Empty query, returning empty results")
            return []

        if not self.documents:
            logger.debug("No documents available, returning empty results")
            return []

        # Simple keyword matching with relevance scoring
        query_terms = set(query.lower().split())
        results = []

        for doc_id, text in self.documents.items():
            score = self._calculate_relevance_score(query_terms, text)
            if score > 0:
                results.append((doc_id, text, score))

        # Sort by relevance score (descending)
        results.sort(key=lambda x: x[2], reverse=True)

        # Return the top results with metadata
        top_results = []
        for doc_id, text, score in results[:limit]:
            top_results.append(
                {
                    "text": text,
                    "metadata": {"doc_id": doc_id, "retriever": "InMemoryRetriever"},
                    "score": score,
                }
            )

        logger.debug(f"Retrieved {len(top_results)} documents with metadata")
        return top_results

    def _calculate_relevance_score(self, query_terms: set, text: str) -> float:
        """Calculate relevance score using Jaccard similarity.

        Args:
            query_terms: Set of query terms (lowercase).
            text: Document text to score.

        Returns:
            Relevance score between 0.0 and 1.0.
        """
        if not query_terms:
            return 0.0

        doc_terms = set(text.lower().split())

        # Jaccard similarity: intersection / union
        intersection = len(query_terms.intersection(doc_terms))
        union = len(query_terms.union(doc_terms))

        if union == 0:
            return 0.0

        return intersection / union

    def clear(self) -> None:
        """Remove all documents from the retriever."""
        count = len(self.documents)
        self.documents.clear()
        logger.debug(f"Cleared {count} documents from retriever")

    def get_document_count(self) -> int:
        """Get the number of documents in the retriever.

        Returns:
            Number of documents currently stored.
        """
        return len(self.documents)

    def get_document(self, doc_id: str) -> str:
        """Get a specific document by ID.

        Args:
            doc_id: The document ID to retrieve.

        Returns:
            The document text.

        Raises:
            KeyError: If the document ID is not found.
        """
        return self.documents[doc_id]

    def has_document(self, doc_id: str) -> bool:
        """Check if a document exists in the retriever.

        Args:
            doc_id: The document ID to check.

        Returns:
            True if the document exists, False otherwise.
        """
        return doc_id in self.documents

    def remove_document(self, doc_id: str) -> bool:
        """Remove a document from the retriever.

        Args:
            doc_id: The document ID to remove.

        Returns:
            True if the document was removed, False if it didn't exist.
        """
        if doc_id in self.documents:
            del self.documents[doc_id]
            logger.debug(f"Removed document {doc_id}")
            return True
        else:
            logger.debug(f"Document {doc_id} not found for removal")
            return False
