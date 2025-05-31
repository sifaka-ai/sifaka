"""Retriever protocol for Sifaka.

This module defines the simple Retriever protocol that all retriever implementations
must follow. The protocol is designed to be minimal and easy to implement.

The protocol supports simple text-based retrieval for use with PydanticAI tools
and Sifaka critics that need contextual information.

Example:
    ```python
    from sifaka.retrievers.protocol import Retriever
    from sifaka.retrievers.memory import InMemoryRetriever

    # Create a retriever
    retriever: Retriever = InMemoryRetriever()

    # Document management
    retriever.add_document("doc1", "This is about AI")
    retriever.add_document("doc2", "This is about ML")

    # Check document existence
    if retriever.has_document("doc1"):
        text = retriever.get_document("doc1")
        print(f"Document count: {retriever.get_document_count()}")

    # Retrieve relevant documents
    results = retriever.retrieve("artificial intelligence", limit=5)

    # Clean up
    retriever.remove_document("doc1")
    retriever.clear()  # Remove all documents
    ```
"""

from typing import List, Protocol


class Retriever(Protocol):
    """Protocol for retriever implementations.

    This protocol defines the essential interface that all retrievers must implement
    to be compatible with Sifaka chains and PydanticAI tools.

    The protocol includes core document management operations that are fundamental
    for any retriever implementation. It's designed to be:
    - Simple to implement
    - Easy to test
    - Compatible with PydanticAI tools
    - Suitable for both sync and async usage
    - Complete for basic document management
    """

    def retrieve(self, query: str, limit: int = 10) -> List[str]:
        """Retrieve relevant documents for a query.

        Args:
            query: The query to retrieve documents for.
            limit: Maximum number of documents to return.

        Returns:
            A list of relevant document texts, ordered by relevance.
        """
        ...

    def add_document(self, doc_id: str, text: str) -> None:
        """Add a document to the retriever.

        Args:
            doc_id: Unique identifier for the document.
            text: The document text content.
        """
        ...

    def has_document(self, doc_id: str) -> bool:
        """Check if a document exists in the retriever.

        Args:
            doc_id: The document ID to check.

        Returns:
            True if the document exists, False otherwise.
        """
        ...

    def get_document(self, doc_id: str) -> str:
        """Get a specific document by ID.

        Args:
            doc_id: The document ID to retrieve.

        Returns:
            The document text.

        Raises:
            KeyError: If the document ID is not found.
        """
        ...

    def remove_document(self, doc_id: str) -> bool:
        """Remove a document from the retriever.

        Args:
            doc_id: The document ID to remove.

        Returns:
            True if the document was removed, False if it didn't exist.
        """
        ...

    def get_document_count(self) -> int:
        """Get the number of documents in the retriever.

        Returns:
            Number of documents currently stored.
        """
        ...

    def clear(self) -> None:
        """Remove all documents from the retriever."""
        ...
