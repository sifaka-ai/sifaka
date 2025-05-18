"""
Simple retriever for Sifaka.

This module provides a simple retriever that returns documents based on keyword matching.
"""

import logging
from typing import List

from sifaka.retrievers.base import Retriever

# Configure logger
logger = logging.getLogger(__name__)


class SimpleRetriever(Retriever):
    """Simple retriever that returns documents based on keyword matching.

    This retriever is for demonstration purposes and simply returns documents
    that contain any of the words in the query.

    Attributes:
        documents: A list of documents to search through.
    """

    def __init__(self, documents: List[str]):
        """Initialize the simple retriever.

        Args:
            documents: A list of documents to search through.
        """
        self.documents = documents
        logger.debug(f"Initialized SimpleRetriever with {len(documents)} documents")

    def retrieve(self, query: str) -> List[str]:
        """Retrieve relevant documents for a query.

        This simple implementation just checks if any word in the query
        appears in the document.

        Args:
            query: The query to retrieve documents for.

        Returns:
            A list of relevant document texts.
        """
        if not query:
            logger.warning("Empty query provided to SimpleRetriever")
            return []

        # Normalize query
        query = query.lower()
        query_words = set(query.split())

        # Filter out stop words and very short words
        query_words = {word for word in query_words if len(word) > 2}

        # Find matching documents
        results = []
        for doc in self.documents:
            doc_lower = doc.lower()
            if any(word in doc_lower for word in query_words):
                results.append(doc)

        logger.debug(f"SimpleRetriever found {len(results)} matching documents for query: {query}")
        return results
