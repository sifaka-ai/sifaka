"""Base retriever implementations for Sifaka.

This module provides base retriever implementations that can be used to retrieve
relevant documents for a query. Retrievers are used to provide context for models
and critics, enabling them to generate more informed and accurate responses.

Retrievers can be used in two main ways in the Sifaka framework:
1. Pre-generation retrieval: Retrieve documents before generating text to provide context
2. Post-generation retrieval: Retrieve documents after generating text to validate or improve it
"""

from typing import Any, Dict, List, Optional

from sifaka.core.interfaces import Retriever
from sifaka.core.thought import Document, Thought
from sifaka.utils.error_handling import RetrieverError, error_context
from sifaka.utils.logging import get_logger

# Configure logger
logger = get_logger(__name__)


class MockRetriever:
    """Mock retriever for testing.

    This retriever returns predefined documents for any query, making it useful
    for testing and development.

    Attributes:
        documents: List of documents to return for any query.
        max_results: Maximum number of documents to return.
    """

    def __init__(
        self,
        documents: Optional[List[str]] = None,
        max_results: int = 3,
    ):
        """Initialize the retriever.

        Args:
            documents: List of documents to return for any query.
            max_results: Maximum number of documents to return.
        """
        self.documents = documents or [
            "This is a mock document about artificial intelligence.",
            "This is a mock document about machine learning.",
            "This is a mock document about natural language processing.",
            "This is a mock document about deep learning.",
            "This is a mock document about neural networks.",
        ]
        self.max_results = max_results

    def retrieve(self, query: str) -> List[str]:
        """Retrieve relevant documents for a query.

        Args:
            query: The query to retrieve documents for.

        Returns:
            A list of relevant document texts.
        """
        with error_context(
            component="Retriever",
            operation="retrieval",
            error_class=RetrieverError,
            message_prefix="Failed to retrieve documents",
        ):
            logger.debug(f"Retrieving documents for query: {query[:50]}...")
            
            # In a real implementation, this would search for relevant documents
            # For the mock, we just return the predefined documents
            results = self.documents[:self.max_results]
            
            logger.debug(f"Retrieved {len(results)} documents")
            return results

    def retrieve_for_thought(self, thought: Thought, is_pre_generation: bool = True) -> Thought:
        """Retrieve documents for a thought.

        Args:
            thought: The thought to retrieve documents for.
            is_pre_generation: Whether this is pre-generation or post-generation retrieval.

        Returns:
            The thought with retrieved documents added.
        """
        with error_context(
            component="Retriever",
            operation="retrieval for thought",
            error_class=RetrieverError,
            message_prefix="Failed to retrieve documents for thought",
        ):
            # Determine the query based on whether this is pre or post-generation
            if is_pre_generation:
                query = thought.prompt
            else:
                # For post-generation, use both the prompt and the generated text
                query = f"{thought.prompt}\n\n{thought.text}"
            
            # Retrieve documents
            document_texts = self.retrieve(query)
            
            # Convert to Document objects
            documents = [
                Document(
                    text=text,
                    metadata={"source": "mock", "query": query},
                    score=1.0 - (i * 0.1),  # Mock scores
                )
                for i, text in enumerate(document_texts)
            ]
            
            # Add documents to the thought
            if is_pre_generation:
                return thought.add_pre_generation_context(documents)
            else:
                return thought.add_post_generation_context(documents)


class InMemoryRetriever:
    """In-memory retriever for simple document collections.

    This retriever stores documents in memory and performs simple keyword matching
    to retrieve relevant documents for a query.

    Attributes:
        documents: Dictionary mapping document IDs to document texts.
        metadata: Dictionary mapping document IDs to metadata.
        max_results: Maximum number of documents to return.
    """

    def __init__(
        self,
        documents: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, Dict[str, Any]]] = None,
        max_results: int = 3,
    ):
        """Initialize the retriever.

        Args:
            documents: Dictionary mapping document IDs to document texts.
            metadata: Dictionary mapping document IDs to metadata.
            max_results: Maximum number of documents to return.
        """
        self.documents = documents or {}
        self.metadata = metadata or {}
        self.max_results = max_results

    def add_document(
        self,
        doc_id: str,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add a document to the retriever.

        Args:
            doc_id: The document ID.
            text: The document text.
            metadata: Optional metadata for the document.
        """
        self.documents[doc_id] = text
        if metadata:
            self.metadata[doc_id] = metadata

    def retrieve(self, query: str) -> List[str]:
        """Retrieve relevant documents for a query.

        Args:
            query: The query to retrieve documents for.

        Returns:
            A list of relevant document texts.
        """
        with error_context(
            component="Retriever",
            operation="retrieval",
            error_class=RetrieverError,
            message_prefix="Failed to retrieve documents",
        ):
            logger.debug(f"Retrieving documents for query: {query[:50]}...")
            
            # Simple keyword matching
            query_terms = set(query.lower().split())
            results = []
            
            for doc_id, text in self.documents.items():
                # Count how many query terms appear in the document
                doc_terms = set(text.lower().split())
                matches = len(query_terms.intersection(doc_terms))
                
                if matches > 0:
                    results.append((doc_id, text, matches))
            
            # Sort by number of matches (descending)
            results.sort(key=lambda x: x[2], reverse=True)
            
            # Return the top results
            top_results = [text for _, text, _ in results[:self.max_results]]
            
            logger.debug(f"Retrieved {len(top_results)} documents")
            return top_results

    def retrieve_for_thought(self, thought: Thought, is_pre_generation: bool = True) -> Thought:
        """Retrieve documents for a thought.

        Args:
            thought: The thought to retrieve documents for.
            is_pre_generation: Whether this is pre-generation or post-generation retrieval.

        Returns:
            The thought with retrieved documents added.
        """
        with error_context(
            component="Retriever",
            operation="retrieval for thought",
            error_class=RetrieverError,
            message_prefix="Failed to retrieve documents for thought",
        ):
            # Determine the query based on whether this is pre or post-generation
            if is_pre_generation:
                query = thought.prompt
            else:
                # For post-generation, use both the prompt and the generated text
                query = f"{thought.prompt}\n\n{thought.text}"
            
            # Retrieve documents
            document_texts = self.retrieve(query)
            
            # Convert to Document objects
            documents = [
                Document(
                    text=text,
                    metadata={"source": "in-memory", "query": query},
                    score=1.0 - (i * 0.1),  # Simple scoring
                )
                for i, text in enumerate(document_texts)
            ]
            
            # Add documents to the thought
            if is_pre_generation:
                return thought.add_pre_generation_context(documents)
            else:
                return thought.add_post_generation_context(documents)
