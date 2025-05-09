"""
Result models for retrieval components.

This module provides result models for retrieval components in the Sifaka framework.
These models define the structure of results returned by retrievers and related components.
"""

from typing import Any, Dict, Generic, List, Optional, TypeVar, Union
from pydantic import BaseModel, Field

T = TypeVar("T")


class DocumentMetadata(BaseModel):
    """
    Metadata for a retrieved document.

    This class defines the metadata for a retrieved document,
    including its ID, source, and other attributes.
    """

    document_id: str = Field(
        description="The ID of the document",
    )
    source: Optional[str] = Field(
        default=None,
        description="The source of the document (e.g., file path, URL)",
    )
    created_at: Optional[str] = Field(
        default=None,
        description="The creation timestamp of the document",
    )
    updated_at: Optional[str] = Field(
        default=None,
        description="The last update timestamp of the document",
    )
    additional_metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata for the document",
    )


class RetrievedDocument(BaseModel, Generic[T]):
    """
    A retrieved document with content and metadata.

    This class defines a retrieved document, including its content,
    metadata, and relevance score.
    """

    content: T = Field(
        description="The content of the document",
    )
    metadata: DocumentMetadata = Field(
        default_factory=DocumentMetadata,
        description="Metadata for the document",
    )
    score: Optional[float] = Field(
        default=None,
        description="The relevance score of the document",
    )


class RetrievalResult(BaseModel, Generic[T]):
    """
    Result of a retrieval operation.

    This class defines the result of a retrieval operation,
    including the retrieved documents and query information.
    """

    documents: List[RetrievedDocument[T]] = Field(
        default_factory=list,
        description="The retrieved documents",
    )
    query: str = Field(
        description="The query used for retrieval",
    )
    processed_query: Optional[str] = Field(
        default=None,
        description="The processed query (after preprocessing)",
    )
    total_results: int = Field(
        default=0,
        description="The total number of results found",
    )
    execution_time_ms: Optional[float] = Field(
        default=None,
        description="The execution time in milliseconds",
    )
    additional_info: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional information about the retrieval operation",
    )

    @property
    def top_document(self) -> Optional[RetrievedDocument[T]]:
        """
        Get the top document from the results.

        Returns:
            The top document, or None if no documents were retrieved
        """
        if not self.documents:
            return None
        return self.documents[0]

    @property
    def top_content(self) -> Optional[T]:
        """
        Get the content of the top document.

        Returns:
            The content of the top document, or None if no documents were retrieved
        """
        top_doc = self.top_document
        if top_doc is None:
            return None
        return top_doc.content

    def get_contents(self) -> List[T]:
        """
        Get the contents of all retrieved documents.

        Returns:
            A list of document contents
        """
        return [doc.content for doc in self.documents]

    def get_formatted_results(self, include_scores: bool = True) -> str:
        """
        Get a formatted string representation of the results.

        Args:
            include_scores: Whether to include relevance scores in the output

        Returns:
            A formatted string representation of the results
        """
        if not self.documents:
            return "No results found for query: " + self.query

        result = f"Results for query: {self.query}\n\n"
        for i, doc in enumerate(self.documents, 1):
            result += f"Document {i}"
            if include_scores and doc.score is not None:
                result += f" (Score: {doc.score:.4f})"
            result += f":\n{doc.content}\n\n"

        return result.strip()


class StringRetrievalResult(RetrievalResult[str]):
    """
    Result of a retrieval operation with string content.

    This class specializes RetrievalResult for string content,
    which is the most common case for text retrieval.
    """

    def get_concatenated_content(self, separator: str = "\n\n") -> str:
        """
        Get the concatenated content of all retrieved documents.

        Args:
            separator: The separator to use between document contents

        Returns:
            The concatenated content of all retrieved documents
        """
        return separator.join(self.get_contents())
