"""
Result models for retrieval components.

This module provides result models for retrieval components in the Sifaka framework.
These models define the structure of results returned by retrievers and related components.

## Result Models

1. **DocumentMetadata**: Metadata for a retrieved document
2. **RetrievedDocument**: A retrieved document with content and metadata
3. **RetrievalResult**: Result of a retrieval operation
4. **StringRetrievalResult**: Result of a retrieval operation with string content

## Usage Examples

```python
from sifaka.retrieval.result import StringRetrievalResult, RetrievedDocument, DocumentMetadata

# Create document metadata
metadata = DocumentMetadata(
    document_id="doc_1",
    source="example.txt",
    created_at="2023-01-01",
)

# Create a retrieved document
document = RetrievedDocument(
    content="This is the document content.",
    metadata=metadata,
    score=0.95,
)

# Create a retrieval result
result = StringRetrievalResult(
    documents=[document],
    query="example query",
    processed_query="processed example query",
    total_results=1,
    execution_time_ms=10.5,
)

# Access result properties
print(f"Query: {result.query}")
print(f"Top document: {result.top_document.content}")
print(f"Score: {result.top_document.score}")
```
"""

from typing import Any, Dict, Generic, List, Optional, TypeVar
from pydantic import BaseModel, Field, ConfigDict

from sifaka.core.base import BaseResult

T = TypeVar("T")


class DocumentMetadata(BaseModel):
    """
    Metadata for a retrieved document.

    This class defines the metadata for a retrieved document,
    including its ID, source, and other attributes.

    ## Attributes

    - document_id: The ID of the document
    - source: The source of the document (e.g., file path, URL)
    - created_at: The creation timestamp of the document
    - updated_at: The last update timestamp of the document
    - additional_metadata: Additional metadata for the document

    ## Usage

    ```python
    metadata = DocumentMetadata(
        document_id="doc_1",
        source="example.txt",
        created_at="2023-01-01",
    )
    ```
    """

    model_config = ConfigDict(
        frozen=False,
        validate_assignment=True,
        extra="forbid",
        str_strip_whitespace=True,
    )

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

    def with_source(self, source: str) -> "DocumentMetadata":
        """
        Create a new metadata object with the specified source.

        Args:
            source: The source of the document

        Returns:
            A new DocumentMetadata object with the updated source
        """
        return self.model_copy(update={"source": source})

    def with_timestamp(
        self, created_at: Optional[str] = None, updated_at: Optional[str] = None
    ) -> "DocumentMetadata":
        """
        Create a new metadata object with the specified timestamps.

        Args:
            created_at: The creation timestamp
            updated_at: The update timestamp

        Returns:
            A new DocumentMetadata object with the updated timestamps
        """
        updates = {}
        if created_at is not None:
            updates["created_at"] = created_at
        if updated_at is not None:
            updates["updated_at"] = updated_at
        return self.model_copy(update=updates)


class RetrievedDocument(BaseModel, Generic[T]):
    """
    A retrieved document with content and metadata.

    This class defines a retrieved document, including its content,
    metadata, and relevance score.

    ## Attributes

    - content: The content of the document
    - metadata: Metadata for the document
    - score: The relevance score of the document

    ## Usage

    ```python
    document = RetrievedDocument(
        content="This is the document content.",
        metadata=DocumentMetadata(document_id="doc_1"),
        score=0.95,
    )
    ```
    """

    model_config = ConfigDict(
        frozen=False,
        validate_assignment=True,
        extra="forbid",
        arbitrary_types_allowed=True,
    )

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
        ge=0.0,
        le=1.0,
    )

    def with_score(self, score: float) -> "RetrievedDocument[T]":
        """
        Create a new document with the specified score.

        Args:
            score: The relevance score

        Returns:
            A new RetrievedDocument with the updated score
        """
        return self.model_copy(update={"score": score})


class RetrievalResult(BaseResult):
    """
    Result of a retrieval operation.

    This class defines the result of a retrieval operation,
    including the retrieved documents and query information.
    It extends BaseResult to provide a consistent result structure
    across the Sifaka framework.

    ## Attributes

    - documents: The retrieved documents
    - query: The query used for retrieval
    - processed_query: The processed query (after preprocessing)
    - total_results: The total number of results found
    - execution_time_ms: The execution time in milliseconds
    - additional_info: Additional information about the retrieval operation

    ## Usage

    ```python
    result = RetrievalResult(
        documents=[document1, document2],
        query="example query",
        processed_query="processed example query",
        total_results=2,
        execution_time_ms=10.5,
        passed=True,
        message="Successfully retrieved documents",
    )

    # Access top document
    top_doc = result.top_document

    # Get all contents
    contents = result.get_contents()
    ```
    """

    model_config = ConfigDict(
        frozen=False,
        validate_assignment=True,
        extra="forbid",
        arbitrary_types_allowed=True,
    )

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
        ge=0,
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

    @property
    def has_results(self) -> bool:
        """
        Check if the result has any documents.

        Returns:
            True if there are documents, False otherwise
        """
        return len(self.documents) > 0

    @property
    def average_score(self) -> Optional[float]:
        """
        Get the average score of all documents.

        Returns:
            The average score, or None if no documents have scores
        """
        scores = [doc.score for doc in self.documents if doc.score is not None]
        if not scores:
            return None
        return sum(scores) / len(scores)

    def get_contents(self) -> List[T]:
        """
        Get the contents of all retrieved documents.

        Returns:
            A list of document contents
        """
        return [doc.content for doc in self.documents]

    def get_documents_with_score_above(self, threshold: float) -> List[RetrievedDocument[T]]:
        """
        Get documents with a score above the threshold.

        Args:
            threshold: The score threshold

        Returns:
            A list of documents with scores above the threshold
        """
        return [doc for doc in self.documents if doc.score is not None and doc.score >= threshold]

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

    def with_additional_info(self, **kwargs: Any) -> "RetrievalResult[T]":
        """
        Create a new result with additional information.

        Args:
            **kwargs: Additional information to add

        Returns:
            A new RetrievalResult with the updated additional information
        """
        return self.model_copy(update={"additional_info": {**self.additional_info, **kwargs}})


class StringRetrievalResult(RetrievalResult):
    """
    Result of a retrieval operation with string content.

    This class specializes RetrievalResult for string content,
    which is the most common case for text retrieval.

    ## Usage

    ```python
    result = StringRetrievalResult(
        documents=[document1, document2],
        query="example query",
        processed_query="processed example query",
        total_results=2,
        execution_time_ms=10.5,
        passed=True,
        message="Successfully retrieved documents",
    )

    # Get concatenated content
    all_text = result.get_concatenated_content()
    ```
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

    def get_content_with_metadata(self, include_scores: bool = True) -> List[Dict[str, Any]]:
        """
        Get the content of all documents with their metadata.

        Args:
            include_scores: Whether to include scores in the output

        Returns:
            A list of dictionaries with content and metadata
        """
        result = []
        for doc in self.documents:
            item = {
                "content": doc.content,
                "metadata": doc.metadata.model_dump(),
            }
            if include_scores and doc.score is not None:
                item["score"] = doc.score
            result.append(item)
        return result
