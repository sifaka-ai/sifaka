"""
Result models for retrieval components.

This module provides result models for retrieval components in the Sifaka framework.
These models define the structure of results returned by retrievers and related components.

## Result Models

1. **DocumentMetadata**: Metadata for a retrieved document
2. **RetrievedDocument**: A retrieved document with content and metadata
3. **StringRetrievalResult**: Result of a retrieval operation with string content

## Usage Examples

```python
from sifaka.retrieval.result import StringRetrievalResult, RetrievedDocument, DocumentMetadata
from sifaka.core.results import RetrievalResult

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

from sifaka.core.results import RetrievalResult

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
        default_factory=lambda: DocumentMetadata(document_id="default"),
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

    # RetrievalResult is now imported from sifaka.core.results


class StringRetrievalResult(RetrievalResult[str]):
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
                "metadata": {
                    "document_id": doc.metadata.document_id,
                    "source": doc.metadata.source,
                    "created_at": doc.metadata.created_at,
                    "updated_at": doc.metadata.updated_at,
                    "additional_metadata": doc.metadata.additional_metadata,
                },
            }
            if include_scores and doc.score is not None:
                item["score"] = doc.score
            result.append(item)
        return result
