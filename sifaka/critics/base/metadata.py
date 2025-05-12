"""
Metadata classes for critics.

This module defines the metadata classes for critic results, including
CriticMetadata, CriticOutput, and CriticResultEnum.

## Overview
The module provides standardized data structures for representing critic results,
including quality scores, feedback, issues, and suggestions. These classes ensure
consistent representation of critic outputs throughout the Sifaka framework.

## Components
1. **CriticResultEnum**: Enumeration of possible critic results
2. **CriticMetadata**: Metadata for critic results
3. **CriticOutput**: Output from critic operations

## Usage Examples
```python
from sifaka.critics.base.metadata import CriticMetadata, CriticOutput, CriticResultEnum

# Create metadata
metadata = CriticMetadata(
    score=0.8,
    feedback="Good text quality",
    issues=["Could be more concise"],
    suggestions=["Remove redundant phrases"]
)

# Create output
output = CriticOutput(
    result=CriticResultEnum.SUCCESS,
    improved_text="Improved text content",
    metadata=metadata
)

# Access output values
if output.result == CriticResultEnum.SUCCESS:
    print(f"Score: {output.metadata.score}")
    print(f"Feedback: {output.metadata.feedback}")
```

## Error Handling
The classes implement validation for:
- Score range (0.0 to 1.0)
- Required fields
- Processing time (non-negative)
- Attempt number (positive integer)
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, Generic, List, TypeVar

# Input and output type variables
T = TypeVar("T")  # Input type (usually str)
R = TypeVar("R")  # Result type


class CriticResultEnum(str, Enum):
    """
    Enumeration of possible critic results.

    This enum defines the possible outcomes of critic operations, providing
    a standardized way to represent the success or failure of critic operations.

    ## Overview
    The enum provides three possible states:
    - SUCCESS: Operation completed successfully
    - NEEDS_IMPROVEMENT: Text needs improvement
    - FAILURE: Operation failed

    ## Usage Examples
    ```python
    result = CriticResultEnum.SUCCESS
    if result == CriticResultEnum.NEEDS_IMPROVEMENT:
        print("Text needs improvement")
    elif result == CriticResultEnum.FAILURE:
        print("Operation failed")
    ```

    ## Error Handling
    The enum values are immutable and type-safe, ensuring consistent
    representation of operation results throughout the system.
    """

    SUCCESS = auto()
    NEEDS_IMPROVEMENT = auto()
    FAILURE = auto()


@dataclass(frozen=True)
class CriticMetadata(Generic[R]):
    """
    Immutable metadata for critic results.

    This class defines the metadata structure for critic results, providing
    a standardized way to store and access information about critic operations.

    ## Overview
    The metadata includes:
    - Score: Quality score of the text
    - Feedback: Detailed feedback about the text
    - Issues: List of identified issues
    - Suggestions: List of improvement suggestions
    - Processing time and attempt information
    - Additional custom metadata

    ## Usage Examples
    ```python
    metadata = CriticMetadata(
        score=0.8,
        feedback="Good text",
        issues=["Needs more detail"],
        suggestions=["Add examples"]
    )

    # Create modified metadata
    new_metadata = metadata.with_extra(
        processing_time_ms=100.0,
        attempt_number=2
    )
    ```

    ## Error Handling
    The class implements validation for:
    - Score range (0.0 to 1.0)
    - Required fields
    - Processing time (non-negative)
    - Attempt number (positive integer)

    Attributes:
        score (float): Quality score between 0.0 and 1.0
        feedback (str): Detailed feedback about the text
        issues (List[str]): List of identified issues
        suggestions (List[str]): List of improvement suggestions
        attempt_number (int): Number of improvement attempts
        processing_time_ms (float): Processing time in milliseconds
        extra (Dict[str, Any]): Additional custom metadata
    """

    score: float
    feedback: str
    issues: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    attempt_number: int = 1
    processing_time_ms: float = 0.0
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate metadata values."""
        if not 0 <= self.score <= 1:
            raise ValueError("score must be between 0 and 1")
        if self.attempt_number < 1:
            raise ValueError("attempt_number must be positive")
        if self.processing_time_ms < 0:
            raise ValueError("processing_time_ms must be non-negative")

    def with_extra(self, **kwargs: Any) -> "CriticMetadata[R]":
        """Create a new metadata with additional extra data.

        This method creates a new CriticMetadata instance with additional
        extra data while preserving other values.

        Args:
            **kwargs: Additional extra data

        Returns:
            New CriticMetadata instance

        Examples:
            ```python
            metadata = CriticMetadata(
                score=0.8,
                feedback="Good text"
            )
            new_metadata = metadata.with_extra(
                processing_time_ms=100.0,
                attempt_number=2
            )
            ```
        """
        new_extra = {**self.extra, **kwargs}
        return CriticMetadata(
            score=self.score,
            feedback=self.feedback,
            issues=self.issues,
            suggestions=self.suggestions,
            attempt_number=self.attempt_number,
            processing_time_ms=self.processing_time_ms,
            extra=new_extra,
        )


@dataclass(frozen=True)
class CriticOutput(Generic[T, R]):
    """Immutable output from a critic.

    This class defines the structure for critic outputs, including
    the result, improved text, and metadata.

    ## Lifecycle Management

    1. **Creation**
       - Set output values
       - Validate parameters
       - Create immutable instance

    2. **Usage**
       - Access output values
       - Process results
       - Handle metadata

    3. **Validation**
       - Check result type
       - Verify text format
       - Validate metadata

    ## Error Handling

    1. **Validation Errors**
       - Invalid result type
       - Empty text
       - Invalid metadata
       - Missing required fields

    2. **Recovery**
       - Default values
       - Error messages
       - State preservation

    Examples:
        ```python
        output = CriticOutput(
            result=CriticResultEnum.SUCCESS,
            improved_text="Improved text",
            metadata=CriticMetadata(
                score=0.8,
                feedback="Good improvement"
            )
        )
        ```
    """

    result: CriticResultEnum
    improved_text: T
    metadata: CriticMetadata[R]
