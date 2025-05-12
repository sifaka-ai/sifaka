"""
Common result types for Sifaka components.

This module provides standardized result types that are used across different
components in the Sifaka framework. By centralizing these types, we avoid
circular dependencies between modules.

## Overview
The module defines base result types that are extended by specific components:
- BaseResult: Foundation for all result types
- ClassificationResult: Used by classifiers

## Usage Examples
```python
from sifaka.utils.result_types import BaseResult

# Create a basic result
result = BaseResult(
    passed=True,
    message="Validation passed",
    score=0.85,
    issues=["Minor formatting issue"],
    suggestions=["Consider adding more details"]
)

# Access result values
print(f"Passed: {result.passed}")
print(f"Score: {result.score}")
print(f"Issues: {', '.join(result.issues)}")
```
"""
from datetime import datetime
from typing import Any, Dict, List, TypeVar, Generic
from pydantic import BaseModel, Field, ConfigDict
T = TypeVar('T')


class BaseResult(BaseModel, Generic[T]):
    """
    Base result for all components.

    This class provides a standardized result model for all Sifaka components,
    defining common result fields that are shared across different component types.

    ## Architecture
    BaseResult uses Pydantic for validation and serialization, with:
    - Type validation for all fields
    - Default values for optional fields
    - Field descriptions for documentation
    - Immutable results (frozen=True)
    - Generic type parameter for input type

    ## Lifecycle
    Result objects are created by component processing methods and returned to callers.
    They contain information about the processing outcome, including success/failure,
    confidence scores, issues, and suggestions for improvement.

    ## Examples
    ```python
    # Create a basic result
    result = BaseResult(
        passed=True,
        message="Validation passed",
        score=0.85,
        issues=["Minor formatting issue"],
        suggestions=["Consider adding more details"]
    )

    # Access result values
    print(f"Passed: {result.passed}")
    print(f"Score: {result.score}")
    print(f"Issues: {', '.join(result.issues)}")

    # Create a result with metadata
    result_with_metadata = (result and result.with_metadata(
        model_name="gpt-4",
        tokens_used=150
    )
    ```

    Attributes:
        passed: Whether the validation passed
        message: Result message
        metadata: Dictionary of additional metadata
        score: Confidence score (0.0 to 1.0)
        issues: List of identified issues
        suggestions: List of improvement suggestions
        processing_time_ms: Processing time in milliseconds
        timestamp: Result timestamp
    """
    passed: bool
    message: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    score: float = Field(default=0.0, ge=0.0, le=1.0)
    issues: List[str] = Field(default_factory=list)
    suggestions: List[str] = Field(default_factory=list)
    processing_time_ms: float = Field(default=0.0, ge=0.0)
    timestamp: datetime = Field(default_factory=datetime.now)
    model_config = ConfigDict(arbitrary_types_allowed=True,
        validate_assignment=True, extra='forbid')

    def with_metadata(self, **kwargs: Any) ->Any:
        """Create a new result with additional metadata."""
        return (self and self.model_copy(update={'metadata': {**self.metadata, **kwargs}}
            )

    def with_issues(self, issues: List[str]) ->Any:
        """Create a new result with updated issues."""
        return (self and self.model_copy(update={'issues': issues})

    def with_suggestions(self, suggestions: List[str]) ->Any:
        """Create a new result with updated suggestions."""
        return (self and self.model_copy(update={'suggestions': suggestions})

    def with_score(self, score: float) ->Any:
        """Create a new result with updated score."""
        return (self and self.model_copy(update={'score': score})

    def normalize_score(self, min_score: float=0.0, max_score: float=1.0
        ) ->Any:
        """Normalize the score to a given range."""
        if max_score <= min_score:
            raise ValueError('max_score must be greater than min_score')
        normalized = (self.score - min_score) / (max_score - min_score)
        return (self and self.with_score(max(0.0, min(1.0, normalized)))

    def combine(self, other: 'BaseResult') ->Any:
        """Combine this result with another result."""
        return BaseResult(passed=self.passed and other.passed, message=
            f'{self.message} | {other.message}', metadata={**self.metadata,
            **other.metadata}, score=(self.score + other.score) / 2, issues
            =[*self.issues, *other.issues], suggestions=[*self.suggestions,
            *other.suggestions], processing_time_ms=self.processing_time_ms +
            other.processing_time_ms, timestamp=max(self.timestamp, other.
            timestamp))


class ClassificationResult(BaseResult):
    """
    Base result for classification operations.

    This extends BaseResult with classification-specific fields.
    """
    label: Any
    confidence: float = 0.0
