"""
Classification Result Module

This module provides the ClassificationResult class for representing the results of classification.
It standardizes the format of classification results, including label, confidence, and metadata.

## Components
1. **ClassificationResult**: Represents the result of classification

## Usage Examples
```python
from sifaka.classifiers.result import ClassificationResult

# Create a classification result
result = ClassificationResult(
    label="positive",
    confidence=0.85,
    metadata={"scores": {"positive": 0.85, "negative": 0.10}},
    issues=["Low confidence for negative class"],
    suggestions=["Provide more training data for negative class"]
)

# Access result properties
print(f"Label: {result.label}")
print(f"Confidence: {result.confidence:.2f}")

# Access metadata
for key, value in result.metadata.items():
    print(f"{key}: {value}")

# Access issues and suggestions
for issue in result.issues:
    print(f"Issue: {issue}")
for suggestion in result.suggestions:
    print(f"Suggestion: {suggestion}")

# Create a new result with additional metadata
enhanced_result = result.with_metadata(
    processing_time_ms=42,
    model_version="1.2.3"
)
```
"""

from typing import Any, Dict, List, Optional
import time
from pydantic import BaseModel, Field, computed_field


class ClassificationResult(BaseModel):
    """Result of a classification operation."""

    label: str = Field(description="The classification label")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence score (0-1)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    issues: List[str] = Field(default_factory=list, description="List of issues found")
    suggestions: List[str] = Field(
        default_factory=list, description="List of improvement suggestions"
    )
    timestamp: float = Field(default_factory=time.time, description="Result creation timestamp")

    def with_metadata(self, **kwargs: Any) -> "ClassificationResult":
        """
        Create a new result with additional metadata.

        Args:
            **kwargs: Metadata key-value pairs

        Returns:
            New result with updated metadata
        """
        return self.model_copy(update={"metadata": {**self.metadata, **kwargs}})

    def with_issues(self, issues: List[str]) -> "ClassificationResult":
        """
        Create a new result with additional issues.

        Args:
            issues: List of issues to add

        Returns:
            New result with updated issues
        """
        return self.model_copy(update={"issues": self.issues + issues})

    def with_suggestions(self, suggestions: List[str]) -> "ClassificationResult":
        """
        Create a new result with additional suggestions.

        Args:
            suggestions: List of suggestions to add

        Returns:
            New result with updated suggestions
        """
        return self.model_copy(update={"suggestions": self.suggestions + suggestions})

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert result to dictionary.

        Returns:
            Dictionary representation of the result
        """
        return {
            "label": self.label,
            "confidence": self.confidence,
            "metadata": self.metadata,
            "issues": self.issues,
            "suggestions": self.suggestions,
            "timestamp": self.timestamp,
        }
