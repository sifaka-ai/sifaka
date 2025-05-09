"""
Data models for classifiers.

This module provides the data models used by classifiers in the Sifaka framework,
including classification results, configuration, and related data structures.
"""

from dataclasses import dataclass
from typing import Any, Dict, Generic, List, Optional, TypeVar, Union

from pydantic import BaseModel, ConfigDict, Field

# Type variables for generic models
R = TypeVar("R")  # Result label type
T = TypeVar("T")  # Input text type


class ClassifierConfig(BaseModel, Generic[T]):
    """
    Configuration for classifiers.

    This class represents the configuration for a classifier, including
    labels, minimum confidence threshold, and caching settings.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        from_attributes=True,
        validate_assignment=True,
    )

    labels: List[str] = Field(
        description="List of possible classification labels",
        min_length=1,
    )
    min_confidence: float = Field(
        default=0.5,
        description="Minimum confidence threshold for classification",
        ge=0.0,
        le=1.0,
    )
    cache_size: int = Field(
        default=0,
        description="Size of the classification cache (0 to disable)",
        ge=0,
    )
    params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional parameters for the classifier",
    )

    def with_options(self, **kwargs: Any) -> "ClassifierConfig[T]":
        """
        Create a new config with updated options.

        Args:
            **kwargs: Options to update

        Returns:
            New ClassifierConfig with updated options
        """
        # Create a copy of the current config
        data = self.model_dump()

        # Update with new options
        for key, value in kwargs.items():
            if key == "params" and isinstance(value, dict):
                # Merge params dictionaries
                data["params"] = {**data.get("params", {}), **value}
            else:
                # Set other options directly
                data[key] = value

        # Create a new config with the updated data
        return ClassifierConfig(**data)


@dataclass(frozen=True)
class ClassificationResult(Generic[R]):
    """
    Immutable result of a classification operation.

    This class represents the result of a classification operation, including
    the predicted label, confidence score, and additional metadata. It is
    designed to be immutable to ensure result integrity.

    ## Lifecycle

    1. **Creation**: Instantiate with required and optional parameters
       - Provide label and confidence (required)
       - Add optional metadata dictionary

    2. **Access**: Read properties to get classification details
       - Access label for the classification result
       - Read confidence for the prediction strength
       - Examine metadata for additional details

    3. **Enhancement**: Create new instances with additional metadata
       - Use with_metadata() to add new metadata
       - Original result remains unchanged (immutable)
       - Chain multiple with_metadata() calls for progressive enhancement

    ## Error Handling

    The class implements these error handling patterns:
    - Validation of confidence range (0-1)
    - Immutability to prevent result tampering
    - Type checking for critical parameters
    - Structured metadata for error details

    ## Examples

    Creating and using a classification result:

    ```python
    from sifaka.classifiers.models import ClassificationResult

    # Create a result
    result = ClassificationResult(
        label="positive",
        confidence=0.85,
        metadata={
            "scores": {"positive": 0.85, "negative": 0.10, "neutral": 0.05},
            "text_length": 120
        }
    )

    # Access the result
    if result.confidence > 0.8:
        print(f"High confidence classification: {result.label}")

    # Access metadata
    if "scores" in result.metadata:
        print(f"Negative score: {result.metadata['scores']['negative']:.2f}")

    # Add additional metadata
    enhanced = result.with_metadata(
        processing_time_ms=42,
        model_version="1.2.3"
    )

    print(f"Processing time: {enhanced.metadata['processing_time_ms']} ms")

    # Chain metadata additions
    enhanced = result.with_metadata(
        timestamp="2023-07-01T12:34:56"
    ).with_metadata(
        action_taken="content_removed"
    )

    print(f"Action taken: {enhanced.metadata['action_taken']}")
    print(f"Timestamp: {enhanced.metadata['timestamp']}")
    ```

    Attributes:
        label: The predicted label/class
        confidence: Confidence score for the prediction (0-1)
        metadata: Additional metadata about the classification
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)  # Immutable model

    label: R = Field(description="The classification label")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score between 0 and 1")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    def with_metadata(self, **kwargs: Any) -> "ClassificationResult[R]":
        """
        Create a new result with additional metadata.

        This method creates a new ClassificationResult with the same label and confidence,
        but with additional metadata merged with the existing metadata. The original
        result remains unchanged due to immutability.

        Args:
            **kwargs: Additional metadata to add to the result

        Returns:
            A new ClassificationResult with merged metadata
        """
        # Create a new metadata dictionary with existing and new metadata
        new_metadata = {**self.metadata, **kwargs}

        # Return a new instance with the same label and confidence but updated metadata
        return ClassificationResult(
            label=self.label, confidence=self.confidence, metadata=new_metadata
        )
