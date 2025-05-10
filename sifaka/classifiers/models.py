"""
Classifier Models Module

A module that provides data models for Sifaka classifiers.

## Overview
This module provides the core data models used by classifiers in the Sifaka framework,
including classification results, configuration, and related data structures. These
models ensure consistent data handling and type safety across the framework.

## Components
- ClassifierConfig: Configuration model for classifiers
- ClassificationResult: Result model for classification operations

## Usage Examples
```python
from sifaka.classifiers.models import ClassifierConfig, ClassificationResult

# Create a classifier configuration
config = ClassifierConfig[str](
    labels=["positive", "negative", "neutral"],
    min_confidence=0.7,
    cache_size=100,
    params={"threshold": 0.8}
)

# Create a classification result
result = ClassificationResult[str](
    label="positive",
    confidence=0.85,
    metadata={"scores": {"positive": 0.85, "negative": 0.10}}
)

# Update configuration
new_config = config.with_options(
    min_confidence=0.8,
    params={"new_param": "value"}
)

# Enhance result with metadata
enhanced_result = result.with_metadata(
    processing_time_ms=42,
    model_version="1.2.3"
)
```

## Error Handling
The models handle errors by:
- Validating input parameters (e.g., confidence range)
- Ensuring type safety through generics
- Providing clear error messages for invalid inputs

## Configuration
The models support various configuration options:
- Generic type parameters for flexibility
- Validation rules for critical fields
- Immutable result objects for data integrity
"""

from dataclasses import dataclass
from typing import Any, Dict, Generic, List, Optional, TypeVar, Union

from pydantic import BaseModel, ConfigDict, Field
from sifaka.core.base import BaseResult

# Type variables for generic models
R = TypeVar("R")  # Result label type
T = TypeVar("T")  # Input text type


class ClassifierConfig(BaseModel, Generic[T]):
    """
    Configuration for classifiers.

    This class represents the configuration for a classifier, including
    labels, minimum confidence threshold, and caching settings.

    ## Architecture
    The configuration follows a layered approach:
    - Base configuration with required fields
    - Optional parameters for customization
    - Validation rules for critical values

    ## Lifecycle
    1. Creation: Instantiate with required and optional parameters
    2. Validation: Parameters are validated on creation
    3. Modification: Create new instances with with_options()
    4. Usage: Pass to classifiers for configuration

    ## Error Handling
    - Validates required fields (e.g., labels)
    - Enforces value ranges (e.g., confidence)
    - Provides clear error messages

    ## Examples
    ```python
    from sifaka.classifiers.models import ClassifierConfig

    # Create basic configuration
    config = ClassifierConfig[str](
        labels=["positive", "negative"],
        min_confidence=0.7
    )

    # Create with all options
    full_config = ClassifierConfig[str](
        labels=["spam", "ham"],
        min_confidence=0.8,
        cache_size=100,
        params={"threshold": 0.9}
    )

    # Update configuration
    updated = config.with_options(
        min_confidence=0.8,
        params={"new_param": "value"}
    )
    ```

    Attributes:
        labels (List[str]): List of possible classification labels
        min_confidence (float): Minimum confidence threshold
        cache_size (int): Size of the classification cache
        params (Dict[str, Any]): Additional parameters
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

        Creates a new ClassifierConfig instance with the specified options
        updated, while preserving other settings from the current config.

        Args:
            **kwargs (Any): Options to update in the new config

        Returns:
            ClassifierConfig[T]: New configuration with updated options

        Example:
            ```python
            config = ClassifierConfig[str](
                labels=["positive", "negative"],
                min_confidence=0.7
            )

            # Update specific options
            updated = config.with_options(
                min_confidence=0.8,
                params={"threshold": 0.9}
            )
            ```
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


class ClassificationResult(BaseResult, Generic[R]):
    """
    Result of a classification operation.

    This class represents the result of a classification operation, including
    the predicted label, confidence score, and additional metadata.
    It extends BaseResult to provide a consistent result structure
    across the Sifaka framework.

    ## Architecture
    The result follows a structured pattern:
    - Extends BaseResult for consistency
    - Generic type parameter for label type
    - Structured metadata for extensibility

    ## Lifecycle
    1. Creation: Instantiate with required and optional parameters
    2. Access: Read properties to get classification details
    3. Enhancement: Create new instances with additional metadata

    ## Error Handling
    - Validates confidence range (0-1)
    - Type checks critical parameters
    - Inherits error handling from BaseResult

    ## Examples
    ```python
    from sifaka.classifiers.models import ClassificationResult

    # Create a result
    result = ClassificationResult(
        label="positive",
        confidence=0.85,
        passed=True,
        message="Classification successful",
        metadata={
            "scores": {"positive": 0.85, "negative": 0.10},
            "text_length": 120
        }
    )

    # Access properties
    print(f"Label: {result.label}")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"Passed: {result.passed}")

    # Add metadata
    enhanced = result.with_metadata(
        processing_time_ms=42,
        model_version="1.2.3"
    )

    # Add issues and suggestions
    with_issues = enhanced.with_issues(["Low confidence score"])
    with_suggestions = with_issues.with_suggestions(["Provide more training data"])
    ```

    Attributes:
        label (R): The predicted label/class
        confidence (float): Confidence score for the prediction (0-1)
        passed (bool): Whether the classification passed (inherited from BaseResult)
        message (str): Human-readable message (inherited from BaseResult)
        metadata (Dict[str, Any]): Additional metadata (inherited from BaseResult)
        issues (List[str]): List of issues (inherited from BaseResult)
        suggestions (List[str]): List of suggestions (inherited from BaseResult)
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    label: R = Field(description="The classification label")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score between 0 and 1")

    def with_metadata(self, **kwargs: Any) -> "ClassificationResult[R]":
        """
        Create a new result with additional metadata.

        Creates a new ClassificationResult with the same label, confidence, and other
        BaseResult properties, but with additional metadata merged with the existing metadata.

        Args:
            **kwargs (Any): Additional metadata to add to the result

        Returns:
            ClassificationResult[R]: New result with merged metadata

        Example:
            ```python
            result = ClassificationResult(
                label="positive",
                confidence=0.85,
                passed=True,
                message="Classification successful",
                metadata={"scores": {"positive": 0.85}}
            )

            # Add metadata
            enhanced = result.with_metadata(
                processing_time_ms=42,
                model_version="1.2.3"
            )

            # Chain metadata additions
            final = enhanced.with_metadata(
                timestamp="2023-07-01T12:34:56"
            )
            ```
        """
        # Create a new metadata dictionary with existing and new metadata
        new_metadata = {**self.metadata, **kwargs}

        # Return a new instance with the same properties but updated metadata
        return ClassificationResult(
            label=self.label,
            confidence=self.confidence,
            passed=self.passed,
            message=self.message,
            issues=self.issues,
            suggestions=self.suggestions,
            metadata=new_metadata,
        )
