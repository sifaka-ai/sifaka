"""
Result creation utilities for Sifaka.

This module provides standardized result creation utilities for the Sifaka framework,
including functions for creating rule results, classification results, and critic results.

## Result Types

The module provides standardized result creation for different result types:

1. **RuleResult**: Results from rules and validators
2. **ClassificationResult**: Results from classifiers
3. **CriticMetadata**: Results from critics

## Result Creation

The module provides standardized result creation functions:

1. **create_rule_result**: Create a standardized rule result
2. **create_classification_result**: Create a standardized classification result
3. **create_critic_result**: Create a standardized critic result

## Usage Examples

```python
from sifaka.utils.results import (
    create_rule_result, create_classification_result, create_critic_result
)

# Create a rule result
result = create_rule_result(
    passed=True,
    message="Validation passed",
    component_name="LengthValidator",
    metadata={"text_length": 100}
)

# Create a classification result
result = create_classification_result(
    label="positive",
    confidence=0.85,
    component_name="SentimentClassifier",
    metadata={"pos_score": 0.85, "neg_score": 0.15}
)

# Create a critic result
result = create_critic_result(
    score=0.75,
    feedback="Good content, but could be more concise",
    component_name="ContentCritic",
    issues=["Too verbose", "Redundant information"]
)
```
"""

from typing import Any, Dict, List, Optional, TypeVar, Union, cast

from sifaka.rules.base import RuleResult
from sifaka.classifiers.base import ClassificationResult
from sifaka.critics.models import CriticMetadata
from sifaka.utils.common import create_standard_result


# Type variables for generic types
R = TypeVar("R")


def create_rule_result(
    passed: bool,
    message: str,
    component_name: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    severity: Optional[str] = None,
    rule_id: Optional[str] = None,
    cost: Optional[float] = None,
) -> RuleResult:
    """
    Create a standardized rule result.

    This function creates a standardized RuleResult with consistent metadata
    structure and formatting.

    Args:
        passed: Whether the validation passed
        message: Human-readable result message
        component_name: Name of the component that created the result
        metadata: Additional result metadata
        severity: Severity level of the result
        rule_id: Identifier for the rule
        cost: Computational cost of the validation

    Returns:
        Standardized RuleResult
    """
    # Create base metadata
    final_metadata: Dict[str, Any] = {}

    # Add component name if provided
    if component_name:
        final_metadata["component"] = component_name

    # Add severity if provided
    if severity:
        final_metadata["severity"] = severity

    # Add rule_id if provided
    if rule_id:
        final_metadata["rule_id"] = rule_id

    # Add cost if provided
    if cost:
        final_metadata["cost"] = cost

    # Add additional metadata
    if metadata:
        final_metadata.update(metadata)

    # Use the standardized result creation function
    standard_result = create_standard_result(
        output=None,  # Rule results don't have an output field
        metadata=final_metadata,
        success=passed,
        message=message,
    )

    # Create result using the standard result data
    return RuleResult(passed=passed, message=message, metadata=standard_result["metadata"])


def create_classification_result(
    label: R,
    confidence: float,
    component_name: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    threshold: Optional[float] = None,
    model_name: Optional[str] = None,
    processing_time: Optional[float] = None,
) -> ClassificationResult[R]:
    """
    Create a standardized classification result.

    This function creates a standardized ClassificationResult with consistent
    metadata structure and formatting.

    Args:
        label: Classification label
        confidence: Confidence score (0.0 to 1.0)
        component_name: Name of the component that created the result
        metadata: Additional result metadata
        threshold: Classification threshold used
        model_name: Name of the model used for classification
        processing_time: Time taken to perform the classification

    Returns:
        Standardized ClassificationResult
    """
    # Create base metadata
    final_metadata: Dict[str, Any] = {}

    # Add component name if provided
    if component_name:
        final_metadata["component"] = component_name

    # Add threshold if provided
    if threshold is not None:
        final_metadata["threshold"] = threshold

    # Add model name if provided
    if model_name:
        final_metadata["model_name"] = model_name

    # Add processing time if provided
    if processing_time is not None:
        final_metadata["processing_time"] = processing_time

    # Add additional metadata
    if metadata:
        final_metadata.update(metadata)

    # Use the standardized result creation function
    standard_result = create_standard_result(
        output=label,
        metadata=final_metadata,
        success=True,  # Classification results don't have a success/failure concept
        processing_time_ms=processing_time * 1000 if processing_time is not None else None,
    )

    # Create result using the standard result data
    return ClassificationResult[R](
        label=label, confidence=confidence, metadata=standard_result["metadata"]
    )


def create_critic_result(
    score: float,
    feedback: str,
    component_name: Optional[str] = None,
    issues: Optional[List[str]] = None,
    suggestions: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    model_name: Optional[str] = None,
    processing_time: Optional[float] = None,
) -> CriticMetadata:
    """
    Create a standardized critic result.

    This function creates a standardized CriticMetadata with consistent
    structure and formatting.

    Args:
        score: Quality score (0.0 to 1.0)
        feedback: Human-readable feedback
        component_name: Name of the component that created the result
        issues: List of identified issues
        suggestions: List of improvement suggestions
        metadata: Additional result metadata
        model_name: Name of the model used for critique
        processing_time: Time taken to perform the critique

    Returns:
        Standardized CriticMetadata
    """
    # Create base metadata
    final_metadata: Dict[str, Any] = {}

    # Add component name if provided
    if component_name:
        final_metadata["component"] = component_name

    # Add model name if provided
    if model_name:
        final_metadata["model_name"] = model_name

    # Add processing time if provided
    if processing_time is not None:
        final_metadata["processing_time"] = processing_time

    # Add additional metadata
    if metadata:
        final_metadata.update(metadata)

    # Use the standardized result creation function
    standard_result = create_standard_result(
        output=feedback,
        metadata=final_metadata,
        success=score >= 0.5,  # Consider scores above 0.5 as "success"
        message=feedback,
        processing_time_ms=processing_time * 1000 if processing_time is not None else None,
    )

    # Create result using the standard result data
    return CriticMetadata(
        score=score,
        feedback=feedback,
        issues=issues or [],
        suggestions=suggestions or [],
        metadata=standard_result["metadata"],
    )


def create_error_result(
    message: str,
    component_name: Optional[str] = None,
    error_type: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    severity: str = "error",
) -> RuleResult:
    """
    Create a standardized error result.

    This function creates a standardized RuleResult for error conditions
    with consistent metadata structure and formatting.

    Args:
        message: Human-readable error message
        component_name: Name of the component that created the result
        error_type: Type of error
        metadata: Additional result metadata
        severity: Severity level of the error

    Returns:
        Standardized RuleResult for error condition

    Examples:
        ```python
        from sifaka.utils.results import create_error_result

        # Create a basic error result
        result = create_error_result(
            message="Failed to process input",
            component_name="TextProcessor"
        )

        # Access result properties
        print(f"Passed: {result.passed}")  # Always False for error results
        print(f"Message: {result.message}")
        print(f"Error: {result.metadata.get('error')}")  # True
        print(f"Component: {result.metadata.get('component')}")

        # Create with error type
        result = create_error_result(
            message="Invalid input format",
            component_name="TextProcessor",
            error_type="ValidationError"
        )
        print(f"Error type: {result.metadata.get('error_type')}")

        # Create with custom severity
        result = create_error_result(
            message="Minor formatting issue",
            component_name="TextProcessor",
            severity="warning"
        )
        print(f"Severity: {result.metadata.get('severity')}")

        # Create with additional metadata
        result = create_error_result(
            message="Failed to process input",
            component_name="TextProcessor",
            metadata={
                "input_length": 100,
                "max_allowed_length": 50,
                "error_code": "E1001"
            }
        )
        print(f"Error code: {result.metadata.get('error_code')}")
        ```
    """
    # Create base metadata
    final_metadata: Dict[str, Any] = {"error": True, "severity": severity}

    # Add component name if provided
    if component_name:
        final_metadata["component"] = component_name

    # Add error type if provided
    if error_type:
        final_metadata["error_type"] = error_type

    # Add additional metadata
    if metadata:
        final_metadata.update(metadata)

    # Use the standardized result creation function
    standard_result = create_standard_result(
        output=None,
        metadata=final_metadata,
        success=False,  # Error results are always failures
        message=message,
    )

    # Create result using the standard result data
    return RuleResult(passed=False, message=message, metadata=standard_result["metadata"])


def create_unknown_result(
    component_name: Optional[str] = None,
    reason: str = "unknown",
    metadata: Optional[Dict[str, Any]] = None,
) -> ClassificationResult[str]:
    """
    Create a standardized unknown classification result.

    This function creates a standardized ClassificationResult for unknown
    or unclassifiable inputs.

    Args:
        component_name: Name of the component that created the result
        reason: Reason for the unknown classification
        metadata: Additional result metadata

    Returns:
        Standardized ClassificationResult for unknown classification

    Examples:
        ```python
        from sifaka.utils.results import create_unknown_result

        # Create a basic unknown result
        result = create_unknown_result(
            component_name="LanguageClassifier"
        )

        # Access result properties
        print(f"Label: {result.label}")  # Always "unknown"
        print(f"Confidence: {result.confidence}")  # Always 0.0
        print(f"Reason: {result.metadata.get('reason')}")  # "unknown" by default
        print(f"Component: {result.metadata.get('component')}")

        # Create with custom reason
        result = create_unknown_result(
            component_name="LanguageClassifier",
            reason="insufficient_text"
        )
        print(f"Reason: {result.metadata.get('reason')}")

        # Create with additional metadata
        result = create_unknown_result(
            component_name="LanguageClassifier",
            reason="insufficient_text",
            metadata={
                "text_length": 5,
                "min_required_length": 20,
                "language_hints": ["en", "fr"]
            }
        )
        print(f"Text length: {result.metadata.get('text_length')}")
        print(f"Language hints: {result.metadata.get('language_hints')}")

        # Use in a classifier
        def classify_language(text: str) -> ClassificationResult[str]:
            if not text or len(text) < 20:
                return create_unknown_result(
                    component_name="LanguageClassifier",
                    reason="insufficient_text",
                    metadata={"text_length": len(text)}
                )
            # Normal classification logic...
            return ClassificationResult(label="en", confidence=0.95)
        ```
    """
    # Create base metadata
    final_metadata: Dict[str, Any] = {"reason": reason}

    # Add component name if provided
    if component_name:
        final_metadata["component"] = component_name

    # Add additional metadata
    if metadata:
        final_metadata.update(metadata)

    # Use the standardized result creation function
    standard_result = create_standard_result(
        output="unknown",
        metadata=final_metadata,
        success=False,  # Unknown results are considered failures
        message=f"Unknown classification: {reason}",
    )

    # Create result using the standard result data
    return ClassificationResult[str](
        label="unknown", confidence=0.0, metadata=standard_result["metadata"]
    )


def merge_metadata(*metadata_dicts: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Merge multiple metadata dictionaries.

    This function merges multiple metadata dictionaries, with later
    dictionaries taking precedence over earlier ones for overlapping keys.

    Args:
        *metadata_dicts: Metadata dictionaries to merge

    Returns:
        Merged metadata dictionary

    Examples:
        ```python
        from sifaka.utils.results import merge_metadata

        # Merge two dictionaries
        metadata1 = {"component": "TextProcessor", "version": "1.0"}
        metadata2 = {"status": "success", "processing_time": 0.5}
        merged = merge_metadata(metadata1, metadata2)
        print(merged)  # {'component': 'TextProcessor', 'version': '1.0', 'status': 'success', 'processing_time': 0.5}

        # Handle overlapping keys (later dictionaries take precedence)
        metadata1 = {"component": "TextProcessor", "status": "pending"}
        metadata2 = {"status": "success", "processing_time": 0.5}
        merged = merge_metadata(metadata1, metadata2)
        print(merged)  # {'component': 'TextProcessor', 'status': 'success', 'processing_time': 0.5}

        # Handle None values
        metadata1 = {"component": "TextProcessor"}
        metadata2 = None
        metadata3 = {"status": "success"}
        merged = merge_metadata(metadata1, metadata2, metadata3)
        print(merged)  # {'component': 'TextProcessor', 'status': 'success'}

        # Use in result creation
        from sifaka.utils.results import create_rule_result

        base_metadata = {"component": "LengthValidator", "version": "1.0"}
        result_metadata = {"text_length": 100, "min_length": 50, "max_length": 200}

        result = create_rule_result(
            passed=True,
            message="Text length is acceptable",
            metadata=merge_metadata(base_metadata, result_metadata)
        )
        ```
    """
    result: Dict[str, Any] = {}

    # Process metadata dictionaries in order
    for metadata in metadata_dicts:
        if metadata:
            result.update(metadata)

    return result
