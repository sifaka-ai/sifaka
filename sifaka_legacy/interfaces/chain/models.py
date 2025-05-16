"""
Model classes for chain interfaces.

This module defines the model classes used by chain interfaces in the Sifaka framework.
These models provide standardized data structures for chain operations.

## Models

1. **ValidationResult**: Result of a validation operation
"""

from typing import Any, Dict, List

from pydantic import BaseModel, Field


class ValidationResult(BaseModel):
    """
    Result of a validation operation.

    This class represents the result of validating an output against a rule or
    validator. It includes information about whether the validation passed,
    a descriptive message, a numerical score, and optional lists of issues
    and improvement suggestions.

    ## Architecture
    The ValidationResult class is designed as an immutable Pydantic model
    with fields for storing validation results. It follows a value object
    pattern, where instances represent the outcome of a validation operation.

    ## Lifecycle
    1. Creation: Instantiated with validation outcome
    2. Usage: Accessed to retrieve validation status and details

    ## Examples
    ```python
    # Create a validation result
    result = ValidationResult(
        passed=True,
        message="Length validation passed",
        score=1.0
    )

    # Create a failed validation with issues and suggestions
    failed_result = ValidationResult(
        passed=False,
        message="Content validation failed",
        score=0.3,
        issues=["Contains inappropriate content"],
        suggestions=["Remove inappropriate content"]
    )

    # Check validation status
    if result.passed:
        print("Validation passed!")
    else:
        print(f"Validation failed: {result.message}")
        print(f"Issues: {result.issues}")
        print(f"Suggestions: {result.suggestions}")
    ```

    Attributes:
        passed (bool): Whether the validation passed
        message (str): Validation message
        score (float): Validation score (0.0 to 1.0)
        issues (List[str]): List of issues found
        suggestions (List[str]): List of improvement suggestions
        metadata (Dict[str, Any]): Additional metadata
    """

    passed: bool = Field(description="Whether the validation passed")
    message: str = Field(description="Validation message")
    score: float = Field(default=0.0, ge=0.0, le=1.0, description="Validation score")
    issues: List[str] = Field(default_factory=list, description="List of issues found")
    suggestions: List[str] = Field(
        default_factory=list, description="List of improvement suggestions"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )
