"""
Rule result models for Sifaka.

This module defines the result models for rules in the Sifaka framework.
These models provide a standardized way to represent validation results.

Usage Example:
    ```python
    from sifaka.rules.result import RuleResult

    # Create a basic result
    result = RuleResult(
        passed=True,
        rule_name="length_rule",
        message="Text length is within acceptable range",
        metadata={"length": 100, "min_length": 10, "max_length": 1000},
        score=0.8
    )

    # Check if validation passed
    if result:
        print("Validation passed!")

    # Add additional metadata
    enhanced_result = result.with_metadata(
        processing_time_ms=42.5,
        validator_type="LengthValidator"
    )
    ```
"""

from typing import Any, Dict, List, Optional

from pydantic import Field
from sifaka.core.base import BaseResult


class RuleResult(BaseResult):
    """
    Immutable result of a rule validation.

    This class extends BaseResult to add rule-specific result information.
    It is immutable to prevent accidental modification after creation and
    ensures consistent result handling throughout the Sifaka framework.

    Lifecycle:
        1. Creation: Instantiated with validation outcome
           - Set passed/failed status
           - Provide descriptive message
           - Include relevant metadata

        2. Usage: Accessed by rules and handlers
           - Check validation status
           - Process result message
           - Analyze metadata

        3. Augmentation: New instances created with additional metadata
           - Create enhanced results with with_metadata()
           - Add rule-specific information
           - Include performance metrics

    Examples:
        ```python
        from sifaka.rules.result import RuleResult

        # Create a basic result
        result = RuleResult(
            passed=True,
            rule_name="length_rule",
            message="Text length is within acceptable range",
            metadata={"length": 100, "min_length": 10, "max_length": 1000},
            score=0.8
        )

        # Check if validation passed
        if result:
            print("Validation passed!")
        else:
            print(f"Validation failed: {result.message}")

        # Access metadata
        print(f"Text length: {result.metadata.get('length')}")
        ```
    """

    rule_name: str = Field(
        default="unnamed_rule",
        description="Name of the rule that produced this result",
    )
    severity: str = Field(
        default="error",
        description="Severity level of the result",
    )
    category: str = Field(
        default="general",
        description="Category of the rule that produced this result",
    )
    tags: List[str] = Field(
        default_factory=list,
        description="List of tags associated with the rule",
    )
    rule_id: Optional[str] = Field(
        default=None,
        description="Identifier of the rule that produced this result",
    )

    def with_rule_id(self, rule_id: str) -> "RuleResult":
        """Create a new result with the rule ID set."""
        return self.model_copy(update={"rule_id": rule_id})

    def with_severity(self, severity: str) -> "RuleResult":
        """Create a new result with updated severity."""
        return self.model_copy(update={"severity": severity})

    def with_category(self, category: str) -> "RuleResult":
        """Create a new result with updated category."""
        return self.model_copy(update={"category": category})

    def with_tags(self, tags: List[str]) -> "RuleResult":
        """Create a new result with updated tags."""
        return self.model_copy(update={"tags": tags})


def create_rule_result(
    passed: bool,
    message: str,
    component_name: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    severity: str = "error",
    category: str = "general",
    tags: Optional[List[str]] = None,
    rule_id: Optional[str] = None,
    score: float = 0.0,
    issues: Optional[List[str]] = None,
    suggestions: Optional[List[str]] = None,
    processing_time_ms: Optional[float] = None,
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
        category: Category of the rule
        tags: List of tags for the rule
        rule_id: Identifier for the rule
        score: Confidence score (0.0 to 1.0)
        issues: List of identified issues
        suggestions: List of improvement suggestions
        processing_time_ms: Processing time in milliseconds

    Returns:
        Standardized RuleResult
    """
    # Create base metadata
    final_metadata: Dict[str, Any] = {}

    # Add component name if provided
    if component_name:
        final_metadata["component"] = component_name

    # Add additional metadata
    if metadata:
        final_metadata.update(metadata)

    # Create result
    return RuleResult(
        passed=passed,
        message=message,
        rule_name=component_name or "unnamed_rule",
        metadata=final_metadata,
        severity=severity,
        category=category,
        tags=tags or [],
        rule_id=rule_id,
        score=score,
        issues=issues or [],
        suggestions=suggestions or [],
        processing_time_ms=processing_time_ms or 0.0,
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
    """
    # Create base metadata
    final_metadata: Dict[str, Any] = {"error": True}

    # Add component name if provided
    if component_name:
        final_metadata["component"] = component_name

    # Add error type if provided
    if error_type:
        final_metadata["error_type"] = error_type

    # Add additional metadata
    if metadata:
        final_metadata.update(metadata)

    # Create error issues and suggestions
    issues = ["Error occurred during validation"]
    if error_type:
        issues.append(f"Error type: {error_type}")

    suggestions = ["Check input and try again"]

    # Create result
    return RuleResult(
        passed=False,
        message=message,
        rule_name=component_name or "error_result",
        metadata=final_metadata,
        severity=severity,
        category="error",
        tags=["error"],
        score=0.0,
        issues=issues,
        suggestions=suggestions,
    )
