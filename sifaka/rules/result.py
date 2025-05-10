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

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, ConfigDict


class RuleResult(BaseModel):
    """
    Immutable result of a rule validation.

    This class provides a standardized way to represent validation results
    with metadata and optional scores. It is immutable to prevent accidental
    modification after creation and ensures consistent result handling
    throughout the Sifaka framework.

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

    model_config = ConfigDict(frozen=True, extra="forbid")

    passed: bool = Field(
        description="Whether the validation passed",
    )
    rule_name: str = Field(
        default="unnamed_rule",
        description="Name of the rule that produced this result",
    )
    message: str = Field(
        default="",
        description="Human-readable message describing the result",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata about the validation",
    )
    score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Optional score for the validation (0.0 to 1.0)",
    )
    issues: List[str] = Field(
        default_factory=list,
        description="List of issues identified during validation",
    )
    suggestions: List[str] = Field(
        default_factory=list,
        description="List of suggestions for fixing issues",
    )
    processing_time_ms: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Time taken to perform the validation in milliseconds",
    )

    def __bool__(self) -> bool:
        """
        Convert to boolean.

        Returns:
            True if validation passed, False otherwise

        Examples:
            ```python
            if result:
                print("Validation passed!")
            else:
                print("Validation failed!")
            ```
        """
        return self.passed

    def with_metadata(self, **metadata: Any) -> "RuleResult":
        """
        Create a new result with additional metadata.

        This method is useful for adding metadata to a result
        without modifying the original result.

        Args:
            **metadata: Metadata to add

        Returns:
            New result with additional metadata

        Examples:
            ```python
            # Add processing time metadata
            enhanced_result = result.with_metadata(
                processing_time_ms=42.5,
                validator_type="LengthValidator"
            )
            ```
        """
        new_metadata = {**self.metadata, **metadata}
        return RuleResult(
            passed=self.passed,
            rule_name=self.rule_name,
            message=self.message,
            metadata=new_metadata,
            score=self.score,
            issues=self.issues,
            suggestions=self.suggestions,
            processing_time_ms=self.processing_time_ms,
        )

    def with_issues(self, *new_issues: str) -> "RuleResult":
        """
        Create a new result with additional issues.

        Args:
            *new_issues: Issues to add

        Returns:
            New result with additional issues

        Examples:
            ```python
            # Add issues
            result_with_issues = result.with_issues(
                "Text contains prohibited content",
                "Text exceeds maximum length"
            )
            ```
        """
        issues = list(self.issues)
        issues.extend(new_issues)
        return RuleResult(
            passed=self.passed,
            rule_name=self.rule_name,
            message=self.message,
            metadata=self.metadata,
            score=self.score,
            issues=issues,
            suggestions=self.suggestions,
            processing_time_ms=self.processing_time_ms,
        )

    def with_suggestions(self, *new_suggestions: str) -> "RuleResult":
        """
        Create a new result with additional suggestions.

        Args:
            *new_suggestions: Suggestions to add

        Returns:
            New result with additional suggestions

        Examples:
            ```python
            # Add suggestions
            result_with_suggestions = result.with_suggestions(
                "Consider shortening the text",
                "Remove prohibited content"
            )
            ```
        """
        suggestions = list(self.suggestions)
        suggestions.extend(new_suggestions)
        return RuleResult(
            passed=self.passed,
            rule_name=self.rule_name,
            message=self.message,
            metadata=self.metadata,
            score=self.score,
            issues=self.issues,
            suggestions=suggestions,
            processing_time_ms=self.processing_time_ms,
        )
