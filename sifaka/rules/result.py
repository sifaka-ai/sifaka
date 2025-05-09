"""
Rule result models for Sifaka.

This module defines the result models for rules in the Sifaka framework.
These models provide a standardized way to represent validation results.
"""

from typing import Any, Dict, Optional

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
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    passed: bool = Field(
        description="Whether the validation passed",
    )
    rule_name: str = Field(
        description="Name of the rule that produced this result",
    )
    message: str = Field(
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

    def __bool__(self) -> bool:
        """
        Convert to boolean.

        Returns:
            True if validation passed, False otherwise
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
        """
        new_metadata = {**self.metadata, **metadata}
        return RuleResult(
            passed=self.passed,
            rule_name=self.rule_name,
            message=self.message,
            metadata=new_metadata,
            score=self.score,
        )
