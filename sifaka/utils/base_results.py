"""
Base Result Classes for Sifaka

This module provides base result classes that all component-specific results inherit from.
It defines the common structure and behavior for results across the Sifaka framework.

The module is designed to avoid circular dependencies by providing base classes that
component-specific result classes can inherit from, without importing from component modules.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, TypeVar, Generic

from pydantic import BaseModel, Field, ConfigDict

# Type variables
T = TypeVar("T")  # Input type


class BaseResult(BaseModel, Generic[T]):
    """
    Base result for all components.

    This class provides a standardized result model for all Sifaka components,
    defining common result fields that are shared across different component types.

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

    model_config = ConfigDict(
        arbitrary_types_allowed=True, validate_assignment=True, extra="forbid"
    )

    def with_metadata(self, **kwargs: Any) -> "BaseResult":
        """Create a new result with additional metadata."""
        return (self and self.model_copy(update={"metadata": {**self.metadata, **kwargs}})

    def with_issues(self, issues: List[str]) -> "BaseResult":
        """Create a new result with updated issues."""
        return (self and self.model_copy(update={"issues": issues})

    def with_suggestions(self, suggestions: List[str]) -> "BaseResult":
        """Create a new result with updated suggestions."""
        return (self and self.model_copy(update={"suggestions": suggestions})

    def with_score(self, score: float) -> "BaseResult":
        """Create a new result with updated score."""
        return (self and self.model_copy(update={"score": score})


class BaseRuleResult(BaseResult):
    """
    Base result for rule validation.

    This class extends BaseResult to add rule-specific result information.

    Attributes:
        rule_name: Name of the rule that produced this result
        severity: Severity level of the result (error, warning, info)
        category: Category of the rule that produced this result
        tags: List of tags associated with the rule
        rule_id: Identifier of the rule that produced this result
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

    def with_rule_id(self, rule_id: str) -> "BaseRuleResult":
        """Create a new result with the rule ID set."""
        return (self and self.model_copy(update={"rule_id": rule_id})

    def with_severity(self, severity: str) -> "BaseRuleResult":
        """Create a new result with updated severity."""
        return (self and self.model_copy(update={"severity": severity})

    def with_category(self, category: str) -> "BaseRuleResult":
        """Create a new result with updated category."""
        return (self and self.model_copy(update={"category": category})

    def with_tags(self, tags: List[str]) -> "BaseRuleResult":
        """Create a new result with updated tags."""
        return (self and self.model_copy(update={"tags": tags})


class CriticMetadata(BaseResult):
    """
    Metadata for critic results.

    This class provides a standardized structure for critic metadata,
    including scores, feedback, issues, and suggestions.
    It extends BaseResult to provide a consistent result structure
    across the Sifaka framework.

    ## Overview
    The class provides:
    - Score tracking (0.0 to 1.0)
    - Human-readable feedback
    - Issue identification
    - Improvement suggestions
    - Additional metadata storage

    ## Usage Examples
    ```python
    from sifaka.utils.base_results import CriticMetadata

    # Create basic metadata
    metadata = CriticMetadata(
        score=0.85,
        feedback="Good text quality",
        passed=True,
        message="Critique completed successfully",
        issues=["Could be more concise"],
        suggestions=["Remove redundant phrases"]
    )

    # Create metadata with additional data
    metadata = CriticMetadata(
        score=0.75,
        feedback="Text needs improvement",
        passed=False,
        message="Text needs improvement",
        issues=["Too verbose", "Unclear structure"],
        suggestions=["Simplify language", "Add clear sections"],
        metadata={
            "processing_time": 1.5,
            "confidence": 0.9
        }
    )
    ```

    ## Error Handling
    The class implements:
    - Score range validation (0.0 to 1.0)
    - Required field validation
    - Type checking for all fields
    - Default value handling

    Attributes:
        score: Score for the critique (0.0 to 1.0)
        feedback: Human-readable feedback
        issues: List of identified issues (inherited from BaseResult)
        suggestions: List of improvement suggestions (inherited from BaseResult)
        metadata: Additional metadata (inherited from BaseResult)
        passed: Whether the critique passed (inherited from BaseResult)
        message: Human-readable message (inherited from BaseResult)
    """

    score: float = Field(
        ge=0.0,
        le=1.0,
        description="Score for the critique (0.0 to 1.0)",
    )
    feedback: str = Field(
        description="Human-readable feedback",
    )


def def create_base_result(
    passed: bool,
    message: str,
    metadata: Optional[Dict[str, Any]] = None,
    score: float = 0.0,
    issues: Optional[Optional[List[str]]] = None,
    suggestions: Optional[Optional[List[str]]] = None,
    processing_time_ms: Optional[Optional[float]] = None,
) -> BaseResult:
    """
    Create a standardized base result.

    Args:
        passed: Whether the validation passed
        message: Human-readable result message
        metadata: Additional result metadata
        score: Confidence score (0.0 to 1.0)
        issues: List of identified issues
        suggestions: List of improvement suggestions
        processing_time_ms: Processing time in milliseconds

    Returns:
        Standardized BaseResult
    """
    return BaseResult(
        passed=passed,
        message=message,
        metadata=metadata or {},
        score=score,
        issues=issues or [],
        suggestions=suggestions or [],
        processing_time_ms=processing_time_ms or 0.0,
    )


def def create_base_rule_result(
    passed: bool,
    message: str,
    rule_name: str = "unnamed_rule",
    metadata: Optional[Dict[str, Any]] = None,
    severity: str = "error",
    category: str = "general",
    tags: Optional[Optional[List[str]]] = None,
    rule_id: Optional[Optional[str]] = None,
    score: float = 0.0,
    issues: Optional[Optional[List[str]]] = None,
    suggestions: Optional[Optional[List[str]]] = None,
    processing_time_ms: Optional[Optional[float]] = None,
) -> BaseRuleResult:
    """
    Create a standardized base rule result.

    Args:
        passed: Whether the validation passed
        message: Human-readable result message
        rule_name: Name of the rule that produced this result
        metadata: Additional result metadata
        severity: Severity level of the result
        category: Category of the rule that produced this result
        tags: List of tags associated with the rule
        rule_id: Identifier of the rule that produced this result
        score: Confidence score (0.0 to 1.0)
        issues: List of identified issues
        suggestions: List of improvement suggestions
        processing_time_ms: Processing time in milliseconds

    Returns:
        Standardized BaseRuleResult
    """
    return BaseRuleResult(
        passed=passed,
        message=message,
        rule_name=rule_name,
        metadata=metadata or {},
        severity=severity,
        category=category,
        tags=tags or [],
        rule_id=rule_id,
        score=score,
        issues=issues or [],
        suggestions=suggestions or [],
        processing_time_ms=processing_time_ms or 0.0,
    )


def def create_critic_metadata(
    score: float,
    feedback: str,
    passed: bool,
    message: str,
    metadata: Optional[Dict[str, Any]] = None,
    issues: Optional[Optional[List[str]]] = None,
    suggestions: Optional[Optional[List[str]]] = None,
    processing_time_ms: Optional[Optional[float]] = None,
) -> CriticMetadata:
    """
    Create a standardized critic metadata result.

    Args:
        score: Score for the critique (0.0 to 1.0)
        feedback: Human-readable feedback
        passed: Whether the critique passed
        message: Human-readable message
        metadata: Additional metadata
        issues: List of identified issues
        suggestions: List of improvement suggestions
        processing_time_ms: Processing time in milliseconds

    Returns:
        Standardized CriticMetadata
    """
    return CriticMetadata(
        score=score,
        feedback=feedback,
        passed=passed,
        message=message,
        metadata=metadata or {},
        issues=issues or [],
        suggestions=suggestions or [],
        processing_time_ms=processing_time_ms or 0.0,
    )
