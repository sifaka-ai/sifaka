"""Modern Pydantic models for critic results and feedback.

This module provides clean, structured Pydantic models for the new critic
architecture. These models completely replace the old dictionary-based
approach and string parsing logic with type-safe, validated structures.

The models are designed to work seamlessly with PydanticAI's structured output
capabilities and provide rich metadata for analysis and debugging.

Key Features:
- Full type safety with Pydantic v2
- Rich validation and error handling
- Structured violations and suggestions
- Detailed confidence scoring
- Performance metrics tracking
- No backwards compatibility (clean slate)
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, model_validator


class SeverityLevel(str, Enum):
    """Severity levels for violations and issues."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class ConfidenceScore(BaseModel):
    """Detailed confidence score with breakdowns.

    This model provides a structured way to represent confidence scores
    with detailed breakdowns of different confidence factors.
    """

    overall: float = Field(
        ..., ge=0.0, le=1.0, description="Overall confidence score between 0.0 and 1.0"
    )

    # Detailed confidence breakdowns
    content_quality: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Confidence in content quality assessment"
    )

    grammar_accuracy: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Confidence in grammar and language assessment"
    )

    factual_accuracy: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Confidence in factual accuracy assessment"
    )

    coherence: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Confidence in coherence and flow assessment"
    )

    # Metadata about confidence calculation
    calculation_method: Optional[str] = Field(
        None, description="Method used to calculate confidence"
    )

    factors_considered: List[str] = Field(
        default_factory=list, description="List of factors considered in confidence calculation"
    )

    uncertainty_sources: List[str] = Field(
        default_factory=list, description="Sources of uncertainty that lower confidence"
    )

    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional confidence-related metadata"
    )

    class Config:
        """Pydantic configuration."""

        frozen = True


class ViolationReport(BaseModel):
    """Structured report of a specific violation or issue.

    This model represents a single violation or issue found during critique,
    with severity level, location information, and suggested fixes.
    """

    violation_type: str = Field(..., description="Type or category of the violation")

    description: str = Field(..., description="Detailed description of the violation")

    severity: SeverityLevel = Field(..., description="Severity level of the violation")

    # Location information (optional)
    location: Optional[str] = Field(
        None, description="Location in text where violation occurs (e.g., 'paragraph 2', 'line 15')"
    )

    start_position: Optional[int] = Field(
        None, ge=0, description="Character position where violation starts"
    )

    end_position: Optional[int] = Field(
        None, ge=0, description="Character position where violation ends"
    )

    # Violation details
    rule_violated: Optional[str] = Field(
        None, description="Specific rule or principle that was violated"
    )

    evidence: Optional[str] = Field(None, description="Text evidence supporting the violation")

    suggested_fix: Optional[str] = Field(
        None, description="Suggested fix for this specific violation"
    )

    # Confidence and metadata
    confidence: float = Field(
        1.0, ge=0.0, le=1.0, description="Confidence in this violation report"
    )

    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional violation-specific metadata"
    )

    @model_validator(mode="after")
    def validate_position_order(self):
        """Validate that end_position is after start_position."""
        if (
            self.end_position is not None
            and self.start_position is not None
            and self.end_position <= self.start_position
        ):
            raise ValueError("end_position must be greater than start_position")
        return self

    class Config:
        """Pydantic configuration."""

        frozen = True


class ImprovementSuggestion(BaseModel):
    """Structured improvement suggestion with metadata.

    This model represents a specific suggestion for improving the text,
    with priority, rationale, and implementation details.
    """

    suggestion: str = Field(..., description="The improvement suggestion text")

    category: str = Field(
        ..., description="Category of improvement (e.g., 'grammar', 'clarity', 'structure')"
    )

    priority: SeverityLevel = Field(
        SeverityLevel.MEDIUM, description="Priority level for this suggestion"
    )

    # Implementation details
    rationale: Optional[str] = Field(
        None, description="Explanation of why this improvement is suggested"
    )

    implementation: Optional[str] = Field(
        None, description="Specific steps or changes to implement this suggestion"
    )

    example: Optional[str] = Field(None, description="Example of how the improvement would look")

    # Location information (optional)
    applies_to: Optional[str] = Field(None, description="Part of text this suggestion applies to")

    start_position: Optional[int] = Field(
        None, ge=0, description="Character position where suggestion applies (start)"
    )

    end_position: Optional[int] = Field(
        None, ge=0, description="Character position where suggestion applies (end)"
    )

    # Impact and confidence
    expected_impact: Optional[str] = Field(
        None, description="Expected impact of implementing this suggestion"
    )

    confidence: float = Field(1.0, ge=0.0, le=1.0, description="Confidence in this suggestion")

    # Dependencies and conflicts
    depends_on: List[str] = Field(
        default_factory=list, description="Other suggestions this one depends on"
    )

    conflicts_with: List[str] = Field(
        default_factory=list, description="Other suggestions this one conflicts with"
    )

    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional suggestion-specific metadata"
    )

    @model_validator(mode="after")
    def validate_position_order(self):
        """Validate that end_position is after start_position."""
        if (
            self.end_position is not None
            and self.start_position is not None
            and self.end_position <= self.start_position
        ):
            raise ValueError("end_position must be greater than start_position")
        return self

    class Config:
        """Pydantic configuration."""

        frozen = True


class CritiqueFeedback(BaseModel):
    """Modern structured critique feedback model.

    This model provides clean, type-safe feedback from critics with
    structured violations, suggestions, and rich metadata for analysis.
    """

    # Core feedback
    message: str = Field(..., description="Main critique message or summary")

    needs_improvement: bool = Field(
        ..., description="Whether the text needs improvement based on this critique"
    )

    # Structured feedback components
    violations: List[ViolationReport] = Field(
        default_factory=list, description="Specific violations or issues found"
    )

    suggestions: List[ImprovementSuggestion] = Field(
        default_factory=list, description="Specific improvement suggestions"
    )

    # Confidence and scoring
    confidence: ConfidenceScore = Field(..., description="Detailed confidence information")

    # Critic information
    critic_name: str = Field(..., description="Name of the critic that generated this feedback")

    critic_version: Optional[str] = Field(
        None, description="Version of the critic (for tracking changes)"
    )

    # Processing information
    processing_time_ms: Optional[float] = Field(
        None, ge=0.0, description="Time taken to generate this feedback in milliseconds"
    )

    timestamp: datetime = Field(
        default_factory=datetime.now, description="When this feedback was generated"
    )

    # Rich metadata
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional critic-specific metadata"
    )

    class Config:
        """Pydantic configuration."""

        frozen = True


class CriticResult(BaseModel):
    """Main result container for critic operations with validation.

    This is the primary model that all critics should return, providing
    a standardized interface for critic results with rich validation
    and metadata support.
    """

    # Core result data
    feedback: CritiqueFeedback = Field(..., description="Structured critique feedback")

    # Operation metadata
    operation_type: str = Field(
        ..., description="Type of operation performed (e.g., 'critique', 'improve')"
    )

    success: bool = Field(True, description="Whether the operation completed successfully")

    # Error information (if applicable)
    error_message: Optional[str] = Field(None, description="Error message if operation failed")

    error_type: Optional[str] = Field(None, description="Type of error that occurred")

    # Performance metrics
    total_processing_time_ms: float = Field(
        ..., ge=0.0, description="Total time taken for the operation in milliseconds"
    )

    model_calls: int = Field(0, ge=0, description="Number of model API calls made")

    tokens_used: Optional[int] = Field(None, ge=0, description="Total tokens used (if available)")

    # Context information
    input_text_length: int = Field(..., ge=0, description="Length of input text in characters")

    input_hash: Optional[str] = Field(
        None, description="Hash of input text for caching/deduplication"
    )

    # Validation context (if available)
    validation_context: Optional[Dict[str, Any]] = Field(
        None, description="Validation context that influenced the critique"
    )

    # Rich metadata
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional result metadata")

    # Timestamp
    created_at: datetime = Field(
        default_factory=datetime.now, description="When this result was created"
    )

    @model_validator(mode="after")
    def validate_result_consistency(self):
        """Validate that the result is internally consistent."""
        # For failed operations, ensure feedback reflects the failure
        if not self.success and not self.feedback.needs_improvement:
            raise ValueError("Failed operations should indicate improvement is needed")

        # Ensure error message is provided for failed operations
        if not self.success and not self.error_message:
            raise ValueError("Error message is required when success=False")

        return self

    class Config:
        """Pydantic configuration."""

        frozen = True
