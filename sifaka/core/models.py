"""Core data models for Sifaka's text improvement system.

This module defines the primary data structures used throughout Sifaka:
- Generation: Individual text generations with metadata
- ToolUsage: Tracking of tool/API calls during processing
- ValidationResult: Results from quality validators
- CritiqueResult: Feedback from critic evaluations
- SifakaResult: Complete result container with audit trail

All models use Pydantic for validation and include automatic memory
bounds to prevent unbounded growth during long-running sessions."""

import uuid
from collections import deque
from datetime import datetime
from typing import Any, Deque, Dict, List, Optional

from pydantic import BaseModel, Field


class Generation(BaseModel):
    """Represents a single text generation attempt with full metadata.

    Each generation captures the text produced, the model used, timing
    information, and the prompt that created it. This provides full
    traceability for understanding how text evolved.

    Attributes:
        text: The generated text content
        model: Name of the LLM model used (e.g., 'gpt-4', 'claude-3')
        iteration: Which improvement iteration this generation belongs to
        prompt: The exact prompt sent to the LLM (optional, for debugging)
        timestamp: When this generation was created
        tokens_used: Number of tokens consumed by this generation
        processing_time: Time in seconds to generate this text
    """

    text: str
    model: str
    iteration: int
    prompt: Optional[str] = None  # The prompt used to generate this text
    timestamp: datetime = Field(default_factory=datetime.now)
    tokens_used: int = 0
    processing_time: float = 0.0


class ToolUsage(BaseModel):
    """Tracks usage of external tools or APIs during processing.

    This generic structure captures tool calls made by critics like
    self_rag (which may use search/retrieval tools) without assuming
    specific tool interfaces.

    Attributes:
        tool_name: Name of the tool or API called
        status: 'success' or 'failure' status of the call
        input_data: String representation of tool input (optional)
        result_count: Number of results returned, if applicable
        error_message: Error details if the tool call failed
        timestamp: When the tool was called
        processing_time: Time in seconds for the tool call
        metadata: Additional tool-specific data as key-value pairs
    """

    tool_name: str
    status: str = Field(description="success or failure")
    input_data: Optional[str] = Field(
        default=None, description="Tool input without assuming structure"
    )
    result_count: Optional[int] = Field(
        default=None, description="Number of results if applicable"
    )
    error_message: Optional[str] = Field(
        default=None, description="Error details if failed"
    )
    timestamp: datetime = Field(default_factory=datetime.now)
    processing_time: float = 0.0
    metadata: Optional[Dict[str, Any]] = Field(
        default=None, description="Additional tool-specific metadata"
    )


class ValidationResult(BaseModel):
    """Result from a quality validation check.

    Validators ensure text meets specific criteria like minimum length,
    appropriate content, or custom business rules.

    Attributes:
        validator: Name of the validator that ran
        passed: Whether the validation passed (True) or failed (False)
        score: Optional numeric score (0.0-1.0) for scoring validators
        details: Human-readable explanation of the validation result
        timestamp: When this validation was performed
    """

    validator: str
    passed: bool
    score: Optional[float] = None
    details: str = ""
    timestamp: datetime = Field(default_factory=datetime.now)


class CritiqueResult(BaseModel):
    """Comprehensive feedback from a critic's evaluation.

    Critics analyze text and provide structured feedback about quality,
    issues, and suggestions for improvement. This model captures all
    aspects of their evaluation including confidence and metadata.

    Attributes:
        critic: Name of the critic that provided this feedback
        feedback: Main feedback text explaining the evaluation
        suggestions: List of specific, actionable improvement suggestions
        needs_improvement: Whether the critic thinks text needs improvement
        confidence: Critic's confidence in its assessment (0.0-1.0)
        metadata: Additional critic-specific data (e.g., scores, analysis)
        timestamp: When this critique was generated
        model_used: The LLM model used by this critic
        temperature_used: The temperature setting used
        prompt_sent: The exact prompt sent to the LLM (for debugging)
        tokens_used: Number of tokens consumed
        processing_time: Time in seconds to generate critique
        tools_used: List of any tools called during critique
    """

    critic: str
    feedback: str
    suggestions: List[str]
    needs_improvement: bool = True
    confidence: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.now)

    # Traceability fields
    model_used: Optional[str] = None
    temperature_used: Optional[float] = None
    prompt_sent: Optional[str] = None  # The actual prompt sent to the LLM
    tokens_used: int = 0
    processing_time: float = 0.0

    # Tool usage during this critique
    tools_used: List[ToolUsage] = Field(default_factory=list)


class SifakaResult(BaseModel):
    """Complete result container with full audit trail of the improvement process.

    This is the main object returned by improve() functions. It contains
    the final improved text along with a complete history of all generations,
    critiques, and validations. Uses memory-bounded collections (deque) to
    prevent unbounded growth during long sessions.

    Attributes:
        final_text: The final improved text after all iterations
        original_text: The original input text before any improvements
        iteration: Number of improvement iterations completed
        generations: History of all text generations (max 10 most recent)
        validations: History of all validation results (max 20 most recent)
        critiques: History of all critic feedback (max 20 most recent)
        tools_used: History of all tool usage (max 50 most recent)
        id: Unique identifier for this result
        created_at: When this improvement session started
        updated_at: When this result was last modified
        processing_time: Total time in seconds for all processing
        config_used: Configuration dictionary used for this session

    Example:
        >>> result = await improve("Short text")
        >>> print(f"Original: {result.original_text}")
        >>> print(f"Improved: {result.final_text}")
        >>> print(f"Iterations: {result.iteration}")
        >>> for critique in result.critiques:
        ...     print(f"{critique.critic}: {critique.feedback}")
    """

    # Core results
    final_text: str
    original_text: str
    iteration: int = 0

    # Audit trail (automatically memory-bounded using deque)
    generations: Deque[Generation] = Field(default_factory=lambda: deque(maxlen=10))
    validations: Deque[ValidationResult] = Field(
        default_factory=lambda: deque(maxlen=20)
    )
    critiques: Deque[CritiqueResult] = Field(default_factory=lambda: deque(maxlen=20))
    tools_used: Deque[ToolUsage] = Field(default_factory=lambda: deque(maxlen=50))

    # Metadata
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    processing_time: float = 0.0

    # Configuration used (for traceability)
    config_used: Optional[Dict[str, Any]] = None

    def add_generation(
        self,
        text: str,
        model: str,
        prompt: Optional[str] = None,
        tokens: int = 0,
        processing_time: float = 0.0,
    ) -> None:
        """Add a new text generation to the history.

        Automatically manages memory bounds - old generations are removed
        when the deque reaches its maximum size.

        Args:
            text: The generated text
            model: Model name that generated this text
            prompt: The prompt used (optional, for debugging)
            tokens: Number of tokens consumed
            processing_time: Time taken to generate
        """
        gen = Generation(
            text=text,
            model=model,
            iteration=self.iteration,
            prompt=prompt,
            tokens_used=tokens,
            processing_time=processing_time,
        )
        self.generations.append(gen)
        self.updated_at = datetime.now()

    def add_validation(
        self,
        validator: str,
        passed: bool,
        score: Optional[float] = None,
        details: str = "",
    ) -> None:
        """Add a validation result to the history.

        Args:
            validator: Name of the validator
            passed: Whether validation passed
            score: Optional score (0.0-1.0)
            details: Human-readable explanation
        """
        result = ValidationResult(
            validator=validator, passed=passed, score=score, details=details
        )
        self.validations.append(result)
        self.updated_at = datetime.now()

    def add_critique(
        self,
        critic: str,
        feedback: str,
        suggestions: List[str],
        needs_improvement: bool = True,
        confidence: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add a critique result to the history.

        Args:
            critic: Name of the critic
            feedback: Main feedback text
            suggestions: List of improvement suggestions
            needs_improvement: Whether improvement is needed
            confidence: Confidence score (0.0-1.0)
            metadata: Additional critic-specific data
        """
        result = CritiqueResult(
            critic=critic,
            feedback=feedback,
            suggestions=suggestions,
            needs_improvement=needs_improvement,
            confidence=confidence,
            metadata=metadata or {},
        )
        self.critiques.append(result)
        self.updated_at = datetime.now()

    def increment_iteration(self) -> None:
        """Increment the iteration counter.

        Called at the start of each improvement iteration to track progress.
        """
        self.iteration += 1
        self.updated_at = datetime.now()

    def set_final_text(self, text: str) -> None:
        """Set the final improved text.

        Args:
            text: The final text after all improvements
        """
        self.final_text = text
        self.updated_at = datetime.now()

    @property
    def current_text(self) -> str:
        """Get the current text being worked on.

        Returns the most recent generation if any exist, otherwise
        returns the original text. This represents the current state
        of the text improvement process.

        Returns:
            The most recent text version
        """
        return (
            list(self.generations)[-1].text if self.generations else self.original_text
        )

    @property
    def all_passed(self) -> bool:
        """Check if all validators are currently passing.

        Examines the most recent validation result for each unique
        validator to determine overall validation status.

        Returns:
            True if all validators passed on their last run, False otherwise
        """
        if not self.validations:
            return False
        # Check last validation for each validator
        recent_validations = {}
        for val in reversed(self.validations):
            if val.validator not in recent_validations:
                recent_validations[val.validator] = val.passed
        return all(recent_validations.values())

    @property
    def needs_improvement(self) -> bool:
        """Check if any critics think the text needs improvement.

        Examines the most recent critique from each unique critic
        to determine if further improvement is recommended.

        Returns:
            True if any critic suggests improvement, False if all are satisfied
        """
        if not self.critiques:
            return True
        # Check most recent critique from each critic
        recent_critiques = {}
        for crit in reversed(self.critiques):
            if crit.critic not in recent_critiques:
                recent_critiques[crit.critic] = crit.needs_improvement
        return any(recent_critiques.values())

    def _get_most_recent_critique(self) -> Optional[CritiqueResult]:
        """Get the most recent critique result.

        Returns:
            The most recent CritiqueResult, or None if no critiques exist
        """
        if not self.critiques:
            return None
        critiques_list = list(self.critiques)
        return critiques_list[-1] if critiques_list else None

    def get_quality_progression(self) -> Dict[str, List[int]]:
        """Get basic progression metrics across all versions."""
        # Simple length and word count progression
        text_lengths = [len(self.original_text)]
        word_counts = [len(self.original_text.split())]

        for gen in self.generations:
            text_lengths.append(len(gen.text))
            word_counts.append(len(gen.text.split()))

        return {
            "text_length_progression": text_lengths,
            "word_count_progression": word_counts,
        }


# Config class moved to config.py
