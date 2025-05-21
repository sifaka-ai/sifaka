"""
Thought container for Sifaka.

This module defines the Thought class, which is the central state container for Sifaka.
It passes information between models, validators, and critics.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class ValidationResult(BaseModel):
    """Result of a validation check."""

    validator_name: str
    passed: bool
    score: Optional[float] = None
    details: Dict[str, Any] = {}
    message: str = ""


class CriticFeedback(BaseModel):
    """Feedback from a critic."""

    critic_name: str
    feedback: str
    suggestions: List[str] = []
    details: Dict[str, Any] = {}


class RetrievedContext(BaseModel):
    """Context retrieved from an external source."""

    source: str
    content: str
    metadata: Dict[str, Any] = {}
    relevance_score: Optional[float] = None


@dataclass
class Thought:
    """
    Central state container for Sifaka.

    Thoughts contain all information related to a generation, including:
    - The prompt and generated text
    - Retrieved context
    - Validation results
    - Critic feedback

    Thoughts can be persisted and retrieved for analysis and debugging.
    """

    # Input and output
    prompt: str
    text: str = ""

    # Context
    retrieved_context: List[RetrievedContext] = field(default_factory=list)

    # Validation
    validation_results: List[ValidationResult] = field(default_factory=list)
    validation_passed: bool = False

    # Critic feedback
    critic_feedback: List[CriticFeedback] = field(default_factory=list)

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Iteration number (0 for initial generation, incremented for each improvement)
    iteration: int = 0

    # History of previous versions of this thought
    history: List["Thought"] = field(default_factory=list)

    def add_retrieved_context(
        self,
        source: str,
        content: str,
        metadata: Dict[str, Any] = None,
        relevance_score: Optional[float] = None,
    ) -> None:
        """Add retrieved context to the thought."""
        self.retrieved_context.append(
            RetrievedContext(
                source=source,
                content=content,
                metadata=metadata or {},
                relevance_score=relevance_score,
            )
        )

    def add_validation_result(
        self,
        validator_name: str,
        passed: bool,
        score: Optional[float] = None,
        details: Dict[str, Any] = None,
        message: str = "",
    ) -> None:
        """Add a validation result to the thought."""
        self.validation_results.append(
            ValidationResult(
                validator_name=validator_name,
                passed=passed,
                score=score,
                details=details or {},
                message=message,
            )
        )

    def add_critic_feedback(
        self,
        critic_name: str,
        feedback: str,
        suggestions: List[str] = None,
        details: Dict[str, Any] = None,
    ) -> None:
        """Add critic feedback to the thought."""
        self.critic_feedback.append(
            CriticFeedback(
                critic_name=critic_name,
                feedback=feedback,
                suggestions=suggestions or [],
                details=details or {},
            )
        )

    def update_validation_status(self) -> bool:
        """Update the validation_passed flag based on validation results."""
        self.validation_passed = all(result.passed for result in self.validation_results)
        return self.validation_passed

    def create_new_version(self, text: str) -> "Thought":
        """Create a new version of this thought with updated text."""
        # Add current state to history
        history = self.history.copy()
        current_copy = Thought(
            prompt=self.prompt,
            text=self.text,
            retrieved_context=self.retrieved_context,
            validation_results=self.validation_results,
            validation_passed=self.validation_passed,
            critic_feedback=self.critic_feedback,
            metadata=self.metadata,
            iteration=self.iteration,  # Preserve the current iteration number
            history=[],  # Don't include history in the history to avoid recursion
        )
        history.append(current_copy)

        # Create new version with incremented iteration number
        return Thought(
            prompt=self.prompt,
            text=text,
            retrieved_context=self.retrieved_context.copy(),
            validation_results=[],  # Reset validation results
            validation_passed=False,
            critic_feedback=[],  # Reset critic feedback
            metadata=self.metadata.copy(),
            iteration=self.iteration + 1,  # Increment the iteration number
            history=history,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert the thought to a dictionary for serialization."""
        return {
            "prompt": self.prompt,
            "text": self.text,
            "retrieved_context": [context.model_dump() for context in self.retrieved_context],
            "validation_results": [result.model_dump() for result in self.validation_results],
            "validation_passed": self.validation_passed,
            "critic_feedback": [feedback.model_dump() for feedback in self.critic_feedback],
            "metadata": self.metadata,
            "iteration": self.iteration,  # Include the iteration number
            "history": [thought.to_dict() for thought in self.history],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Thought":
        """Create a thought from a dictionary."""
        history_data = data.pop("history", [])

        # Extract iteration number or default to 0
        iteration = data.pop("iteration", 0)

        thought = cls(**data, iteration=iteration)

        # Reconstruct nested objects
        thought.retrieved_context = [
            RetrievedContext(**context) for context in data.get("retrieved_context", [])
        ]
        thought.validation_results = [
            ValidationResult(**result) for result in data.get("validation_results", [])
        ]
        thought.critic_feedback = [
            CriticFeedback(**feedback) for feedback in data.get("critic_feedback", [])
        ]

        # Reconstruct history
        thought.history = [cls.from_dict(history_item) for history_item in history_data]

        return thought
