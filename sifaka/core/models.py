"""Core models for Sifaka with memory bounds and type safety."""

import uuid
from datetime import datetime
from typing import List, Optional, Dict, Any, Deque
from collections import deque

from pydantic import BaseModel, Field

from .metrics import analyze_suggestion_implementation


class Generation(BaseModel):
    """A single text generation with metadata."""

    text: str
    model: str
    iteration: int
    prompt: Optional[str] = None  # The prompt used to generate this text
    timestamp: datetime = Field(default_factory=datetime.now)
    tokens_used: int = 0
    processing_time: float = 0.0

    # Improvement metrics
    improvement_metrics: Optional[Dict[str, Any]] = None
    suggestion_implementation: Optional[Dict[str, Any]] = None
    quality_indicators: Optional[Dict[str, float]] = None


class ValidationResult(BaseModel):
    """Result from a validation check."""

    validator: str
    passed: bool
    score: Optional[float] = None
    details: str = ""
    timestamp: datetime = Field(default_factory=datetime.now)


class CritiqueResult(BaseModel):
    """Result from a critic's feedback."""

    critic: str
    feedback: str
    suggestions: List[str]
    needs_improvement: bool = True
    confidence: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.now)


class SifakaResult(BaseModel):
    """Complete result with memory-bounded audit trail."""

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

    # Metadata
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    processing_time: float = 0.0

    def add_generation(
        self,
        text: str,
        model: str,
        prompt: Optional[str] = None,
        tokens: int = 0,
        processing_time: float = 0.0,
    ) -> None:
        """Add a generation."""
        # Analyze suggestion implementation if we have recent critiques
        suggestion_implementation = None
        recent_critique = self._get_most_recent_critique()
        if recent_critique and recent_critique.suggestions:
            suggestion_implementation = analyze_suggestion_implementation(
                recent_critique.suggestions, self.current_text, text
            )

        gen = Generation(
            text=text,
            model=model,
            iteration=self.iteration,
            prompt=prompt,
            tokens_used=tokens,
            processing_time=processing_time,
            suggestion_implementation=suggestion_implementation,
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
        """Add a validation result (automatically bounded by deque)."""
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
    ) -> None:
        """Add a critique result (automatically bounded by deque)."""
        result = CritiqueResult(
            critic=critic,
            feedback=feedback,
            suggestions=suggestions,
            needs_improvement=needs_improvement,
            confidence=confidence,
        )
        self.critiques.append(result)
        self.updated_at = datetime.now()

    def increment_iteration(self) -> None:
        """Move to next iteration."""
        self.iteration += 1
        self.updated_at = datetime.now()

    def set_final_text(self, text: str) -> None:
        """Set the final improved text."""
        self.final_text = text
        self.updated_at = datetime.now()

    @property
    def current_text(self) -> str:
        """Get the most recent generation or original text."""
        return (
            list(self.generations)[-1].text if self.generations else self.original_text
        )

    @property
    def all_passed(self) -> bool:
        """Check if all recent validations passed."""
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
        """Check if any recent critics suggest improvement."""
        if not self.critiques:
            return True
        # Check most recent critique from each critic
        recent_critiques = {}
        for crit in reversed(self.critiques):
            if crit.critic not in recent_critiques:
                recent_critiques[crit.critic] = crit.needs_improvement
        return any(recent_critiques.values())

    def _get_most_recent_critique(self) -> Optional[CritiqueResult]:
        """Get the most recent critique."""
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
