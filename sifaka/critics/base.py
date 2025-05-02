"""
Base module for critics that provide feedback and validation on prompts.

This module defines the core interfaces and base implementations for critics
that analyze and improve text outputs based on rule violations.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, Final, List, Protocol, TypeVar, runtime_checkable

from typing_extensions import TypeGuard


class CriticResult(str, Enum):
    """Enumeration of possible critic results."""

    SUCCESS = auto()
    NEEDS_IMPROVEMENT = auto()
    FAILURE = auto()


@dataclass(frozen=True)
class CriticConfig:
    """Immutable configuration for critics."""

    name: str
    description: str
    min_confidence: float = 0.7
    max_attempts: int = 3
    cache_size: int = 100
    priority: int = 1
    cost: float = 1.0

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if not self.name or not self.name.strip():
            raise ValueError("name cannot be empty or whitespace")
        if not self.description or not self.description.strip():
            raise ValueError("description cannot be empty or whitespace")
        if not 0 <= self.min_confidence <= 1:
            raise ValueError("min_confidence must be between 0 and 1")
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        if self.cache_size < 0:
            raise ValueError("cache_size must be non-negative")
        if self.priority < 0:
            raise ValueError("priority must be non-negative")
        if self.cost < 0:
            raise ValueError("cost must be non-negative")


@dataclass(frozen=True)
class CriticMetadata:
    """Immutable metadata for critic results."""

    score: float
    feedback: str
    issues: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    attempt_number: int = 1
    processing_time_ms: float = 0.0

    def __post_init__(self) -> None:
        """Validate metadata values."""
        if not 0 <= self.score <= 1:
            raise ValueError("score must be between 0 and 1")
        if self.attempt_number < 1:
            raise ValueError("attempt_number must be positive")
        if self.processing_time_ms < 0:
            raise ValueError("processing_time_ms must be non-negative")


@dataclass(frozen=True)
class CriticOutput:
    """Immutable output from a critic."""

    result: CriticResult
    improved_text: str
    metadata: CriticMetadata


@runtime_checkable
class TextValidator(Protocol):
    """Protocol for text validation."""

    @property
    def config(self) -> CriticConfig:
        """Get validator configuration."""
        ...

    def validate(self, text: str) -> bool:
        """Validate text against quality standards."""
        ...


@runtime_checkable
class TextImprover(Protocol):
    """Protocol for text improvement."""

    @property
    def config(self) -> CriticConfig:
        """Get improver configuration."""
        ...

    def improve(self, text: str, violations: List[Dict[str, Any]]) -> str:
        """Improve text based on violations."""
        ...


@runtime_checkable
class TextCritic(Protocol):
    """Protocol for text critiquing."""

    @property
    def config(self) -> CriticConfig:
        """Get critic configuration."""
        ...

    def critique(self, text: str) -> CriticMetadata:
        """Critique text and provide feedback."""
        ...


class BaseCritic(ABC):
    """Abstract base class for critics implementing all protocols."""

    def __init__(self, config: CriticConfig) -> None:
        """Initialize the critic with configuration."""
        self._config = config
        self._validate_config()

    @property
    def config(self) -> CriticConfig:
        """Get critic configuration."""
        return self._config

    def _validate_config(self) -> None:
        """Validate critic configuration."""
        # Allow both dataclass and pydantic CriticConfig
        if not hasattr(self.config, "name") or not hasattr(self.config, "description"):
            raise TypeError("config must have name and description attributes")

    def is_valid_text(self, text: Any) -> TypeGuard[str]:
        """Type guard to ensure text is a valid string."""
        return isinstance(text, str) and bool(text.strip())

    @abstractmethod
    def validate(self, text: str) -> bool:
        """Validate text against quality standards."""

    @abstractmethod
    def improve(self, text: str, violations: List[Dict[str, Any]]) -> str:
        """Improve text based on violations."""

    @abstractmethod
    def critique(self, text: str) -> CriticMetadata:
        """Critique text and provide feedback."""

    def process(self, text: str, violations: List[Dict[str, Any]]) -> CriticOutput:
        """Process text and return improved version with metadata."""
        if not self.is_valid_text(text):
            raise ValueError("text must be a non-empty string")

        metadata = self.critique(text)

        if metadata.score >= self.config.min_confidence:
            return CriticOutput(result=CriticResult.SUCCESS, improved_text=text, metadata=metadata)

        improved_text = self.improve(text, violations)
        if not self.is_valid_text(improved_text):
            return CriticOutput(
                result=CriticResult.FAILURE,
                improved_text=text,  # Return original if improvement failed
                metadata=metadata,
            )

        return CriticOutput(
            result=CriticResult.NEEDS_IMPROVEMENT, improved_text=improved_text, metadata=metadata
        )


# Common validation patterns
DEFAULT_MIN_CONFIDENCE: Final[float] = 0.7
DEFAULT_MAX_ATTEMPTS: Final[int] = 3
DEFAULT_CACHE_SIZE: Final[int] = 100

T = TypeVar("T", bound=BaseCritic)


def create_critic(critic_class: type[T], name: str, description: str, **kwargs: Any) -> T:
    """Factory function to create a critic instance with configuration."""
    config = CriticConfig(
        name=name,
        description=description,
        min_confidence=kwargs.pop("min_confidence", DEFAULT_MIN_CONFIDENCE),
        max_attempts=kwargs.pop("max_attempts", DEFAULT_MAX_ATTEMPTS),
        cache_size=kwargs.pop("cache_size", DEFAULT_CACHE_SIZE),
        priority=kwargs.pop("priority", 1),
        cost=kwargs.pop("cost", 1.0),
    )
    return critic_class(config=config, **kwargs)


class Critic(BaseCritic):
    """Default implementation of a text critic."""

    def validate(self, text: str) -> bool:
        """Validate text against quality standards."""
        if not self.is_valid_text(text):
            return False
        return len(text.strip()) > 0

    def improve(self, text: str, violations: List[Dict[str, Any]]) -> str:
        """Improve text based on violations."""
        if not violations:
            return text
        # Apply basic improvements based on violations
        improved = text
        for violation in violations:
            if "fix" in violation:
                improved = violation["fix"](improved)
        return improved

    def critique(self, text: str) -> CriticMetadata:
        """Critique text and provide feedback."""
        if not self.is_valid_text(text):
            return CriticMetadata(
                score=0.0,
                feedback="Invalid or empty text",
                issues=["Text must be a non-empty string"],
                suggestions=["Provide non-empty text input"],
            )

        # Basic critique based on text length and content
        score = min(1.0, len(text.strip()) / 100)  # Simple scoring based on length
        feedback = "Text meets basic requirements" if score >= 0.5 else "Text needs improvement"
        issues = [] if score >= 0.5 else ["Text may be too short"]
        suggestions = [] if score >= 0.5 else ["Consider adding more content"]

        return CriticMetadata(
            score=score,
            feedback=feedback,
            issues=issues,
            suggestions=suggestions,
        )


__all__ = [
    "Critic",
    "BaseCritic",
    "CriticConfig",
    "CriticMetadata",
    "CriticOutput",
    "CriticResult",
    "TextValidator",
    "TextImprover",
    "TextCritic",
    "create_critic",
]
