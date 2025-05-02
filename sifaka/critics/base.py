"""
Base module for critics that provide feedback and validation on prompts.

This module defines the core interfaces and base implementations for critics
that analyze and improve text outputs based on rule violations.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Dict,
    Final,
    Generic,
    List,
    Optional,
    Protocol,
    Type,
    TypeVar,
    Union,
    cast,
    overload,
    runtime_checkable
)

from typing_extensions import TypeGuard


# Input and output type variables
T = TypeVar('T')  # Input type (usually str)
R = TypeVar('R')  # Result type
C = TypeVar('C', bound='BaseCritic')  # Critic type


class CriticResult(str, Enum):
    """Enumeration of possible critic results."""

    SUCCESS = auto()
    NEEDS_IMPROVEMENT = auto()
    FAILURE = auto()


@dataclass(frozen=True)
class CriticConfig(Generic[T]):
    """Immutable configuration for critics."""

    name: str
    description: str
    min_confidence: float = 0.7
    max_attempts: int = 3
    cache_size: int = 100
    priority: int = 1
    cost: float = 1.0
    params: Dict[str, Any] = field(default_factory=dict)

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

    def with_params(self, **kwargs: Any) -> "CriticConfig[T]":
        """Create a new config with updated parameters."""
        new_params = {**self.params, **kwargs}
        return CriticConfig(
            name=self.name,
            description=self.description,
            min_confidence=self.min_confidence,
            max_attempts=self.max_attempts,
            cache_size=self.cache_size,
            priority=self.priority,
            cost=self.cost,
            params=new_params,
        )


@dataclass(frozen=True)
class CriticMetadata(Generic[R]):
    """Immutable metadata for critic results."""

    score: float
    feedback: str
    issues: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    attempt_number: int = 1
    processing_time_ms: float = 0.0
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate metadata values."""
        if not 0 <= self.score <= 1:
            raise ValueError("score must be between 0 and 1")
        if self.attempt_number < 1:
            raise ValueError("attempt_number must be positive")
        if self.processing_time_ms < 0:
            raise ValueError("processing_time_ms must be non-negative")

    def with_extra(self, **kwargs: Any) -> "CriticMetadata[R]":
        """Create a new metadata with additional extra data."""
        new_extra = {**self.extra, **kwargs}
        return CriticMetadata(
            score=self.score,
            feedback=self.feedback,
            issues=self.issues,
            suggestions=self.suggestions,
            attempt_number=self.attempt_number,
            processing_time_ms=self.processing_time_ms,
            extra=new_extra,
        )


@dataclass(frozen=True)
class CriticOutput(Generic[T, R]):
    """Immutable output from a critic."""

    result: CriticResult
    improved_text: T
    metadata: CriticMetadata[R]


@runtime_checkable
class TextValidator(Protocol[T]):
    """Protocol for text validation."""

    @property
    def config(self) -> CriticConfig[T]:
        """Get validator configuration."""
        ...

    def validate(self, text: T) -> bool:
        """Validate text against quality standards."""
        ...


@runtime_checkable
class TextImprover(Protocol[T, R]):
    """Protocol for text improvement."""

    @property
    def config(self) -> CriticConfig[T]:
        """Get improver configuration."""
        ...

    def improve(self, text: T, violations: List[Dict[str, Any]]) -> T:
        """Improve text based on violations."""
        ...


@runtime_checkable
class TextCritic(Protocol[T, R]):
    """Protocol for text critiquing."""

    @property
    def config(self) -> CriticConfig[T]:
        """Get critic configuration."""
        ...

    def critique(self, text: T) -> CriticMetadata[R]:
        """Critique text and provide feedback."""
        ...


class BaseCritic(ABC, Generic[T, R]):
    """
    Abstract base class for critics implementing all protocols.

    This class defines the interface for critics and provides common functionality.
    All critics should inherit from this class and implement the required methods.

    The BaseCritic follows a component-based architecture where functionality is
    delegated to specialized components:
    - PromptManager: Creates prompts for validation, critique, improvement, and reflection
    - ResponseParser: Parses responses from language models
    - MemoryManager: Manages memory for critics (optional)
    - CritiqueService: Provides methods for validation, critique, and improvement

    Type parameters:
        T: The input type (usually str)
        R: The result type
    """

    def __init__(self, config: CriticConfig[T]) -> None:
        """
        Initialize the critic with configuration.

        Args:
            config: The critic configuration
        """
        self._config = config
        self._validate_config()

    @property
    def config(self) -> CriticConfig[T]:
        """
        Get critic configuration.

        Returns:
            The critic configuration
        """
        return self._config

    def _validate_config(self) -> None:
        """
        Validate critic configuration.

        Raises:
            TypeError: If config is invalid
        """
        # Allow both dataclass and pydantic CriticConfig
        if not hasattr(self.config, "name") or not hasattr(self.config, "description"):
            raise TypeError("config must have name and description attributes")

    def is_valid_text(self, text: Any) -> TypeGuard[T]:
        """
        Type guard to ensure text is a valid string.

        Args:
            text: The text to check

        Returns:
            True if text is a valid string, False otherwise
        """
        return isinstance(text, str) and bool(text.strip())

    @abstractmethod
    def validate(self, text: T) -> bool:
        """
        Validate text against quality standards.

        Args:
            text: The text to validate

        Returns:
            True if the text meets quality standards, False otherwise

        Raises:
            ValueError: If text is empty
        """
        pass

    @abstractmethod
    def improve(self, text: T, violations: List[Dict[str, Any]]) -> T:
        """
        Improve text based on violations.

        Args:
            text: The text to improve
            violations: List of rule violations

        Returns:
            The improved text

        Raises:
            ValueError: If text is empty
        """
        pass

    @abstractmethod
    def critique(self, text: T) -> CriticMetadata[R]:
        """
        Critique text and provide feedback.

        Args:
            text: The text to critique

        Returns:
            CriticMetadata containing the critique details

        Raises:
            ValueError: If text is empty
        """
        pass

    @abstractmethod
    def improve_with_feedback(self, text: T, feedback: str) -> T:
        """
        Improve text based on feedback.

        Args:
            text: The text to improve
            feedback: Feedback to guide the improvement

        Returns:
            The improved text

        Raises:
            ValueError: If text is empty
        """
        pass

    def process(self, text: T, violations: List[Dict[str, Any]]) -> CriticOutput[T, R]:
        """
        Process text and return improved version with metadata.

        Args:
            text: The text to process
            violations: List of rule violations

        Returns:
            CriticOutput containing the result, improved text, and metadata

        Raises:
            ValueError: If text is empty
        """
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


@overload
def create_critic(
    critic_class: Type[C],
    name: str,
    description: str,
) -> C: ...


@overload
def create_critic(
    critic_class: Type[C],
    name: str,
    description: str,
    min_confidence: float,
    max_attempts: int,
    cache_size: int,
    priority: int,
    cost: float,
    **kwargs: Any,
) -> C: ...


def create_critic(
    critic_class: Type[C],
    name: str = "custom_critic",
    description: str = "Custom critic implementation",
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    cache_size: int = DEFAULT_CACHE_SIZE,
    priority: int = 1,
    cost: float = 1.0,
    config: Optional[CriticConfig[Any]] = None,
    **kwargs: Any,
) -> C:
    """
    Factory function to create a critic instance with configuration.

    This function creates a critic instance of the specified class with
    the given configuration parameters.

    Args:
        critic_class: The critic class to instantiate
        name: Name of the critic
        description: Description of the critic
        min_confidence: Minimum confidence threshold
        max_attempts: Maximum number of improvement attempts
        cache_size: Size of the cache
        priority: Priority of the critic
        cost: Cost of using the critic
        config: Optional critic configuration (overrides other parameters)
        **kwargs: Additional keyword arguments for the critic

    Returns:
        An instance of the specified critic class

    Raises:
        TypeError: If the created instance is not a BaseCritic
    """
    # Extract params from kwargs if present
    params: Dict[str, Any] = kwargs.pop("params", {})

    # Use provided config or create one from parameters
    if config is None:
        config = CriticConfig[Any](
            name=name,
            description=description,
            min_confidence=min_confidence,
            max_attempts=max_attempts,
            cache_size=cache_size,
            priority=priority,
            cost=cost,
            params=params,
        )

    # Create the critic instance with the config
    result = critic_class(config, **kwargs)
    assert isinstance(result, BaseCritic), f"Expected BaseCritic, got {type(result)}"
    return result


class Critic(BaseCritic[str, str]):
    """
    Default implementation of a text critic.

    This class provides a simple implementation of the BaseCritic interface
    for basic text validation, critique, and improvement.
    """

    def validate(self, text: str) -> bool:
        """
        Validate text against quality standards.

        Args:
            text: The text to validate

        Returns:
            True if the text meets quality standards, False otherwise
        """
        if not self.is_valid_text(text):
            return False
        return len(text.strip()) > 0

    def improve(self, text: str, violations: List[Dict[str, Any]]) -> str:
        """
        Improve text based on violations.

        Args:
            text: The text to improve
            violations: List of rule violations

        Returns:
            The improved text
        """
        if not violations:
            return text
        # Apply basic improvements based on violations
        improved = text
        for violation in violations:
            if "fix" in violation:
                improved = violation["fix"](improved)
        return improved

    def improve_with_feedback(self, text: str, feedback: str) -> str:
        """
        Improve text based on feedback.

        Args:
            text: The text to improve
            feedback: Feedback to guide the improvement

        Returns:
            The improved text
        """
        # Simple implementation that just appends the feedback
        return f"{text}\n\nImproved based on feedback: {feedback}"

    def critique(self, text: str) -> CriticMetadata[str]:
        """
        Critique text and provide feedback.

        Args:
            text: The text to critique

        Returns:
            CriticMetadata containing the critique details
        """
        if not self.is_valid_text(text):
            return CriticMetadata[str](
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

        return CriticMetadata[str](
            score=score,
            feedback=feedback,
            issues=issues,
            suggestions=suggestions,
        )


def create_basic_critic(
    name: str = "basic_critic",
    description: str = "Basic text critic",
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    **kwargs: Any,
) -> Critic:
    """
    Create a basic text critic with the given configuration.

    Args:
        name: Name of the critic
        description: Description of the critic
        min_confidence: Minimum confidence threshold
        max_attempts: Maximum number of improvement attempts
        **kwargs: Additional configuration parameters

    Returns:
        A configured Critic instance
    """
    return create_critic(
        critic_class=Critic,
        name=name,
        description=description,
        min_confidence=min_confidence,
        max_attempts=max_attempts,
        **kwargs,
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
    "create_basic_critic",
]
