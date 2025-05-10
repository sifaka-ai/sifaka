"""
Core base classes for Sifaka components.

This module provides shared base classes and utilities used by both critics and rules,
implementing common patterns for state management, configuration, and error handling.
"""

from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import (
    Any,
    Dict,
    Generic,
    List,
    Optional,
    Type,
    TypeVar,
    Protocol,
    runtime_checkable,
    Union,
)
import re
import time
from datetime import datetime

from pydantic import BaseModel, Field, ConfigDict, PrivateAttr

from sifaka.utils.state import StateManager
from sifaka.utils.logging import get_logger

logger = get_logger(__name__)

# Type variables
T = TypeVar("T")  # Input type
R = TypeVar("R")  # Result type
C = TypeVar("C", bound="BaseComponent")  # Component type


class ComponentResultEnum(str, Enum):
    """Enumeration of possible component results."""

    SUCCESS = auto()
    NEEDS_IMPROVEMENT = auto()
    FAILURE = auto()


class ValidationPattern:
    """Common validation patterns."""

    EMAIL = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    URL = r"^https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)$"
    PHONE = r"^\+?1?\d{9,15}$"
    DATE = r"^\d{4}-\d{2}-\d{2}$"
    TIME = r"^\d{2}:\d{2}(:\d{2})?$"
    IPV4 = r"^(\d{1,3}\.){3}\d{1,3}$"
    IPV6 = r"^([0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}$"


class BaseConfig(BaseModel):
    """Base configuration for all components."""

    name: str = Field(description="Component name", min_length=1)
    description: str = Field(description="Component description", min_length=1)
    min_confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    cache_size: int = Field(default=100, ge=0)
    priority: int = Field(default=1, ge=1)
    cost: float = Field(default=1.0, ge=0.0)
    params: Dict[str, Any] = Field(default_factory=dict)
    track_performance: bool = Field(default=True)
    track_errors: bool = Field(default=True)

    model_config = ConfigDict(
        arbitrary_types_allowed=True, validate_assignment=True, extra="forbid"
    )


class BaseResult(BaseModel, Generic[T]):
    """Base result for all components."""

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
        return self.model_copy(update={"metadata": {**self.metadata, **kwargs}})

    def with_issues(self, issues: List[str]) -> "BaseResult":
        """Create a new result with updated issues."""
        return self.model_copy(update={"issues": issues})

    def with_suggestions(self, suggestions: List[str]) -> "BaseResult":
        """Create a new result with updated suggestions."""
        return self.model_copy(update={"suggestions": suggestions})

    def with_score(self, score: float) -> "BaseResult":
        """Create a new result with updated score."""
        return self.model_copy(update={"score": score})

    def normalize_score(self, min_score: float = 0.0, max_score: float = 1.0) -> "BaseResult":
        """Normalize the score to a given range."""
        if max_score <= min_score:
            raise ValueError("max_score must be greater than min_score")

        normalized = (self.score - min_score) / (max_score - min_score)
        return self.with_score(max(0.0, min(1.0, normalized)))

    def combine(self, other: "BaseResult") -> "BaseResult":
        """Combine this result with another result."""
        return BaseResult(
            passed=self.passed and other.passed,
            message=f"{self.message} | {other.message}",
            metadata={**self.metadata, **other.metadata},
            score=(self.score + other.score) / 2,
            issues=[*self.issues, *other.issues],
            suggestions=[*self.suggestions, *other.suggestions],
            processing_time_ms=self.processing_time_ms + other.processing_time_ms,
            timestamp=max(self.timestamp, other.timestamp),
        )


@runtime_checkable
class Validatable(Protocol[T]):
    """Protocol for components that can validate inputs."""

    def validate(self, input: T) -> BaseResult:
        """Validate the input."""
        ...

    def can_validate(self, input: T) -> bool:
        """Check if this component can validate the input."""
        ...


class BaseComponent(ABC, Generic[T, R]):
    """Base class for all Sifaka components."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True, validate_assignment=True, extra="forbid"
    )

    name: str = Field(description="Component name", min_length=1)
    description: str = Field(description="Component description", min_length=1)
    config: BaseConfig

    # Add state manager as a private attribute
    _state_manager = None

    def __init__(
        self, name: str, description: str, config: Optional[BaseConfig] = None, **kwargs: Any
    ) -> None:
        """Initialize the component."""
        # Store name and description as instance variables
        self._name = name
        self._description = description
        # Initialize state first
        self._initialize_state()
        # Then set config (which might need state to be initialized)
        self._config = config or BaseConfig(name=name, description=description, **kwargs)

    def _initialize_state(self) -> None:
        """Initialize component state."""
        # Create state manager
        from sifaka.utils.state import StateManager

        self._state_manager = StateManager()

        # Initialize state
        self._state_manager.update("initialized", False)
        self._state_manager.update("cache", {})
        self._state_manager.set_metadata("component_type", self.__class__.__name__)
        self._state_manager.set_metadata("validation_count", 0)
        self._state_manager.set_metadata("success_count", 0)
        self._state_manager.set_metadata("failure_count", 0)
        self._state_manager.set_metadata("improvement_count", 0)
        self._state_manager.set_metadata("total_processing_time_ms", 0.0)
        self._state_manager.set_metadata("error_count", 0)
        self._state_manager.set_metadata("last_error", None)
        self._state_manager.set_metadata("last_error_time", None)

    @property
    def name(self) -> str:
        """Get component name."""
        return self._name

    @property
    def description(self) -> str:
        """Get component description."""
        return self._description

    @property
    def config(self) -> BaseConfig:
        """Get component configuration."""
        return self._config

    @config.setter
    def config(self, value: BaseConfig) -> None:
        """Set component configuration."""
        self._config = value

    @property
    def min_confidence(self) -> float:
        """Get minimum confidence threshold."""
        return self.config.min_confidence

    def validate_input(self, input: Any) -> bool:
        """Validate input type and format."""
        if not input:
            return False
        return isinstance(input, str)  # Default to string validation

    def handle_empty_input(self, input: str) -> Optional[BaseResult]:
        """Handle empty input validation."""
        if not input:
            return BaseResult(
                passed=False,
                message="Empty input",
                metadata={"error_type": "empty_input"},
                score=0.0,
                issues=["Input is empty"],
                suggestions=["Provide non-empty input"],
            )
        return None

    def validate_text_length(
        self, text: str, min_length: int = 0, max_length: Optional[int] = None
    ) -> bool:
        """Validate text length."""
        if not isinstance(text, str):
            return False
        if len(text) < min_length:
            return False
        if max_length is not None and len(text) > max_length:
            return False
        return True

    def validate_text_pattern(self, text: str, pattern: str) -> bool:
        """Validate text against a pattern."""
        if not isinstance(text, str):
            return False
        return bool(re.match(pattern, text))

    def validate_text_contains(self, text: str, required_chars: List[str]) -> bool:
        """Validate text contains required characters."""
        if not isinstance(text, str):
            return False
        return all(char in text for char in required_chars)

    def get_statistics(self) -> Dict[str, Any]:
        """Get component statistics."""
        total_count = self._state_manager.get_metadata("validation_count", 0)
        success_count = self._state_manager.get_metadata("success_count", 0)
        failure_count = self._state_manager.get_metadata("failure_count", 0)
        total_time = self._state_manager.get_metadata("total_processing_time_ms", 0.0)
        error_count = self._state_manager.get_metadata("error_count", 0)

        return {
            "name": self.name,
            "validation_count": total_count,
            "success_count": success_count,
            "failure_count": failure_count,
            "improvement_count": self._state_manager.get_metadata("improvement_count", 0),
            "success_rate": success_count / total_count if total_count > 0 else 0.0,
            "error_rate": error_count / total_count if total_count > 0 else 0.0,
            "average_processing_time_ms": total_time / total_count if total_count > 0 else 0.0,
            "cache_size": len(self._state_manager.get("cache", {})),
            "initialized": self._state_manager.get("initialized", False),
            "last_error": self._state_manager.get_metadata("last_error"),
            "last_error_time": self._state_manager.get_metadata("last_error_time"),
        }

    def clear_cache(self) -> None:
        """Clear component cache."""
        self._state_manager.update("cache", {})

    def reset_statistics(self) -> None:
        """Reset component statistics."""
        self._state_manager.set_metadata("validation_count", 0)
        self._state_manager.set_metadata("success_count", 0)
        self._state_manager.set_metadata("failure_count", 0)
        self._state_manager.set_metadata("improvement_count", 0)
        self._state_manager.set_metadata("total_processing_time_ms", 0.0)
        self._state_manager.set_metadata("error_count", 0)
        self._state_manager.set_metadata("last_error", None)
        self._state_manager.set_metadata("last_error_time", None)

    def update_statistics(self, result: BaseResult) -> None:
        """Update component statistics based on result."""
        validation_count = self._state_manager.get_metadata("validation_count", 0)
        self._state_manager.set_metadata("validation_count", validation_count + 1)

        if result.passed:
            success_count = self._state_manager.get_metadata("success_count", 0)
            self._state_manager.set_metadata("success_count", success_count + 1)
        else:
            failure_count = self._state_manager.get_metadata("failure_count", 0)
            self._state_manager.set_metadata("failure_count", failure_count + 1)

        if result.suggestions:
            improvement_count = self._state_manager.get_metadata("improvement_count", 0)
            self._state_manager.set_metadata("improvement_count", improvement_count + 1)

        if self.config.track_performance:
            total_time = self._state_manager.get_metadata("total_processing_time_ms", 0.0)
            self._state_manager.set_metadata(
                "total_processing_time_ms", total_time + result.processing_time_ms
            )

    def record_error(self, error: Exception) -> None:
        """Record an error occurrence."""
        if self.config.track_errors:
            error_count = self._state_manager.get_metadata("error_count", 0)
            self._state_manager.set_metadata("error_count", error_count + 1)
            self._state_manager.set_metadata("last_error", str(error))
            self._state_manager.set_metadata("last_error_time", datetime.now())

    @abstractmethod
    def process(self, input: T) -> R:
        """Process the input and return a result."""
        ...

    def warm_up(self) -> None:
        """Prepare the component for use."""
        if not self._state_manager.get("initialized", False):
            self._state_manager.update("initialized", True)

    def cleanup(self) -> None:
        """Clean up component resources."""
        self.clear_cache()
        self._state_manager.update("initialized", False)

    @classmethod
    def create(cls: Type[C], name: str, description: str, **kwargs: Any) -> C:
        """Create a new component instance."""
        return cls(name=name, description=description, config=BaseConfig(**kwargs))
