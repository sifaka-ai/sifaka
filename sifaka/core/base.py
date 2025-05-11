"""
Core base classes for Sifaka components.

This module provides shared base classes and utilities used by both critics and rules,
implementing common patterns for state management, configuration, and error handling.
"""

import time
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum, auto
import re
from typing import (
    Any,
    Dict,
    Generic,
    List,
    Optional,
    Protocol,
    runtime_checkable,
    Type,
    TypeVar,
    Union,
)

from pydantic import BaseModel, Field, ConfigDict, PrivateAttr

from sifaka.utils.common import update_statistics, record_error
from sifaka.utils.errors import InitializationError
from sifaka.utils.logging import get_logger
from sifaka.utils.patterns import ValidationPattern
from sifaka.utils.state import StateManager

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


# ValidationPattern is imported at the top of the file


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
        from sifaka.utils.text import is_empty_text, handle_empty_text

        # Use the standardized function with passed=False for core components
        # This maintains the current behavior where empty input fails validation
        if isinstance(input, str) and is_empty_text(input):
            result = handle_empty_text(
                text=input,
                passed=False,
                message="Empty input",
                metadata={"error_type": "empty_input"},
                component_type="component",
            )

            # Convert RuleResult to BaseResult if needed
            if result and not isinstance(result, BaseResult):
                return BaseResult(
                    passed=result.passed,
                    message=result.message,
                    metadata=result.metadata,
                    score=result.score,
                    issues=result.issues or ["Input is empty"],
                    suggestions=result.suggestions or ["Provide non-empty input"],
                )
            return result

        # Handle non-string empty inputs
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
        # Use the standardized utility function
        # Convert processing_time_ms to seconds for the utility function
        execution_time = result.processing_time_ms / 1000.0
        update_statistics(
            state_manager=self._state_manager, execution_time=execution_time, success=result.passed
        )

        # Update component-specific statistics
        validation_count = self._state_manager.get_metadata("validation_count", 0)
        self._state_manager.set_metadata("validation_count", validation_count + 1)

        if result.suggestions:
            improvement_count = self._state_manager.get_metadata("improvement_count", 0)
            self._state_manager.set_metadata("improvement_count", improvement_count + 1)

    def record_error(self, error: Exception) -> None:
        """Record an error occurrence."""
        if self.config.track_errors:
            # Use the standardized utility function
            record_error(self._state_manager, error)

    @abstractmethod
    def process(self, input: T) -> R:
        """
        Process the input and return a result.

        This method processes the input and returns a result. Subclasses must
        implement this method to provide component-specific processing logic.

        Args:
            input: The input to process

        Returns:
            The processing result

        Raises:
            ValueError: If input is invalid
            RuntimeError: If processing fails
        """
        # Ensure component is initialized
        if not self._state_manager.get("initialized", False):
            self.warm_up()

        # Process input
        start_time = time.time()

        # Define the operation
        def operation():
            # Actual processing logic (to be implemented by subclasses)
            result = self._process_input(input)
            return result

        # Use standardized error handling
        from sifaka.utils.errors import safely_execute_component_operation

        result = safely_execute_component_operation(
            operation=operation,
            component_name=self.name,
            component_type=self.__class__.__name__,
            additional_metadata={"input_type": type(input).__name__},
        )

        # Update statistics
        processing_time = time.time() - start_time
        self.update_statistics(result, processing_time_ms=processing_time * 1000)

        return result

    def _process_input(self, input: T) -> R:
        """
        Process the input and return a result.

        This method is called by the process method to perform the actual
        processing logic. Subclasses should override this method instead
        of the process method to ensure consistent error handling.

        Args:
            input: The input to process

        Returns:
            The processing result
        """
        raise NotImplementedError("Subclasses must implement _process_input")

    def warm_up(self) -> None:
        """
        Prepare the component for use.

        This method prepares the component for use, performing any
        necessary warm-up operations. It's safe to call multiple times.

        Raises:
            InitializationError: If warm-up fails
        """
        try:
            # Check if already initialized
            if self._state_manager.get("initialized", False):
                logger.debug(f"Component {self.name} already initialized")
                return

            # Initialize resources (can be overridden by subclasses)
            self._initialize_resources()

            # Mark as initialized
            self._state_manager.update("initialized", True)
            self._state_manager.set_metadata("warm_up_time", time.time())

            logger.debug(f"Component {self.name} warmed up successfully")

        except Exception as e:
            self.record_error(e)
            logger.error(f"Failed to warm up component {self.name}: {str(e)}")
            raise InitializationError(f"Failed to warm up component {self.name}: {str(e)}") from e

    def _initialize_resources(self) -> None:
        """
        Initialize component resources.

        This method is called during warm-up to initialize any resources
        needed by the component. Subclasses should override this method
        to perform component-specific initialization.
        """
        pass

    def cleanup(self) -> None:
        """
        Clean up component resources.

        This method cleans up component resources, releasing any
        resources that were acquired during initialization or use.
        It's safe to call multiple times.
        """
        try:
            # Release resources (can be overridden by subclasses)
            self._release_resources()

            # Clear cache
            if hasattr(self, "clear_cache") and callable(getattr(self, "clear_cache")):
                self.clear_cache()

            # Reset initialization flag
            self._state_manager.update("initialized", False)

            logger.debug(f"Component {self.name} cleaned up successfully")

        except Exception as e:
            # Log but don't raise
            logger.error(f"Failed to clean up component {self.name}: {str(e)}")

    def _release_resources(self) -> None:
        """
        Release component resources.

        This method is called during cleanup to release any resources
        acquired during initialization or use. Subclasses should override
        this method to perform component-specific cleanup.
        """
        pass

    @classmethod
    def create(cls: Type[C], name: str, description: str, **kwargs: Any) -> C:
        """Create a new component instance."""
        return cls(name=name, description=description, config=BaseConfig(**kwargs))
