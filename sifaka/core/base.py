"""
Core base classes for Sifaka components.

This module provides shared base classes and utilities used by both critics and rules,
implementing common patterns for state management, configuration, and error handling.
"""

from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import Any, Dict, Generic, List, Optional, Type, TypeVar, Protocol, runtime_checkable
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


class BaseConfig(BaseModel):
    """Base configuration for all components."""

    name: str = Field(description="Component name", min_length=1)
    description: str = Field(description="Component description", min_length=1)
    min_confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    cache_size: int = Field(default=100, ge=0)
    priority: int = Field(default=1, ge=1)
    cost: float = Field(default=1.0, ge=0.0)
    params: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(
        arbitrary_types_allowed=True, validate_assignment=True, extra="forbid"
    )


class BaseResult(BaseModel):
    """Base result for all components."""

    passed: bool
    message: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(
        arbitrary_types_allowed=True, validate_assignment=True, extra="forbid"
    )

    def with_metadata(self, **kwargs: Any) -> "BaseResult":
        """Create a new result with additional metadata."""
        return self.model_copy(update={"metadata": {**self.metadata, **kwargs}})


@runtime_checkable
class Validatable(Protocol[T]):
    """Protocol for components that can validate inputs."""

    def validate(self, input: T) -> bool:
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
    _state = PrivateAttr(default_factory=StateManager)

    def __init__(
        self, name: str, description: str, config: Optional[BaseConfig] = None, **kwargs: Any
    ) -> None:
        """Initialize the component."""
        self.name = name
        self.description = description
        self.config = config or BaseConfig(name=name, description=description, **kwargs)
        self._initialize_state()

    def _initialize_state(self) -> None:
        """Initialize component state."""
        self._state.update("initialized", False)
        self._state.update("cache", {})
        self._state.set_metadata("component_type", self.__class__.__name__)
        self._state.set_metadata("validation_count", 0)

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
                passed=False, message="Empty input", metadata={"error_type": "empty_input"}
            )
        return None

    def get_statistics(self) -> Dict[str, Any]:
        """Get component statistics."""
        return {
            "name": self.name,
            "validation_count": self._state.get_metadata("validation_count", 0),
            "cache_size": len(self._state.get("cache", {})),
            "initialized": self._state.get("initialized", False),
        }

    def clear_cache(self) -> None:
        """Clear component cache."""
        self._state.update("cache", {})

    @abstractmethod
    def process(self, input: T) -> R:
        """Process the input and return a result."""
        ...

    def warm_up(self) -> None:
        """Prepare the component for use."""
        if not self._state.get("initialized", False):
            self._state.update("initialized", True)

    def cleanup(self) -> None:
        """Clean up component resources."""
        self.clear_cache()
        self._state.update("initialized", False)
