"""
Core Base Classes

This module provides the foundational base classes for all Sifaka components, implementing
common patterns for state management, configuration, validation, and error handling.

## Overview
The core base classes define the common interfaces, behaviors, and patterns that are
shared across all Sifaka components. They provide a consistent foundation for building
specialized components like rules, critics, classifiers, and chains.

## Components
- **BaseComponent**: Abstract base class for all Sifaka components
- **BaseConfig**: Configuration class for all components
- **Validatable**: Protocol for components that can validate inputs
- **ComponentResultEnum**: Enumeration of possible component results

## Usage Examples
```python
from sifaka.core.base import BaseComponent, BaseConfig
from sifaka.utils.result_types import BaseResult

# Create a custom component
class MyComponent(BaseComponent[str, BaseResult]):
    def _process_input(self, input: str) -> BaseResult:
        # Process the input
        return BaseResult(
            passed=True,
            message="Input processed successfully",
            score=0.9,
            suggestions=["Consider adding more details"]
        )

# Create a component instance
component = MyComponent(
    name="my_component",
    description="A custom component",
    config=BaseConfig(
        min_confidence=0.7,
        cache_size=100,
        priority=1
    )
)

# Process input
result = component.process("Hello, world!") if component else ""
print(f"Passed: {result.passed}, Score: {result.score}")
```

## Error Handling
The base classes provide standardized error handling through the `safely_execute_component_operation`
utility, which ensures consistent error handling across all components. Components also track
errors and statistics for monitoring and debugging.
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
from sifaka.utils.errors.base import InitializationError
from sifaka.utils.logging import get_logger
from sifaka.utils.state import StateManager
from sifaka.utils.result_types import BaseResult

logger = get_logger(__name__)
T = TypeVar("T", contravariant=True)  # Input type - contravariant
R = TypeVar("R", covariant=True)  # Result type - covariant
C = TypeVar("C", bound="BaseComponent")


class ComponentResultEnum(str, Enum):
    """Enumeration of possible component results."""

    SUCCESS = auto()
    NEEDS_IMPROVEMENT = auto()
    FAILURE = auto()


class BaseConfig(BaseModel):
    """
    Base configuration for all components.

    This class provides a standardized configuration model for all Sifaka components,
    defining common configuration parameters that are shared across different component types.

    ## Architecture
    BaseConfig uses Pydantic for validation and serialization, with:
    - Type validation for all fields
    - Default values for optional fields
    - Field descriptions for documentation
    - Immutable configuration (frozen=True)

    ## Lifecycle
    Configuration objects are typically created during component initialization and
    remain immutable throughout the component's lifecycle. Components can access
    configuration values through their config property.

    ## Examples
    ```python
    # Create a basic configuration
    config = BaseConfig(
        name="my_component",
        description="A custom component",
        min_confidence=0.8,
        cache_size=200,
        priority=2,
        cost=1.5,
        params={"threshold": 0.7}
    )

    # Access configuration values
    print(f"Name: {config.name}")
    print(f"Min confidence: {config.min_confidence}")
    print(f"Custom threshold: {config.params.get('threshold') if params else "")")
    ```

    Attributes:
        name: Component name
        description: Component description
        min_confidence: Minimum confidence threshold (0.0 to 1.0)
        cache_size: Size of the component's result cache
        priority: Priority level of the component (higher values = higher priority)
        cost: Computational cost of the component
        params: Dictionary of additional parameters
        track_performance: Whether to track performance statistics
        track_errors: Whether to track errors
    """

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
    """
    Base class for all Sifaka components.

    This abstract base class provides the foundation for all Sifaka components,
    implementing common functionality for state management, configuration,
    validation, error handling, and lifecycle management.

    ## Architecture
    BaseComponent uses a standardized architecture with:
    - State management through the _state_manager from utils/state.py
    - Configuration through BaseConfig
    - Result handling through BaseResult
    - Standardized error handling and statistics tracking

    ## Lifecycle
    Components follow a consistent lifecycle:
    1. **Initialization**: Component is created with name, description, and config
    2. **Warm-up**: Resources are initialized through warm_up() method
    3. **Processing**: Input is processed through process() method
    4. **Cleanup**: Resources are released through cleanup() method

    ## Error Handling
    Components use standardized error handling through:
    - Validation methods for input validation
    - Error recording through record_error()
    - Statistics tracking for monitoring

    ## Examples
    ```python
    # Create a custom component
    class MyComponent(BaseComponent[str, BaseResult]):
        def _process_input(self, input: str) -> BaseResult:
            # Process the input
            return BaseResult(
                passed=True,
                message="Input processed successfully",
                score=0.9
            )

    # Create a component instance
    component = MyComponent(
        name="my_component",
        description="A custom component"
    )

    # Warm up the component
    component.warm_up() if component else ""

    # Process input
    result = component.process("Hello, world!") if component else ""

    # Clean up resources
    component.cleanup() if component else ""
    ```
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True, validate_assignment=True, extra="forbid"
    )
    name: str = Field(description="Component name", min_length=1)
    description: str = Field(description="Component description", min_length=1)
    config: BaseConfig
    _state_manager: StateManager = PrivateAttr()

    def __init__(
        self, name: str, description: str, config: Optional[BaseConfig] = None, **kwargs: Any
    ) -> None:
        """Initialize the component."""
        self._name = name
        self._description = description
        object.__setattr__(self, "_state_manager", StateManager())
        self._initialize_state()
        self._config = config or BaseConfig(name=name, description=description, **kwargs)

    def _initialize_state(self) -> None:
        """Initialize component state."""
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

    # These methods are provided for backward compatibility
    # since the properties are now defined as class attributes
    def get_name(self) -> str:
        """Get component name."""
        return self._name

    def get_description(self) -> str:
        """Get component description."""
        return self._description

    def get_config(self) -> BaseConfig:
        """Get component configuration."""
        return self._config

    def set_config(self, value: BaseConfig) -> None:
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
        return isinstance(input, str)

    def handle_empty_input(self, input: str) -> Optional[BaseResult]:
        """Handle empty input validation."""
        from sifaka.utils.text import is_empty_text, handle_empty_text

        if isinstance(input, str) and is_empty_text(input):
            result = handle_empty_text(
                text=input,
                passed=False,
                message="Empty input",
                metadata={"error_type": "empty_input"},
                component_type="component",
            )
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
        self, text: str, min_length: int = 0, max_length: Optional[Optional[int]] = None
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

    def update_statistics(
        self, result: BaseResult, processing_time_ms: Optional[float] = None
    ) -> None:
        """Update component statistics based on result."""
        # If processing_time_ms is provided, use it, otherwise calculate from result
        if processing_time_ms is not None:
            execution_time = processing_time_ms / 1000.0
        else:
            execution_time = (
                result.processing_time_ms / 1000.0 if hasattr(result, "processing_time_ms") else 0.0
            )

        # Check if result is an ErrorResult
        from sifaka.utils.errors.results import ErrorResult

        if isinstance(result, ErrorResult):
            success = False
        else:
            success = result.passed if hasattr(result, "passed") else True

        update_statistics(
            state_manager=self._state_manager, execution_time=execution_time, success=success
        )

        validation_count = self._state_manager.get_metadata("validation_count", 0)
        self._state_manager.set_metadata("validation_count", validation_count + 1)

        if hasattr(result, "suggestions") and result.suggestions:
            improvement_count = self._state_manager.get_metadata("improvement_count", 0)
            self._state_manager.set_metadata("improvement_count", improvement_count + 1)

    def record_error(self, error: Exception) -> None:
        """Record an error occurrence."""
        track_errors = getattr(self.config, "track_errors", True)
        if track_errors:
            record_error(self._state_manager, error)

    @abstractmethod
    def process(self, input: T) -> R:
        """
        Process the input and return a result.

        This method processes the input and returns a result. It handles input validation,
        caching, error handling, and statistics tracking. The actual processing logic is
        delegated to the _process_input method, which subclasses must implement.

        ## Workflow
        1. Ensures the component is initialized (calls warm_up if needed)
        2. Validates the input
        3. Checks the cache for existing results
        4. Processes the input through _process_input
        5. Handles errors through standardized error handling
        6. Updates statistics
        7. Returns the result

        Args:
            input: The input to process (type depends on component implementation)

        Returns:
            The processing result (type depends on component implementation)

        Raises:
            ValueError: If input is invalid or empty
            RuntimeError: If processing fails
            InitializationError: If component initialization fails

        Example:
            ```python
            # Process input with a component
            result = (component.process("Hello, world!")

            # Check if processing was successful
            if result.passed:
                print(f"Processing succeeded with score {result.score}")
            else:
                print(f"Processing failed: {result.message}")
                print(f"Issues: {', '.join(result.issues))")
            ```
        """
        if not self._state_manager.get("initialized", False):
            self.warm_up()
        start_time = time.time()

        def operation() -> Any:
            result = self._process_input(input)
            return result

        from sifaka.utils.errors.safe_execution import safely_execute_component_operation
        from sifaka.utils.errors.base import ComponentError

        result = safely_execute_component_operation(
            operation=operation,
            component_name=self.name,
            component_type=self.__class__.__name__,
            error_class=ComponentError,
            additional_metadata={"input_type": type(input).__name__},
        )
        processing_time = time.time() - start_time

        # Handle the case where result might be an ErrorResult
        from sifaka.utils.errors.results import ErrorResult
        from typing import cast
        from sifaka.utils.result_types import BaseResult as UtilsBaseResult

        # Create a BaseResult for statistics if result is an ErrorResult
        if isinstance(result, ErrorResult):
            # Use a dummy BaseResult for statistics
            # Create a dummy result for statistics
            dummy_result: UtilsBaseResult = UtilsBaseResult(
                passed=False,
                message=result.error_message,
                processing_time_ms=processing_time * 1000,
                metadata={
                    "component_name": self.name,
                    "component_type": self.__class__.__name__,
                },
            )
            self.update_statistics(dummy_result)
            # Cast to R to satisfy the type checker
            # This is safe because the caller will check for ErrorResult
            return cast(R, result)
        else:
            # Normal case - update statistics with the actual result
            # We know result is not an ErrorResult, so it's safe to cast to BaseResult
            base_result: UtilsBaseResult = result
            self.update_statistics(base_result)
            return cast(R, result)  # Explicitly cast to satisfy mypy

    def _process_input(self, input: T) -> R:
        """
        Process the input and return a result.

        This method is called by the process method to perform the actual
        processing logic. Subclasses must override this method to implement
        component-specific processing logic. The process method handles common
        concerns like initialization, error handling, and statistics tracking.

        ## Implementation Guidelines
        When implementing this method in subclasses:
        1. Focus on the core processing logic specific to the component
        2. Assume input has already been validated
        3. Don't worry about error handling (handled by process method)
        4. Return a properly formatted result object

        Args:
            input: The input to process (type depends on component implementation)

        Returns:
            The processing result (type depends on component implementation)

        Example:
            ```python
            def _process_input(self, input: str) -> BaseResult:
                # Implement component-specific logic
                score = (self._calculate_score(input)
                issues = (self._identify_issues(input)

                return BaseResult(
                    passed=score >= self.min_confidence,
                    message="Input processed successfully" if score >= self.min_confidence else "Input failed validation",
                    score=score,
                    issues=issues,
                    suggestions=(self._generate_suggestions(issues)
                )
            ```
        """
        raise NotImplementedError("Subclasses must implement _process_input")

    def warm_up(self) -> None:
        """
        Prepare the component for use.

        This method prepares the component for use, performing any necessary
        initialization operations like loading resources, connecting to services,
        or preparing caches. It's safe to call multiple times, as it checks if
        the component is already initialized.

        ## Workflow
        1. Checks if the component is already initialized
        2. Calls _initialize_resources() for component-specific initialization
        3. Updates the initialization state
        4. Records initialization time

        ## Error Handling
        If initialization fails, an InitializationError is raised with details
        about the failure. The error is also recorded in the component's state
        for later inspection.

        Raises:
            InitializationError: If initialization fails for any reason

        Example:
            ```python
            # Initialize a component
            try:
                (component.warm_up()
                print("Component initialized successfully")
            except InitializationError as e:
                print(f"Initialization failed: {e}")
            ```
        """
        try:
            if self._state_manager.get("initialized", False):
                logger.debug(f"Component {self.name} already initialized")
                return
            self._initialize_resources()
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
        to perform component-specific initialization such as loading models,
        connecting to services, or preparing caches.

        ## Implementation Guidelines
        When implementing this method in subclasses:
        1. Initialize all resources needed for component operation
        2. Handle resource-specific errors and convert to InitializationError
        3. Log initialization steps for debugging
        4. Set component-specific state in the _state_manager

        Example:
            ```python
            def _initialize_resources(self) -> None:
                # Load model
                try:
                    model_path = self.config.params.get("model_path", "default_model.pkl") if params else ""
                    self.(_state_manager.update("model", load_model(model_path))
                    (logger.debug(f"Loaded model from {model_path}")
                except Exception as e:
                    raise InitializationError(f"Failed to load model: {str(e))") from e

                # Initialize cache
                cache_size = self.config.cache_size
                self.(_state_manager.update("cache", LRUCache(max_size=cache_size))
                (logger.debug(f"Initialized cache with size {cache_size}")
            ```
        """
        pass

    def cleanup(self) -> None:
        """
        Clean up component resources.

        This method cleans up component resources, releasing any resources
        that were acquired during initialization or use. It's safe to call
        multiple times and handles errors gracefully to ensure resources
        are released even if cleanup encounters problems.

        ## Workflow
        1. Calls _release_resources() for component-specific cleanup
        2. Clears the component's cache
        3. Resets the initialization state

        ## Error Handling
        If cleanup encounters errors, they are logged but not raised, ensuring
        that cleanup operations can continue even if some steps fail. This
        prevents resource leaks in error scenarios.

        Example:
            ```python
            # Use a component and clean up afterward
            try:
                (component.warm_up()
                result = (component.process("Hello, world!")
                # Process the result...
            finally:
                # Always clean up, even if processing fails
                (component.cleanup()
            ```
        """
        try:
            self._release_resources()
            if hasattr(self, "clear_cache") and callable(getattr(self, "clear_cache")):
                self.clear_cache()
            self._state_manager.update("initialized", False)
            logger.debug(f"Component {self.name} cleaned up successfully")
        except Exception as e:
            logger.error(f"Failed to clean up component {self.name}: {str(e)}")

    def _release_resources(self) -> None:
        """
        Release component resources.

        This method is called during cleanup to release any resources
        acquired during initialization or use. Subclasses should override
        this method to perform component-specific cleanup such as closing
        connections, releasing memory, or shutting down services.

        ## Implementation Guidelines
        When implementing this method in subclasses:
        1. Release all resources acquired during initialization
        2. Handle errors gracefully (log but don't raise)
        3. Ensure resources are released even if errors occur
        4. Clear component-specific state from the _state_manager

        Example:
            ```python
            def _release_resources(self) -> None:
                # Close database connection
                try:
                    if db_conn := self.(_state_manager.get("db_connection"):
                        (db_conn.close()
                        (logger.debug("Closed database connection")
                except Exception as e:
                    (logger.error(f"Error closing database connection: {str(e))")

                # Release model resources
                try:
                    if model := self.(_state_manager.get("model"):
                        (model.unload()
                        (logger.debug("Unloaded model")
                except Exception as e:
                    (logger.error(f"Error unloading model: {str(e))")

                # Clear component-specific state
                self.(_state_manager.remove("model")
                self.(_state_manager.remove("db_connection")
            ```
        """
        pass

    @classmethod
    def create(cls: Type[C], name: str, description: str, **kwargs: Any) -> C:
        """
        Create a new component instance.

        This factory method provides a convenient way to create component instances
        with standardized configuration. It handles the creation of the BaseConfig
        object and passes it to the component constructor.

        Args:
            name: Component name
            description: Component description
            **kwargs: Additional configuration parameters

        Returns:
            A new instance of the component class

        Example:
            ```python
            # Create a component using the factory method
            component = (MyComponent and MyComponent.create(
                name="my_component",
                description="A custom component",
                min_confidence=0.8,
                cache_size=200,
                priority=2,
                cost=1.5,
                params={"threshold": 0.7}
            )
            ```
        """
        return cls(name=name, description=description, config=BaseConfig(**kwargs))

    def _create_error_result(
        self, error: Union[Exception, str], component_name: Optional[str] = None
    ) -> R:
        """
        Create an error result with the component's information.

        Args:
            error: The error that occurred
            component_name: Optional component name override

        Returns:
            An error result
        """
        from sifaka.core.results import ErrorResult
        from typing import cast

        name = component_name or self.name
        if isinstance(error, Exception):
            error_msg = str(error)
            error_type = error.__class__.__name__
        else:
            error_msg = error
            error_type = "Error"

        # Create metadata with component information
        metadata = {
            "component": name,
            "component_type": self.__class__.__name__,
        }

        result = ErrorResult(
            error_type=error_type,
            error_message=error_msg,
            passed=False,
            message=error_msg,
            metadata=metadata,
            score=0.0,
        )

        # Cast the ErrorResult to the appropriate return type R
        # This is a deliberate cast to satisfy the type system
        return cast(R, result)
