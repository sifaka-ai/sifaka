"""
Base classes for Sifaka adapters.

This module provides the foundational components for adapter implementations.

## Overview
This module provides the foundation for adapting various components to function as validation rules,
such as classifiers, models, or external services. It defines the core interfaces and base classes
that enable the adapter pattern implementation throughout the Sifaka framework.

## Components
1. Adaptable Protocol: Defines the interface for components that can be adapted
2. BaseAdapter: Abstract base class for implementing specific adapters
3. Validation Types: Support for different input types and validation strategies

## Usage Examples
```python
class CustomAdapter(BaseAdapter[str, CustomComponent]):
    def validate(self, input_value: str, **kwargs) -> RuleResult:
        # Convert component's functionality to validation
        result = self.adaptee.process(input_value)
        return RuleResult(
            passed=result.is_valid,
            message=result.message,
            metadata={"confidence": result.confidence}
        )
```

## Error Handling
- ConfigurationError: Raised when adapter configuration is invalid
- ValidationError: Raised when validation fails due to an error
- TypeError: Raised when input types are incompatible
- AdapterError: Base class for adapter-specific errors

## State Management
The module uses a standardized state management approach:
- Single _state_manager attribute for all mutable state
- State initialization during construction
- State access through state object
- Clear separation of configuration and state
- Execution tracking for monitoring and debugging

## Configuration
- adaptee: The component being adapted
- validation_type: The type of input this adapter validates
"""

import time
from typing import (
    Any,
    Dict,
    Generic,
    Optional,
    Type,
    TypeVar,
)

from pydantic import BaseModel, PrivateAttr, ConfigDict
from sifaka.rules.base import RuleResult
from sifaka.utils.errors import ConfigurationError, ValidationError
from sifaka.utils.state import create_adapter_state
from sifaka.utils.logging import get_logger
from sifaka.utils.errors import SifakaError, handle_error

logger = get_logger(__name__)

# Type variables
T = TypeVar("T")  # Input type
R = TypeVar("R")  # Result type
C = TypeVar("C", bound="Adaptable")  # Adaptable component type


class AdapterError(SifakaError):
    """
    Base class for adapter-specific errors.

    This class provides a standardized structure for adapter exceptions,
    including a message and optional metadata.

    Attributes:
        message: Human-readable error message
        metadata: Additional error context and details
    """

    def __init__(self, message: str, metadata: Optional[Dict[str, Any]] = None):
        """Initialize an AdapterError.

        Args:
            message: Human-readable error message
            metadata: Additional error context and details
        """
        super().__init__(message, metadata)


# Import from the main interfaces directory
from sifaka.interfaces.adapter import Adaptable


class BaseAdapter(BaseModel, Generic[T, C]):
    """
    Base class for implementing adapters that convert components to validators.

    ## Overview
    This class serves as the foundation for creating adapters that bridge between
    external components and Sifaka's validation system. It handles the core
    adaptation logic while allowing specific implementations to customize behavior.

    ## Architecture
    The adapter follows a standard adapter pattern:
    1. Receives an adaptee component
    2. Translates between component and validator interfaces
    3. Provides standardized validation results

    ## Lifecycle
    1. Initialization: Set up with adaptee and configuration
    2. Validation: Convert adaptee's functionality to validation
    3. Result Handling: Standardize validation results

    ## State Management
    The class uses a standardized state management approach:
    - Single _state_manager attribute for all mutable state
    - State initialization during construction
    - State access through state object
    - Clear separation of configuration and state
    - State components:
      - adaptee: The component being adapted
      - initialized: Initialization status
      - execution_count: Number of validation executions
      - last_execution_time: Timestamp of last execution
      - avg_execution_time: Average execution time
      - error_count: Number of validation errors
      - cache: Temporary data storage

    ## Error Handling
    - ConfigurationError: Raised when adaptee is invalid
    - ValidationError: Raised when validation fails
    - TypeError: Raised when input types are incompatible
    - AdapterError: Raised for adapter-specific errors

    ## Examples
    ```python
    class CustomAdapter(BaseAdapter[str, CustomComponent]):
        def validate(self, input_value: str, **kwargs) -> RuleResult:
            result = self.adaptee.process(input_value)
            return RuleResult(
                passed=result.is_valid,
                message=result.message,
                metadata={"confidence": result.confidence}
            )
    ```

    Attributes:
        adaptee (C): The component being adapted
        validation_type (Type[T]): The type of input this adapter validates
    """

    # Pydantic configuration
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # State management using StateManager
    _state_manager = PrivateAttr(default_factory=create_adapter_state)

    @property
    def validation_type(self) -> type:
        """
        Get the type of input this adapter validates.

        Returns:
            type: The type of input this adapter accepts for validation
        """
        return T

    def __init__(self, adaptee: C) -> None:
        """
        Initialize the adapter with a component to adapt.

        Args:
            adaptee (C): The component to adapt for validation

        Raises:
            ConfigurationError: If the adaptee is invalid or incompatible
        """
        # Initialize base model
        super().__init__()

        # Validate adaptee
        self._validate_adaptee(adaptee)

        # Initialize state
        state = self._state_manager.get_state()
        state.adaptee = adaptee
        state.initialized = True
        state.execution_count = 0
        state.error_count = 0
        state.last_execution_time = None
        state.avg_execution_time = 0
        state.cache = {}

        # Set metadata
        self._state_manager.set_metadata("component_type", "adapter")
        self._state_manager.set_metadata("creation_time", time.time())
        self._state_manager.set_metadata("adapter_type", self.__class__.__name__)

    def _validate_adaptee(self, adaptee: Any) -> None:
        """
        Validate that the adaptee is compatible with this adapter.

        Args:
            adaptee (Any): The component to validate

        Raises:
            ConfigurationError: If the adaptee is invalid or incompatible
        """
        if not isinstance(adaptee, Adaptable):
            raise ConfigurationError(
                f"Adaptee must implement Adaptable protocol, got {type(adaptee)}"
            )

    @property
    def adaptee(self) -> C:
        """
        Get the component being adapted.

        Returns:
            C: The component instance being adapted
        """
        return self._state_manager.get_state().adaptee

    def warm_up(self) -> None:
        """
        Initialize the adapter if needed.

        This method ensures the adapter is properly initialized before use.
        It's safe to call multiple times.

        Raises:
            AdapterError: If initialization fails
        """
        try:
            # Check if already initialized
            if self._state_manager.get_state().initialized:
                return

            # Initialize state
            state = self._state_manager.get_state()
            state.initialized = True

            logger.debug(f"Adapter {self.__class__.__name__} initialized")
        except Exception as e:
            error_info = handle_error(e, f"Adapter:{self.__class__.__name__}")
            raise AdapterError(
                f"Failed to initialize adapter: {str(e)}", metadata=error_info
            ) from e

    def handle_empty_text(self, text: str) -> Optional[RuleResult]:
        """
        Handle empty or invalid input text.

        Args:
            text (str): The input text to check

        Returns:
            Optional[RuleResult]: RuleResult if the text is empty or invalid, None otherwise
        """
        from sifaka.utils.text import handle_empty_text

        # For backward compatibility, adapters continue to return passed=True for empty text
        # This ensures consistent behavior with the rest of the codebase
        return handle_empty_text(text, passed=True, component_type="adapter")

    def validate(self, input_value: T, **kwargs) -> RuleResult:
        """
        Validate the input value using the adapted component.

        Args:
            input_value (T): The value to validate
            **kwargs: Additional validation parameters

        Returns:
            RuleResult: RuleResult containing validation outcome and metadata

        Raises:
            ValidationError: If validation fails due to an error
            AdapterError: If adapter-specific error occurs
        """
        # Ensure initialized
        self.warm_up()

        # Get state
        state = self._state_manager.get_state()

        # Track execution
        state.execution_count += 1
        start_time = time.time()

        try:
            # Check cache if enabled
            cache_key = self._get_cache_key(input_value, kwargs)
            if cache_key and cache_key in state.cache:
                logger.debug(f"Cache hit for adapter {self.__class__.__name__}")
                return state.cache[cache_key]

            # Handle empty text if applicable
            if isinstance(input_value, str):
                empty_result = self.handle_empty_text(input_value)
                if empty_result:
                    return empty_result

            # Delegate to specific implementation
            result = self._validate_impl(input_value, **kwargs)

            # Update cache if enabled
            if cache_key:
                state.cache[cache_key] = result

            return result
        except Exception as e:
            # Track error
            state.error_count += 1

            # Handle different error types
            if isinstance(e, ValidationError):
                raise
            elif isinstance(e, AdapterError):
                raise
            else:
                error_info = handle_error(e, f"Adapter:{self.__class__.__name__}")
                raise ValidationError(f"Validation failed: {str(e)}", metadata=error_info) from e
        finally:
            # Update execution stats
            execution_time = time.time() - start_time
            state.last_execution_time = execution_time

            # Update average execution time
            if state.execution_count > 1:
                state.avg_execution_time = (
                    state.avg_execution_time * (state.execution_count - 1) + execution_time
                ) / state.execution_count
            else:
                state.avg_execution_time = execution_time

    def _get_cache_key(self, input_value: T, kwargs: Dict[str, Any]) -> Optional[str]:
        """
        Generate a cache key for the input value and kwargs.

        Args:
            input_value (T): The input value
            kwargs (Dict[str, Any]): Additional parameters

        Returns:
            Optional[str]: Cache key or None if caching is disabled
        """
        # Default implementation - subclasses can override
        if isinstance(input_value, str):
            # Simple string hash for text inputs
            return f"{hash(input_value)}:{hash(str(kwargs))}"
        return None

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about adapter usage.

        Returns:
            Dict[str, Any]: Dictionary with usage statistics
        """
        state = self._state_manager.get_state()
        return {
            "execution_count": state.execution_count,
            "error_count": state.error_count,
            "avg_execution_time": state.avg_execution_time,
            "last_execution_time": state.last_execution_time,
            "cache_size": len(state.cache),
            "adaptee_name": getattr(self.adaptee, "name", str(self.adaptee)),
        }


def create_adapter(
    adapter_type: Type[BaseAdapter[T, C]],
    adaptee: C,
    name: Optional[str] = None,
    description: Optional[str] = None,
    initialize: bool = True,
    **kwargs: Any,
) -> BaseAdapter[T, C]:
    """
    Factory function to create an adapter with standardized configuration.

    ## Overview
    This function simplifies the creation of adapters by providing a
    consistent interface with standardized configuration options.

    ## Architecture
    The factory function follows a standard pattern:
    1. Validate inputs
    2. Create adapter instance
    3. Initialize if requested
    4. Return configured instance

    ## Lifecycle
    1. Validation: Ensure adapter_type is valid
    2. Creation: Create adapter instance
    3. Initialization: Initialize adapter if requested
    4. Return: Return configured adapter

    Args:
        adapter_type (Type[BaseAdapter[T, C]]): The class of the adapter to create
        adaptee (C): The component to adapt
        name (Optional[str]): Optional name for the adapter
        description (Optional[str]): Optional description for the adapter
        initialize (bool): Whether to initialize the adapter immediately
        **kwargs: Additional keyword arguments for the adapter

    Returns:
        BaseAdapter[T, C]: A configured adapter instance

    Raises:
        ConfigurationError: If adapter_type is not a subclass of BaseAdapter
        ConfigurationError: If adaptee doesn't implement required protocol
        AdapterError: If initialization fails

    ## Examples
    ```python
    # Basic usage
    adapter = create_adapter(
        adapter_type=CustomAdapter,
        adaptee=my_component
    )

    # With custom name and description
    adapter = create_adapter(
        adapter_type=CustomAdapter,
        adaptee=my_component,
        name="my_custom_adapter",
        description="A custom adapter for my component"
    )

    # With additional parameters
    adapter = create_adapter(
        adapter_type=CustomAdapter,
        adaptee=my_component,
        threshold=0.8,
        valid_labels=["positive", "neutral"]
    )

    # Without immediate initialization
    adapter = create_adapter(
        adapter_type=CustomAdapter,
        adaptee=my_component,
        initialize=False
    )
    ```
    """
    # Validate adapter type
    if not issubclass(adapter_type, BaseAdapter):
        raise ConfigurationError(
            f"adapter_type must be a subclass of BaseAdapter, got {adapter_type}"
        )

    try:
        # Create adapter instance
        adapter = adapter_type(adaptee=adaptee, **kwargs)

        # Set name and description if provided
        if name:
            adapter._state_manager.set_metadata("name", name)
        if description:
            adapter._state_manager.set_metadata("description", description)

        # Initialize if requested
        if initialize:
            adapter.warm_up()

        return adapter
    except Exception as e:
        # Handle errors
        if isinstance(e, (ConfigurationError, AdapterError)):
            raise

        # Convert other errors to AdapterError
        error_info = handle_error(e, f"AdapterFactory:{adapter_type.__name__}")
        raise AdapterError(f"Failed to create adapter: {str(e)}", metadata=error_info) from e
