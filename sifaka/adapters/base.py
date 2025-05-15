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
        result = self.adaptee.process(input_value) if adaptee else ""
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
from typing import Any, Dict, Generic, Optional, Type, TypeVar, cast
from pydantic import BaseModel, PrivateAttr, ConfigDict
from sifaka.rules.base import RuleResult
from sifaka.utils.errors.base import ConfigurationError, ValidationError
from sifaka.utils.state import create_adapter_state, AdapterState
from sifaka.utils.logging import get_logger
from sifaka.utils.errors.base import SifakaError
from sifaka.utils.errors.handling import handle_error

logger = get_logger(__name__)
T = TypeVar("T")
R = TypeVar("R")
C = TypeVar("C", bound="Adaptable")


class AdapterError(SifakaError):
    """
    Base class for adapter-specific errors.

    This class provides a standardized structure for adapter exceptions,
    including a message and optional metadata.

    Attributes:
        message: Human-readable error message
        metadata: Additional error context and details
    """

    def __init__(self, message: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Initialize an AdapterError.

        Args:
            message: Human-readable error message
            metadata: Additional error context and details
        """
        super().__init__(message, metadata)


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
            result = self.adaptee.process(input_value) if adaptee else ""
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

    model_config = ConfigDict(arbitrary_types_allowed=True)
    _state_manager = PrivateAttr(default_factory=create_adapter_state)

    @property
    def validation_type(self) -> Type[Any]:
        """
        Get the type of input this adapter validates.

        Returns:
            Type[Any]: The type of input this adapter accepts for validation
        """
        # Since we can't directly access the TypeVar T at runtime,
        # we need to use a different approach to determine the validation type
        # This is a simplified implementation that returns str as a fallback
        # In a real implementation, this would need to be more sophisticated
        return str

    def __init__(self, adaptee: C) -> None:
        """
        Initialize the adapter with a component to adapt.

        Args:
            adaptee (C): The component to adapt for validation

        Raises:
            ConfigurationError: If the adaptee is invalid or incompatible
        """
        super().__init__()
        self._validate_adaptee(adaptee)
        # Initialize state attributes using the update method
        self._state_manager.update("adaptee", adaptee)
        self._state_manager.update("initialized", True)
        self._state_manager.update("execution_count", 0)
        self._state_manager.update("error_count", 0)
        self._state_manager.update("last_execution_time", None)
        self._state_manager.update("avg_execution_time", 0)
        self._state_manager.update("cache", {})
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
    def adaptee(self) -> Any:
        """
        Get the component being adapted.

        Returns:
            Any: The component instance being adapted
        """
        return self._state_manager.get("adaptee")

    def warm_up(self) -> None:
        """
        Initialize the adapter if needed.

        This method ensures the adapter is properly initialized before use.
        It's safe to call multiple times.

        Raises:
            AdapterError: If initialization fails
        """
        try:
            if self._state_manager.get("initialized", False):
                return
            self._state_manager.update("initialized", True)
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

        # The handle_empty_text function returns a BaseResult[Any], but we need a RuleResult
        # Convert the result if it's not None
        result = handle_empty_text(text, passed=True, component_type="adapter")
        if result is None:
            return None

        # Convert BaseResult to RuleResult if needed
        if not isinstance(result, RuleResult):
            return RuleResult(
                passed=result.passed, message=result.message, metadata=result.metadata
            )

        return result

    def _validate_impl(self, input_value: Any, **kwargs: Any) -> RuleResult:
        """
        Implement validation logic specific to this adapter.

        This method must be implemented by subclasses to define
        how validation is performed using the adapted component.

        Args:
            input_value (Any): The value to validate
            **kwargs: Additional validation parameters

        Returns:
            RuleResult: RuleResult containing validation outcome and metadata

        Raises:
            ValidationError: If validation fails due to an error
            AdapterError: If adapter-specific error occurs
            NotImplementedError: If the method is not implemented by a subclass
        """
        raise NotImplementedError(
            f"_validate_impl must be implemented by subclass {self.__class__.__name__}"
        )

    def validate(self, input_value: Any, **kwargs: Any) -> RuleResult:  # type: ignore[override]
        """
        Validate the input value using the adapted component.

        This method overrides the validate method from BaseModel but with a different signature.
        It's used for validation in the adapter context rather than model validation.

        Args:
            input_value (Any): The value to validate
            **kwargs: Additional validation parameters

        Returns:
            RuleResult: RuleResult containing validation outcome and metadata

        Raises:
            ValidationError: If validation fails due to an error
            AdapterError: If adapter-specific error occurs
        """
        self.warm_up()
        execution_count = self._state_manager.get("execution_count", 0) + 1
        self._state_manager.update("execution_count", execution_count)
        start_time = time.time()
        try:
            cache_key = self._get_cache_key(input_value, kwargs)
            cache = self._state_manager.get("cache", {})
            if cache_key and cache_key in cache:
                logger.debug(f"Cache hit for adapter {self.__class__.__name__}")
                return cache[cache_key]
            if isinstance(input_value, str):
                empty_result = self.handle_empty_text(input_value)
                if empty_result:
                    return empty_result
            result = self._validate_impl(input_value, **kwargs)
            if cache_key:
                cache[cache_key] = result
                self._state_manager.update("cache", cache)
            return result
        except Exception as e:
            error_count = self._state_manager.get("error_count", 0) + 1
            self._state_manager.update("error_count", error_count)
            if isinstance(e, ValidationError):
                raise
            elif isinstance(e, AdapterError):
                raise
            else:
                error_info = handle_error(e, f"Adapter:{self.__class__.__name__}")
                raise ValidationError(f"Validation failed: {str(e)}", metadata=error_info) from e
        finally:
            execution_time = time.time() - start_time
            self._state_manager.update("last_execution_time", execution_time)
            avg_execution_time = self._state_manager.get("avg_execution_time", 0)
            if execution_count > 1:
                avg_execution_time = (
                    avg_execution_time * (execution_count - 1) + execution_time
                ) / execution_count
            else:
                avg_execution_time = execution_time
            self._state_manager.update("avg_execution_time", avg_execution_time)

    def _get_cache_key(self, input_value: Any, kwargs: Dict[str, Any]) -> Optional[str]:
        """
        Generate a cache key for the input value and kwargs.

        Args:
            input_value (Any): The input value
            kwargs (Dict[str, Any]): Additional parameters

        Returns:
            Optional[str]: Cache key or None if caching is disabled
        """
        if isinstance(input_value, str):
            return f"{hash(input_value)}:{hash(str(kwargs))}"
        return None

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about adapter usage.

        Returns:
            Dict[str, Any]: Dictionary with usage statistics
        """
        return {
            "execution_count": self._state_manager.get("execution_count", 0),
            "error_count": self._state_manager.get("error_count", 0),
            "avg_execution_time": self._state_manager.get("avg_execution_time", 0),
            "last_execution_time": self._state_manager.get("last_execution_time"),
            "cache_size": len(self._state_manager.get("cache", {})),
            "adaptee_name": getattr(self.adaptee, "name", str(self.adaptee)),
        }


def create_adapter(
    adapter_type: Type[BaseAdapter[Any, Any]],
    adaptee: Any,
    name: Optional[str] = None,
    description: Optional[str] = None,
    initialize: bool = True,
    **kwargs: Any,
) -> BaseAdapter[Any, Any]:
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
    if not issubclass(adapter_type, BaseAdapter):
        raise ConfigurationError(
            f"adapter_type must be a subclass of BaseAdapter, got {adapter_type}"
        )
    try:
        adapter = adapter_type(adaptee=adaptee, **kwargs)
        if name:
            adapter._state_manager.set_metadata("name", name)
        if description:
            adapter._state_manager.set_metadata("description", description)
        if initialize:
            adapter.warm_up()
        return adapter
    except Exception as e:
        if isinstance(e, (ConfigurationError, AdapterError)):
            raise
        error_info = handle_error(e, f"AdapterFactory:{adapter_type.__name__}")
        raise AdapterError(f"Failed to create adapter: {str(e)}", metadata=error_info) from e
