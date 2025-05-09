# pylint: disable=all
# flake8: noqa
# mypy: ignore-errors
"""
Base Adapters

Base classes and protocols for adapter-based rules in the Sifaka framework.

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

## Configuration
- adaptee: The component being adapted
- validation_type: The type of input this adapter validates
"""

from typing import Any, Dict, Generic, Optional, Protocol, Type, TypeVar, cast, runtime_checkable

from pydantic import BaseModel, PrivateAttr, ConfigDict
from sifaka.rules.base import BaseValidator, ConfigurationError, RuleResult, ValidationError
from sifaka.utils.state import AdapterState, StateManager, create_adapter_state


T = TypeVar("T")  # Input type
A = TypeVar("A", bound="Adaptable")  # Adaptee type


@runtime_checkable
class Adaptable(Protocol):
    """
    Protocol for components that can be adapted to rules.

    ## Overview
    Any component that can be adapted to a Sifaka rule must implement
    this protocol, which requires a name and description.

    ## Architecture
    The protocol defines a minimal interface that components must implement
    to be compatible with Sifaka's adapter system.

    ## Lifecycle
    1. **Implementation**: Component implements the required properties
       - Provide a name for identification
       - Provide a description of functionality

    2. **Adaptation**: Component is adapted using a compatible adapter
       - Adapter receives component instance
       - Adapter validates component compatibility

    3. **Usage**: Adapted component is used as a Sifaka rule validator
       - Adapter translates between component and rule interfaces
       - Component's functionality is leveraged in validation

    ## Error Handling
    - TypeError: Raised if required properties are not implemented
    - ValueError: Raised if property values are invalid

    ## Examples
    ```python
    class MyComponent:
        @property
        def name(self) -> str:
            return "my_component"

        @property
        def description(self) -> str:
            return "A custom component"
    ```
    """

    @property
    def name(self) -> str:
        """
        Get the component name.

        Returns:
            str: A string name for the component

        Raises:
            NotImplementedError: If the property is not implemented
        """
        ...

    @property
    def description(self) -> str:
        """
        Get the component description.

        Returns:
            str: A string description of the component's purpose

        Raises:
            NotImplementedError: If the property is not implemented
        """
        ...


class BaseAdapter(BaseModel, BaseValidator[T], Generic[T, A]):
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

    ## Error Handling
    - ConfigurationError: Raised when adaptee is invalid
    - ValidationError: Raised when validation fails
    - TypeError: Raised when input types are incompatible

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
        adaptee (A): The component being adapted
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

    def __init__(self, adaptee: A) -> None:
        """
        Initialize the adapter with a component to adapt.

        Args:
            adaptee (A): The component to adapt for validation

        Raises:
            ConfigurationError: If the adaptee is invalid or incompatible
        """
        # Validate adaptee
        self._validate_adaptee(adaptee)

        # Initialize state
        state = self._state_manager.get_state()
        state.adaptee = adaptee
        state.initialized = True

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
    def adaptee(self) -> A:
        """
        Get the component being adapted.

        Returns:
            A: The component instance being adapted
        """
        return self._state_manager.get_state().adaptee

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
        """
        try:
            # Handle empty text if applicable
            if isinstance(input_value, str):
                empty_result = self.handle_empty_text(input_value)
                if empty_result:
                    return empty_result

            # Delegate to specific implementation
            return self._validate_impl(input_value, **kwargs)
        except Exception as e:
            raise ValidationError(f"Validation failed: {str(e)}") from e


def create_adapter(
    adapter_type: Type[BaseAdapter[T, A]], adaptee: A, **kwargs: Any
) -> BaseAdapter[T, A]:
    """
    Factory function to create an adapter with standardized configuration.

    ## Overview
    This function simplifies the creation of adapters by providing a
    consistent interface.

    Args:
        adapter_type (Type[BaseAdapter[T, A]]): The class of the adapter to create
        adaptee (A): The component to adapt
        **kwargs: Additional keyword arguments for the adapter

    Returns:
        BaseAdapter[T, A]: A configured adapter instance

    Raises:
        ConfigurationError: If adapter_type is not a subclass of BaseAdapter
        ConfigurationError: If adaptee doesn't implement required protocol

    ## Examples
    ```python
    adapter = create_adapter(
        adapter_type=CustomAdapter,
        adaptee=my_component,
        additional_param="value"
    )
    ```
    """
    if not issubclass(adapter_type, BaseAdapter):
        raise ConfigurationError(
            f"adapter_type must be a subclass of BaseAdapter, got {adapter_type}"
        )

    return adapter_type(adaptee=adaptee, **kwargs)
