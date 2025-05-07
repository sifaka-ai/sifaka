# pylint: disable=all
# flake8: noqa
# mypy: ignore-errors
"""
Base classes and protocols for adapter-based rules.

This module provides the foundation for adapting various components to function as validation rules,
such as classifiers, models, or external services. It defines the core interfaces and base classes
that enable the adapter pattern implementation throughout the Sifaka framework.

Key Components:
1. Adaptable Protocol: Defines the interface for components that can be adapted
2. BaseAdapter: Abstract base class for implementing specific adapters
3. Validation Types: Support for different input types and validation strategies

See examples in the tests/ directory for usage patterns.
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

    Any component that can be adapted to a Sifaka rule must implement
    this protocol, which requires a name and description.

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
    """

    @property
    def name(self) -> str:
        """
        Get the component name.

        Returns:
            A string name for the component
        """
        ...

    @property
    def description(self) -> str:
        """
        Get the component description.

        Returns:
            A string description of the component's purpose
        """
        ...


class BaseAdapter(BaseModel, BaseValidator[T], Generic[T, A]):
    """
    Base class for implementing adapters that convert components to validators.

    This class serves as the foundation for creating adapters that bridge between
    external components and Sifaka's validation system. It handles the core
    adaptation logic while allowing specific implementations to customize behavior.

    Attributes:
        adaptee: The component being adapted
        validation_type: The type of input this adapter validates

    Lifecycle:
    1. Initialization: Set up with adaptee and configuration
    2. Validation: Convert adaptee's functionality to validation
    3. Result Handling: Standardize validation results

    Examples:
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
            The type of input this adapter accepts for validation
        """
        return T

    def __init__(self, adaptee: A) -> None:
        """
        Initialize the adapter with a component to adapt.

        Args:
            adaptee: The component to adapt for validation

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
            adaptee: The component to validate

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
            The component instance being adapted
        """
        return self._state_manager.get_state().adaptee

    def handle_empty_text(self, text: str) -> Optional[RuleResult]:
        """
        Handle empty or invalid input text.

        Args:
            text: The input text to check

        Returns:
            RuleResult if the text is empty or invalid, None otherwise
        """
        if not text or not text.strip():
            return RuleResult(
                passed=False, message="Empty text provided", metadata={"input_length": len(text)}
            )
        return None

    def validate(self, input_value: T, **kwargs) -> RuleResult:
        """
        Validate the input value using the adapted component.

        Args:
            input_value: The value to validate
            **kwargs: Additional validation parameters

        Returns:
            RuleResult containing validation outcome and metadata

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

    This function simplifies the creation of adapters by providing a
    consistent interface.

    Args:
        adapter_type: The class of the adapter to create
        adaptee: The component to adapt
        **kwargs: Additional keyword arguments for the adapter

    Returns:
        A configured adapter instance

    Raises:
        ConfigurationError: If adapter_type is not a subclass of BaseAdapter
        ConfigurationError: If adaptee doesn't implement required protocol
    """
    if not issubclass(adapter_type, BaseAdapter):
        raise ConfigurationError(
            f"adapter_type must be a subclass of BaseAdapter, got {adapter_type}"
        )

    return adapter_type(adaptee=adaptee, **kwargs)
