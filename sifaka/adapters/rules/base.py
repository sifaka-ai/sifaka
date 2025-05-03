# pylint: disable=all
# flake8: noqa
# mypy: ignore-errors
"""
Base classes and protocols for adapter-based rules.

This module provides the foundation for adapting various components to function as validation rules,
such as classifiers, models, or external services.

See examples in the tests/ directory for usage patterns.
"""

from typing import Any, Dict, Generic, Optional, Protocol, Type, TypeVar, cast, runtime_checkable

from sifaka.rules.base import BaseValidator, ConfigurationError, RuleResult, ValidationError


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


class BaseAdapter(BaseValidator[T], Generic[T, A]):
    """
    Base class for adapters that convert components to validators.

    This abstract class provides the foundation for adapting external
    components to work with Sifaka's rule system. It handles common
    tasks like validating the adaptee and providing an interface for
    validation.

    Type Parameters:
        T: The input type to validate
        A: The adaptee type, must implement Adaptable

    ## Lifecycle

    1. **Initialization**: Receive component to adapt
       - Validate adaptee implements required protocol
       - Store reference to adaptee
       - Set up any additional configuration

    2. **Validation**: Process input through adaptee
       - Handle empty or invalid inputs
       - Preprocess input if necessary
       - Pass input to adaptee with appropriate parameters
       - Catch and handle any adaptee errors

    3. **Result Conversion**: Translate adaptee outputs to RuleResults
       - Convert adaptee-specific outputs to standard RuleResults
       - Include appropriate metadata from adaptee
       - Set passed/failed status based on adaptee response

    ## Error Handling

    Implementations should handle these error cases:

    - **Adaptee Validation**: Check at initialization that adaptee meets requirements
    - **Empty Inputs**: Handle empty inputs with the handle_empty_text() method
    - **Adaptee Errors**: Catch and properly handle errors from the adaptee
    - **Input Type Errors**: Validate input types before processing
    """

    @property
    def validation_type(self) -> type:
        """
        Get the type of input this validator can validate.

        Returns:
            The type this validator can validate (default: str)
        """
        return str

    def __init__(self, adaptee: A) -> None:
        """
        Initialize with adaptee.

        Args:
            adaptee: The component being adapted

        Raises:
            ConfigurationError: If adaptee doesn't implement Adaptable protocol
        """
        self._validate_adaptee(adaptee)
        self._adaptee = adaptee

    def _validate_adaptee(self, adaptee: Any) -> None:
        """
        Validate that adaptee implements the required protocol.

        This internal method ensures the adaptee meets the requirements
        for adaptation.

        Args:
            adaptee: The component to validate

        Raises:
            ConfigurationError: If adaptee doesn't implement Adaptable protocol
        """
        if not isinstance(adaptee, Adaptable):
            raise ConfigurationError(
                f"Adaptee must implement Adaptable protocol, got {type(adaptee)}"
            )

    @property
    def adaptee(self) -> A:
        """
        Get the adaptee.

        Returns:
            The component being adapted
        """
        return self._adaptee

    def handle_empty_text(self, text: str) -> Optional[RuleResult]:
        """
        Handle empty text validation.

        This method provides consistent handling of empty text inputs
        across different adapters.

        Args:
            text: The text to check

        Returns:
            RuleResult if text is empty, None otherwise
        """
        if not isinstance(text, str):
            return None

        if not text.strip():
            return RuleResult(
                passed=True,
                message="Empty text validation skipped",
                metadata={"reason": "empty_input", "adaptee_name": self.adaptee.name},
            )
        return None

    def validate(self, input_value: T, **kwargs) -> RuleResult:
        """
        Validate using the adaptee.

        This method should be implemented by subclasses to perform
        the actual validation using the adaptee.

        Args:
            input_value: Input to validate
            **kwargs: Additional validation context

        Returns:
            RuleResult with validation results

        Raises:
            ValidationError: If validation fails
            NotImplementedError: If not implemented by subclass
        """
        try:
            # Handle empty text if input is a string
            if isinstance(input_value, str):
                empty_result = self.handle_empty_text(input_value)
                if empty_result:
                    return empty_result

            # This is an abstract method that should be implemented by subclasses
            raise NotImplementedError("Subclasses must implement the validate method")
        except Exception as e:
            raise ValidationError(f"Validation failed: {str(e)}") from e


def create_adapter(
    adapter_type: Type[BaseAdapter[T, A]],
    adaptee: A,
    **kwargs: Any
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
        raise ConfigurationError(f"adapter_type must be a subclass of BaseAdapter, got {adapter_type}")

    return adapter_type(adaptee=adaptee, **kwargs)