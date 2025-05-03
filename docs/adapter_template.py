"""
Template for standardized Sifaka adapter implementation.

This template provides the standard structure that all adapter implementations
should follow to ensure consistency across the Sifaka framework.

Usage Example:
    from sifaka.adapters.rules import create_adapted_rule

    # Create adapted rule
    adapted_rule = create_adapted_rule(
        adaptee=some_object_to_adapt,
        param1=value1,
        rule_id="custom_adaptation"
    )

    # Use the adapted rule
    result = adapted_rule.validate("This is a test.")
"""

from typing import Any, Dict, Generic, List, Optional, Protocol, TypeVar, Union, runtime_checkable

from pydantic import BaseModel, Field, ConfigDict

from sifaka.rules.base import Rule, RuleConfig, RuleResult, BaseValidator
from sifaka.adapters.rules.base import Adaptable, BaseAdapter, A


# Define type variables
T = TypeVar("T")  # Input type
A = TypeVar("A", bound="Adaptable")  # Adaptee type


@runtime_checkable
class AdapteeType(Protocol):
    """
    Protocol defining the requirements for adaptable types.

    Classes implementing this protocol can be adapted
    for use within the Sifaka rule system.

    Examples:
        ```python
        @runtime_checkable
        class MyAdaptee(Protocol):
            def some_method(self, input_value: str) -> Any:
                ...

            @property
            def some_property(self) -> Any:
                ...
        ```
    """

    # Define required methods and properties
    def some_method(self, input_value: str) -> Any:
        """Example required method."""
        ...

    @property
    def some_property(self) -> Any:
        """Example required property."""
        ...


class AdapterTypeConfig(BaseModel):
    """
    Configuration for adapter_type.

    All configuration parameters should be defined here
    with proper typing, validation, and documentation.

    Attributes:
        param1: Description of parameter 1
        param2: Description of parameter 2
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    param1: type = Field(
        default=default_value,
        description="Description of parameter 1",
        # Add validation constraints as needed
    )

    param2: type = Field(
        default=default_value,
        description="Description of parameter 2",
        # Add validation constraints as needed
    )


class AdapterTypeAdapter(BaseAdapter[T, AdapteeType]):
    """
    Adapter for adapting AdapteeType to Sifaka validators.

    This adapter converts an AdapteeType instance into a format
    that can be used within the Sifaka rule system.

    Type Parameters:
        T: Input type (typically str)
        AdapteeType: The type being adapted

    Lifecycle:
    1. Initialization: Set up with the adaptee and configuration
    2. Validation: Convert input, delegate to adaptee, convert result
    3. Result: Return standardized validation result

    Examples:
        ```python
        from sifaka.adapters.rules import AdapterTypeAdapter
        from my_package import MyExternalComponent

        adaptee = MyExternalComponent()
        adapter = AdapterTypeAdapter(
            adaptee=adaptee,
            param1=value1,
            param2=value2
        )

        result = adapter.validate("Text to validate")
        ```
    """

    def __init__(
        self,
        adaptee: AdapteeType,
        param1: type = default_value,
        param2: type = default_value,
        **kwargs,
    ):
        """
        Initialize adapter with an adaptee and configuration.

        Args:
            adaptee: The object to adapt
            param1: Parameter 1
            param2: Parameter 2
            **kwargs: Additional configuration parameters
        """
        super().__init__(adaptee)

        # Store configuration
        self._config = AdapterTypeConfig(
            param1=param1,
            param2=param2,
        )

    @property
    def config(self) -> AdapterTypeConfig:
        """Get the adapter configuration."""
        return self._config

    @property
    def param1(self) -> type:
        """Get parameter 1."""
        return self._config.param1

    @property
    def param2(self) -> type:
        """Get parameter 2."""
        return self._config.param2

    def _prepare_input(self, input_value: T) -> Any:
        """
        Prepare input for the adaptee.

        Args:
            input_value: Input to prepare

        Returns:
            Prepared input for the adaptee
        """
        # Implement input preparation
        return input_value

    def _convert_result(self, result: Any) -> RuleResult:
        """
        Convert adaptee result to a Sifaka RuleResult.

        Args:
            result: Result from the adaptee

        Returns:
            Converted RuleResult
        """
        # Implement result conversion
        # For example:
        if isinstance(result, bool):
            return RuleResult(
                passed=result,
                message="Validation passed" if result else "Validation failed",
                metadata={
                    "raw_result": result,
                },
            )
        else:
            # Handle other result types
            return RuleResult(
                passed=bool(result),
                message=str(result),
                metadata={
                    "raw_result": result,
                },
            )

    def validate(self, input_value: T, **kwargs) -> RuleResult:
        """
        Validate input using the adaptee.

        Args:
            input_value: Input to validate
            **kwargs: Additional validation context

        Returns:
            Validation result
        """
        # Handle empty text if input is a string
        if isinstance(input_value, str):
            empty_result = self.handle_empty_text(input_value)
            if empty_result:
                return empty_result

        try:
            # Prepare input
            prepared_input = self._prepare_input(input_value)

            # Delegate to adaptee
            adaptee_result = self.adaptee.some_method(prepared_input)

            # Convert and return result
            return self._convert_result(adaptee_result)
        except Exception as e:
            # Handle exceptions
            return RuleResult(
                passed=False,
                message=f"Validation error: {str(e)}",
                metadata={
                    "error_type": type(e).__name__,
                    "input_type": type(input_value).__name__,
                },
            )


class AdapterTypeRule(Rule):
    """
    Rule that uses an adapted AdapteeType for validation.

    This rule combines the adapter with the Sifaka rule system.

    Lifecycle:
    1. Initialization: Set up with an adaptee and configuration
    2. Validation: Delegate to adapter
    3. Result: Return validation result with rule metadata

    Examples:
        ```python
        from sifaka.adapters.rules import AdapterTypeRule
        from my_package import MyExternalComponent

        adaptee = MyExternalComponent()
        rule = AdapterTypeRule(
            adaptee=adaptee,
            param1=value1,
            name="custom_rule",
            description="Custom description"
        )

        result = rule.validate("Text to validate")
        ```
    """

    def __init__(
        self,
        adaptee: AdapteeType,
        param1: type = default_value,
        param2: type = default_value,
        rule_id: Optional[str] = None,
        name: Optional[str] = None,
        description: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize rule with an adaptee and configuration.

        Args:
            adaptee: The object to adapt
            param1: Parameter 1
            param2: Parameter 2
            rule_id: Unique identifier for the rule
            name: Name of the rule
            description: Description of the rule
            **kwargs: Additional rule configuration
        """
        # Set default name and description if not provided
        if name is None:
            name = f"adapted_{adaptee.__class__.__name__}_rule"
        if description is None:
            description = f"Validates input using adapted {adaptee.__class__.__name__}"

        # Create rule configuration
        rule_config = RuleConfig(
            params={
                "param1": param1,
                "param2": param2,
            }
        )

        # Create adapter
        self._adapter = AdapterTypeAdapter(
            adaptee=adaptee,
            param1=param1,
            param2=param2,
        )

        # Store essential attributes
        self._rule_id = rule_id or name
        self._adaptee = adaptee

        # Initialize base class
        super().__init__(
            name=name,
            description=description,
            config=rule_config,
            **kwargs,
        )

    def _create_default_validator(self) -> BaseValidator[T]:
        """
        Create default validator for this rule.

        Returns:
            Validator using the adapter
        """
        return self._adapter

    @property
    def rule_id(self) -> str:
        """Get the rule ID."""
        return self._rule_id

    @property
    def adaptee(self) -> AdapteeType:
        """Get the adapted object."""
        return self._adaptee

    def validate(self, input_value: T, **kwargs) -> RuleResult:
        """
        Validate input.

        Args:
            input_value: Input to validate
            **kwargs: Additional validation context

        Returns:
            Validation result
        """
        # Delegate to adapter
        result = self._adapter.validate(input_value, **kwargs)

        # Add rule_id to metadata
        return result.with_metadata(rule_id=self._rule_id)


def create_adapted_rule(
    adaptee: AdapteeType,
    param1: type = default_value,
    param2: type = default_value,
    rule_id: Optional[str] = None,
    name: Optional[str] = None,
    description: Optional[str] = None,
    **kwargs,
) -> AdapterTypeRule:
    """
    Create a rule that adapts an AdapteeType.

    This factory function creates a rule that uses an adapter
    to work with the provided adaptee.

    Args:
        adaptee: The object to adapt
        param1: Parameter 1
        param2: Parameter 2
        rule_id: Unique identifier for the rule
        name: Name of the rule
        description: Description of the rule
        **kwargs: Additional rule configuration

    Returns:
        Configured rule

    Examples:
        ```python
        from sifaka.adapters.rules import create_adapted_rule
        from my_package import MyExternalComponent

        adaptee = MyExternalComponent()
        rule = create_adapted_rule(
            adaptee=adaptee,
            param1=value1,
            rule_id="custom_rule"
        )

        result = rule.validate("Text to validate")
        ```
    """
    return AdapterTypeRule(
        adaptee=adaptee,
        param1=param1,
        param2=param2,
        rule_id=rule_id,
        name=name,
        description=description,
        **kwargs,
    )


# Export public components
__all__ = [
    # Protocols
    "AdapteeType",
    # Config classes
    "AdapterTypeConfig",
    # Adapter classes
    "AdapterTypeAdapter",
    # Rule classes
    "AdapterTypeRule",
    # Factory functions
    "create_adapted_rule",
]