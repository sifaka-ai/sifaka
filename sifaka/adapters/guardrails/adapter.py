"""
Guardrails Adapter

Adapter for using Guardrails validators as rules in the Sifaka framework.

## Overview
This module provides adapters for using validators from the Guardrails library as Sifaka rules.
It enables seamless integration between Guardrails' validation capabilities and Sifaka's
rule system, allowing for sophisticated content validation.

## Components
1. **GuardrailsValidatable Protocol**: Defines the expected interface for Guardrails validators
2. **GuardrailsAdapter**: Adapts Guardrails validators to work as Sifaka validators
3. **GuardrailsRule**: Rule that uses a Guardrails validator for validation
4. **Factory Functions**: Simple creation patterns for Guardrails-based rules

## Usage Examples
```python
from guardrails.hub import RegexMatch
from sifaka.adapters.guardrails import create_guardrails_adapter

# Create a Guardrails validator
regex_validator = RegexMatch(regex=r"\\d{3}-\\d{3}-\\d{4}")

# Create a Sifaka adapter using the Guardrails validator
phone_adapter = create_guardrails_adapter(
    guardrails_validator=regex_validator,
    name="Phone Number Format",
    description="Validates that text contains a properly formatted phone number"
)

# Use the adapter for validation
result = phone_adapter.validate("My phone number is 555-123-4567") if phone_adapter else ""
```

## Error Handling
- ImportError: Raised when Guardrails is not installed
- ValidationError: Raised when validation fails
- TypeError: Raised when input types are incompatible
- AdapterError: Raised for adapter-specific errors

## State Management
The module uses a standardized state management approach:
- Single _state_manager attribute for all mutable state
- State initialization during construction
- State access through state object
- Clear separation of configuration and state
- Execution tracking for monitoring and debugging

## Configuration
- guardrails_validator: The Guardrails validator to use
- name: Human-readable name of the adapter
- description: Description of the adapter's purpose
"""

import time
from typing import (
    Any,
    Dict,
    Optional,
    Protocol,
    runtime_checkable,
    List,
    Type,
    cast,
    Union,
    TypeVar,
    TypeAlias,
)
from pydantic import PrivateAttr, BaseModel, Field

# Define type variables for better type hinting
T = TypeVar("T")


# Define base types for Guardrails functionality
class GuardrailsValidatorBase:
    """Placeholder for GuardrailsValidator when the package is not available."""

    def validate(self, value: str, metadata: Optional[Dict[str, Any]] = None) -> Any:
        """Placeholder validate method."""
        return None

    @property
    def name(self) -> str:
        """Placeholder name property."""
        return "PlaceholderValidator"

    @property
    def description(self) -> str:
        """Placeholder description property."""
        return "Placeholder description"

    def get_class(self) -> type:
        """Get the class of this validator."""
        return type(self)


class ValidationResultBase:
    """Placeholder for ValidationResult when the package is not available."""

    message: str = "Placeholder validation result"


class PassResultBase(ValidationResultBase):
    """Placeholder for PassResult when the package is not available."""

    pass


class FailResultBase(ValidationResultBase):
    """Placeholder for FailResult when the package is not available."""

    pass


# Flag to track if guardrails is available
GUARDRAILS_AVAILABLE = False

# Define type aliases regardless of whether guardrails is available
GuardrailsValidator: TypeAlias = Type["GuardrailsValidatorBase"]
ValidationResult: TypeAlias = "ValidationResultBase"
PassResult: TypeAlias = "PassResultBase"
FailResult: TypeAlias = "FailResultBase"

# mypy: disable-error-code="import-untyped"
# Try to import guardrails
try:
    from guardrails.validator_base import Validator
    from guardrails.classes import ValidationResult as GRValidationResult
    from guardrails.classes import PassResult as GRPassResult
    from guardrails.classes import FailResult as GRFailResult

    # Update type aliases using the imported types
    GuardrailsValidator = Type[Validator]  # type: ignore
    ValidationResult = GRValidationResult  # type: ignore
    PassResult = GRPassResult  # type: ignore
    FailResult = GRFailResult  # type: ignore
    GUARDRAILS_AVAILABLE = True
except ImportError:
    # We already defined fallback type aliases above
    pass


from sifaka.core.base import BaseComponent, BaseConfig, BaseResult, ComponentResultEnum, Validatable
from sifaka.rules.base import BaseValidator, RuleResult, Rule, RuleConfig
from sifaka.utils.errors.base import ConfigurationError, ValidationError
from sifaka.adapters.base import BaseAdapter, AdapterError, Adaptable
from sifaka.utils.state import StateManager, create_adapter_state
from sifaka.utils.errors.handling import handle_error
from sifaka.utils.logging import get_logger

logger = get_logger(__name__)


# Define a custom state class to help with type checking
class GuardrailsState:
    """State object for Guardrails adapters and rules."""

    adaptee: GuardrailsValidatorBase
    initialized: bool = False
    execution_count: int = 0
    error_count: int = 0
    last_execution_time: Optional[float] = None
    avg_execution_time: float = 0.0
    cache: Dict[str, Any] = {}
    config_cache: Dict[str, Any] = {}


# Helper function to convert Guardrails result to Sifaka RuleResult
def convert_guardrails_result(gr_result: Any) -> RuleResult:
    """
    Convert a Guardrails validation result to a Sifaka rule result.

    Args:
        gr_result: The Guardrails validation result

    Returns:
        RuleResult: The converted Sifaka rule result
    """
    # Use a safer approach to check for pass/fail status that avoids isinstance with Any
    is_pass_result = False
    try:
        # Check class name instead of using isinstance
        result_class_name = gr_result.__class__.__name__ if hasattr(gr_result, "__class__") else ""
        is_pass_result = result_class_name == "PassResult" or result_class_name.endswith(
            "PassResult"
        )
    except (AttributeError, TypeError):
        pass

    if is_pass_result:
        return RuleResult(
            passed=True,
            message="Validation passed",
            metadata={"guardrails_result": str(gr_result)},
        )
    else:
        message = (
            getattr(gr_result, "message", "Validation failed")
            if hasattr(gr_result, "message")
            else "Validation failed"
        )
        return RuleResult(
            passed=False,
            message=str(message),
            metadata={"guardrails_result": str(gr_result)},
        )


@runtime_checkable
class GuardrailsValidatable(Protocol):
    """
    Protocol for components that can be validated by Guardrails.

    ## Overview
    This protocol defines the interface that Guardrails validators must implement
    to be compatible with Sifaka's adapter system.

    ## Architecture
    The protocol requires a validate method that takes a string value and optional
    metadata, and returns a ValidationResult.

    ## Lifecycle
    1. Implementation: Component implements the required methods
    2. Adaptation: Component is adapted using a compatible adapter
    3. Usage: Adapted component is used for validation

    ## Error Handling
    - ValueError: Raised if input is invalid
    - RuntimeError: Raised if validation fails
    """

    def validate(self, value: str, metadata: Optional[Dict[str, Any]] = None) -> Any:
        """
        Validate a value.

        Args:
            value: The string value to validate
            metadata: Optional metadata for validation

        Returns:
            ValidationResult: The result of validation

        Raises:
            ValueError: If input is invalid
            RuntimeError: If validation fails
        """
        ...

    @property
    def name(self) -> str:
        """
        Get the validator name.

        Returns:
            str: The name of the validator
        """
        ...


# Make GuardrailsValidatorBaseAdapter inherit from the Adaptable protocol to fix type compatibility
class GuardrailsValidatorBaseAdapter(GuardrailsValidatorBase, Adaptable):
    """Extended GuardrailsValidatorBase that implements the Adaptable protocol."""

    # Implementation of the custom description
    _custom_description: str = "Guardrails validator"

    def get_description(self) -> str:
        """Get the description of the validator."""
        return self._custom_description

    def set_description(self, value: str) -> None:
        """Set the description of the validator."""
        self._custom_description = value


class GuardrailsAdapter(BaseAdapter[str, GuardrailsValidatorBaseAdapter]):
    """
    Adapter for using Guardrails validators as Sifaka validators.

    ## Overview
    This adapter converts Guardrails validators into a format compatible
    with Sifaka's validation system, handling the conversion between
    Guardrails' validation results and Sifaka's RuleResult format.

    ## Architecture
    The adapter follows a standard pattern:
    1. Receives a Guardrails validator
    2. Translates between Guardrails and Sifaka interfaces
    3. Provides standardized validation results

    ## Lifecycle
    1. Initialization: Set up with validator and configuration
    2. Validation: Convert validator's functionality to validation
    3. Result Handling: Standardize validation results

    ## State Management
    The class uses a standardized state management approach:
    - Single _state_manager attribute for all mutable state
    - State initialization during construction
    - State access through state object
    - Clear separation of configuration and state
    - State components:
      - adaptee: The Guardrails validator being adapted
      - execution_count: Number of validation executions
      - last_execution_time: Timestamp of last execution
      - avg_execution_time: Average execution time
      - error_count: Number of validation errors
      - cache: Temporary data storage for validation results

    ## Error Handling
    - ImportError: Raised when Guardrails is not installed
    - ValidationError: Raised when validation fails
    - TypeError: Raised when input types are incompatible
    - AdapterError: Raised for adapter-specific errors

    ## Examples
    ```python
    from guardrails.hub import RegexMatch
    from sifaka.adapters.guardrails import GuardrailsAdapter

    # Create a Guardrails validator
    regex_validator = RegexMatch(regex=r"\\d{3}-\\d{3}-\\d{4}")

    # Create an adapter
    adapter = GuardrailsAdapter(regex_validator)

    # Validate text
    result = adapter.validate("My phone number is 555-123-4567") if adapter else ""
    ```
    """

    def __init__(self, guardrails_validator: GuardrailsValidatorBaseAdapter) -> None:
        """
        Initialize the adapter with a Guardrails validator.

        Args:
            guardrails_validator: The Guardrails validator to adapt

        Raises:
            ImportError: If Guardrails is not installed
            ConfigurationError: If validator is invalid
            AdapterError: If initialization fails
        """
        try:
            if not GUARDRAILS_AVAILABLE:
                raise ImportError(
                    "Guardrails is not installed. Please install it with 'pip install guardrails-ai'"
                )
            super().__init__(guardrails_validator)
            self._state_manager.set_metadata("adapter_type", "guardrails")
            self._state_manager.set_metadata("validator_type", type(guardrails_validator).__name__)
            logger.debug(f"Initialized GuardrailsAdapter for {type(guardrails_validator).__name__}")
        except Exception as e:
            if isinstance(e, ImportError):
                raise
            error_info = handle_error(e, f"GuardrailsAdapter:{type(guardrails_validator).__name__}")
            raise AdapterError(
                f"Failed to initialize GuardrailsAdapter: {str(e)}", metadata=error_info
            ) from e

    def _convert_guardrails_result(self, gr_result: Any) -> RuleResult:
        """
        Convert a Guardrails validation result to a Sifaka rule result.

        Args:
            gr_result: The Guardrails validation result

        Returns:
            RuleResult: The converted Sifaka rule result
        """
        return convert_guardrails_result(gr_result)

    def _validate_impl(self, input_value: str, **kwargs) -> RuleResult:
        """
        Implementation of validation logic for the Guardrails adapter.

        This method is called by the base adapter's validate method.

        Args:
            input_value: The text to validate
            **kwargs: Additional validation parameters

        Returns:
            RuleResult containing validation outcome and metadata

        Raises:
            ValidationError: If validation fails due to an error
            AdapterError: If adapter-specific error occurs
        """
        try:
            if not input_value:
                return RuleResult(
                    passed=False,
                    message="Empty text is not allowed",
                    metadata={"error": "empty_text"},
                )
            gr_result = self.adaptee.validate(input_value, metadata=kwargs)
            return self._convert_guardrails_result(gr_result)
        except Exception as e:
            if isinstance(e, (ValidationError, AdapterError)):
                raise
            error_info = handle_error(e, f"GuardrailsAdapter:{type(self.adaptee).__name__}")
            raise ValidationError(
                f"Guardrails validation failed: {str(e)}", metadata=error_info
            ) from e

    def get_detailed_statistics(self) -> Dict[str, Any]:
        """
        Get detailed statistics about adapter usage.

        Returns:
            Dict[str, Any]: Dictionary with detailed usage statistics
        """
        stats = self.get_statistics()
        stats.update(
            {
                "validator_type": self._state_manager.get_metadata("validator_type", "unknown"),
                "validator_class": type(self.adaptee).__name__,
            }
        )
        return stats


class GuardrailsRule(Rule[str]):
    """
    Rule that uses a Guardrails validator for validation.

    This rule wraps a Guardrails validator and provides a Sifaka-compatible
    interface for validation.

    ## Overview
    This rule enables the use of Guardrails validators within Sifaka's rule system,
    allowing for sophisticated content validation using Guardrails' capabilities.

    ## Architecture
    The rule follows a standard pattern:
    1. Wraps a Guardrails validator
    2. Translates between Guardrails and Sifaka interfaces
    3. Provides standardized validation results

    ## Lifecycle
    1. Initialization: Set up with validator and configuration
    2. Validation: Convert validator's functionality to validation
    3. Result Handling: Standardize validation results

    ## State Management
    The class uses a standardized state management approach:
    - Single _state_manager attribute for all mutable state
    - State initialization during construction
    - State access through state object
    - Clear separation of configuration and state
    """

    _state_manager = PrivateAttr(default_factory=create_adapter_state)
    _guardrails_validator: GuardrailsValidatorBase
    _state: GuardrailsState

    def __init__(
        self,
        guardrails_validator: GuardrailsValidatorBase,
        rule_id: Optional[str] = None,
        name: str = "Guardrails Rule",
        description: str = "Rule using Guardrails validator",
        config: Optional[RuleConfig] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize with a Guardrails validator.

        Args:
            guardrails_validator: The Guardrails validator to use
            rule_id: Optional unique identifier for the rule
            name: Human-readable name for the rule
            description: Description of the rule's purpose
            config: Optional additional configuration
            **kwargs: Additional keyword arguments for configuration including:
                - severity: Severity level for rule violations
                - category: Category of the rule
                - tags: List of tags for categorizing the rule
                - priority: Priority level for validation
                - cache_size: Size of the validation cache
                - cost: Computational cost of validation

        Raises:
            ImportError: If Guardrails is not installed
            ConfigurationError: If validator is invalid
            ValueError: If required parameters are missing
        """
        try:
            if not GUARDRAILS_AVAILABLE:
                raise ImportError(
                    "Guardrails is not installed. Please install it with 'pip install guardrails-ai'"
                )
            super().__init__(name, description, config)
            self._guardrails_validator = guardrails_validator
        except Exception as e:
            if isinstance(e, ImportError):
                raise
            error_info = handle_error(e, f"GuardrailsRule:{type(guardrails_validator).__name__}")
            raise ConfigurationError(
                f"Failed to initialize GuardrailsRule: {str(e)}", metadata=error_info
            ) from e

        # Initialize state
        self._state = GuardrailsState()
        self._state.adaptee = guardrails_validator
        self._state.initialized = True
        self._state.execution_count = 0
        self._state.error_count = 0
        self._state.last_execution_time = None
        self._state.avg_execution_time = 0.0
        self._state.cache = {}
        self._state.config_cache = {"config": kwargs}

        # Set metadata
        self._state_manager.set_metadata("component_type", "rule")
        self._state_manager.set_metadata("creation_time", time.time())
        self._state_manager.set_metadata("rule_id", rule_id or "guardrails_rule")
        self._state_manager.set_metadata("validator_type", type(guardrails_validator).__name__)

    def _convert_guardrails_result(self, gr_result: Any) -> RuleResult:
        """
        Convert a Guardrails validation result to a Sifaka rule result.

        Args:
            gr_result: The Guardrails validation result

        Returns:
            RuleResult: The converted Sifaka rule result
        """
        return convert_guardrails_result(gr_result)

    def validate(self, text: str, **kwargs) -> RuleResult:
        """
        Validate text using the Guardrails validator.

        Args:
            text: The text to validate
            **kwargs: Additional validation parameters

        Returns:
            RuleResult: The validation result

        Raises:
            ValidationError: If validation fails due to an error
        """
        state = self._state
        state.execution_count += 1
        start_time = time.time()

        try:
            cache_key = self._get_cache_key(text, kwargs)
            if cache_key and cache_key in state.cache:
                logger.debug("Cache hit for guardrails rule")
                cached_result = state.cache[cache_key]
                # Ensure we're returning a RuleResult
                return cast(RuleResult, cached_result)

            if not text:
                result = RuleResult(
                    passed=False,
                    message="Empty text is not allowed",
                    metadata={"error": "empty_text"},
                )
            else:
                gr_result = state.adaptee.validate(text, metadata=kwargs)
                result = self._convert_guardrails_result(gr_result)

            if cache_key:
                state.cache[cache_key] = result

            return result

        except Exception as e:
            state.error_count += 1
            logger.error(f"Validation error: {str(e)}")

            # Create a RuleResult for the error
            error_result = RuleResult(
                passed=False,
                message=f"Guardrails validation failed: {str(e)}",
                metadata={"error": str(e), "error_type": type(e).__name__},
                score=0.0,
                issues=[f"Validation error: {str(e)}"],
                suggestions=["Check your input and try again"],
                processing_time_ms=(time.time() - start_time) * 1000,
            )

            return error_result

        finally:
            execution_time = time.time() - start_time
            state.last_execution_time = execution_time
            if state.execution_count > 1:
                state.avg_execution_time = (
                    state.avg_execution_time * (state.execution_count - 1) + execution_time
                ) / state.execution_count
            else:
                state.avg_execution_time = execution_time

    def _get_cache_key(self, input_value: str, kwargs: Dict[str, Any]) -> Optional[str]:
        """
        Generate a cache key for the input value and kwargs.

        Args:
            input_value: The input text
            kwargs: Additional parameters

        Returns:
            Optional[str]: Cache key or None if caching is disabled
        """
        return f"{hash(input_value)}:{hash(str(kwargs))}"

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about rule usage.

        Returns:
            Dict[str, Any]: Dictionary with usage statistics
        """
        state = self._state
        return {
            "execution_count": state.execution_count,
            "error_count": state.error_count,
            "avg_execution_time": state.avg_execution_time,
            "last_execution_time": state.last_execution_time,
            "cache_size": len(state.cache),
            "rule_id": self._state_manager.get_metadata("rule_id", "unknown"),
            "validator_type": self._state_manager.get_metadata("validator_type", "unknown"),
        }

    def clear_cache(self) -> None:
        """Clear the rule result cache."""
        self._state.cache = {}
        logger.debug("Rule cache cleared")

    def _create_default_validator(self) -> BaseValidator[str]:
        """
        Create a default validator for this rule.

        This is an implementation of the abstract method required by the Rule base class.

        Returns:
            BaseValidator[str]: A validator that uses the Guardrails validator
        """
        return GuardrailsValidatorAdapter(self)


# Define the validator adapter as a module-level class instead of a nested class
class GuardrailsValidatorAdapter(BaseValidator[str]):
    """
    Adapter for using Guardrails validators within the Sifaka validation system.

    This class allows Guardrails Rule objects to be used as validators in other contexts.
    """

    def __init__(self, guardrails_rule: GuardrailsRule):
        """Initialize with a GuardrailsRule instance"""
        super().__init__(validation_type=str)
        self.guardrails_rule = guardrails_rule

    def validate(self, text: str) -> RuleResult:
        """Delegate validation to the underlying GuardrailsRule"""
        return self.guardrails_rule.validate(text)


def create_guardrails_adapter(
    guardrails_validator: GuardrailsValidatorBase,
    name: Optional[str] = None,
    description: Optional[str] = None,
    initialize: bool = True,
    **kwargs: Any,
) -> GuardrailsAdapter:
    """
    Factory function to create a GuardrailsAdapter.

    ## Overview
    This function simplifies the creation of GuardrailsAdapter instances by providing a
    consistent interface with standardized configuration options.

    ## Architecture
    The factory function follows a standard pattern:
    1. Validate inputs
    2. Create adapter instance
    3. Initialize if requested
    4. Return configured instance

    Args:
        guardrails_validator: The Guardrails validator to adapt
        name: Optional name for the adapter
        description: Optional description for the adapter
        initialize: Whether to initialize the adapter immediately
        **kwargs: Additional keyword arguments

    Returns:
        GuardrailsAdapter: A configured adapter instance

    Raises:
        ImportError: If Guardrails is not installed
        ConfigurationError: If validator is invalid
        AdapterError: If initialization fails

    ## Examples
    ```python
    from guardrails.hub import RegexMatch
    from sifaka.adapters.guardrails import create_guardrails_adapter

    # Create a Guardrails validator
    regex_validator = RegexMatch(regex=r"\\d{3}-\\d{3}-\\d{4}")

    # Create an adapter
    adapter = create_guardrails_adapter(
        guardrails_validator=regex_validator,
        name="Phone Number Validator",
        description="Validates phone numbers in the format XXX-XXX-XXXX"
    )
    ```
    """
    try:
        if not GUARDRAILS_AVAILABLE:
            raise ImportError(
                "Guardrails is not installed. Please install it with 'pip install guardrails-ai'"
            )
        # Cast the validator to our adapter class that implements Adaptable
        adapted_validator = GuardrailsValidatorBaseAdapter()

        # Copy attributes from the original validator to the adapter
        for attr_name in dir(guardrails_validator):
            if not attr_name.startswith("__") and attr_name != "description":
                try:
                    setattr(adapted_validator, attr_name, getattr(guardrails_validator, attr_name))
                except (AttributeError, TypeError):
                    pass  # Skip attributes that can't be set

        # Set the description property
        if description:
            adapted_validator.set_description(description)
        elif hasattr(guardrails_validator, "description"):
            adapted_validator.set_description(getattr(guardrails_validator, "description"))

        adapter = GuardrailsAdapter(adapted_validator)
        if name:
            adapter._state_manager.set_metadata("name", name)
        if description:
            adapter._state_manager.set_metadata("description", description)
        if initialize:
            adapter.warm_up()
        logger.debug(f"Created GuardrailsAdapter for {type(guardrails_validator).__name__}")
        return adapter
    except Exception as e:
        if isinstance(e, (ImportError, AdapterError)):
            raise
        error_info = handle_error(
            e, f"GuardrailsAdapterFactory:{type(guardrails_validator).__name__}"
        )
        raise AdapterError(
            f"Failed to create GuardrailsAdapter: {str(e)}", metadata=error_info
        ) from e


def create_guardrails_rule(
    guardrails_validator: GuardrailsValidatorBase,
    rule_id: Optional[str] = None,
    name: str = "Guardrails Rule",
    description: str = "Rule using Guardrails validator",
    config: Optional[RuleConfig] = None,
    **kwargs,
) -> GuardrailsRule:
    """
    Create a GuardrailsRule instance.

    ## Overview
    This function creates a rule that uses a Guardrails validator for validation.

    Args:
        guardrails_validator: The Guardrails validator to use
        rule_id: Optional unique identifier for the rule
        name: Human-readable name for the rule
        description: Description of the rule's purpose
        config: Optional additional configuration
        **kwargs: Additional keyword arguments

    Returns:
        GuardrailsRule: A configured rule instance

    Raises:
        ImportError: If Guardrails is not installed
    """
    return GuardrailsRule(
        guardrails_validator=guardrails_validator,
        rule_id=rule_id,
        name=name,
        description=description,
        config=config,
        **kwargs,
    )
