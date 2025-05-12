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
result = phone_adapter.validate("My phone number is 555-123-4567")
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
from typing import Any, Dict, Optional, Protocol, runtime_checkable, List, Type

from pydantic import PrivateAttr, BaseModel, Field

try:
    from guardrails.validator_base import Validator as GuardrailsValidator
    from guardrails.classes import ValidationResult, PassResult, FailResult

    GUARDRAILS_AVAILABLE = True
except ImportError:
    GUARDRAILS_AVAILABLE = False

    # Create placeholder classes for type hints when Guardrails isn't installed
    class GuardrailsValidator:
        pass

    class ValidationResult:
        pass

    class PassResult(ValidationResult):
        pass

    class FailResult(ValidationResult):
        pass


from sifaka.core.base import BaseComponent, BaseConfig, BaseResult, ComponentResultEnum, Validatable
from sifaka.rules.base import (
    BaseValidator,
    RuleResult,
    Rule,
    RuleConfig,
)
from sifaka.utils.errors.base import ConfigurationError, ValidationError
from sifaka.adapters.base import BaseAdapter, AdapterError
from sifaka.utils.state import StateManager, create_adapter_state
from sifaka.utils.errors.handling import handle_error
from sifaka.utils.logging import get_logger

logger = get_logger(__name__)


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

    def validate(self, value: str, metadata: Optional[Dict[str, Any]] = None) -> ValidationResult:
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


class GuardrailsAdapter(BaseAdapter[str, GuardrailsValidator]):
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
    result = adapter.validate("My phone number is 555-123-4567")
    ```
    """

    def __init__(self, guardrails_validator: GuardrailsValidator) -> None:
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
            # Check if Guardrails is available
            if not GUARDRAILS_AVAILABLE:
                raise ImportError(
                    "Guardrails is not installed. Please install it with 'pip install guardrails-ai'"
                )

            # Initialize base adapter
            super().__init__(guardrails_validator)

            # Set additional metadata
            self._state_manager.set_metadata("adapter_type", "guardrails")
            self._state_manager.set_metadata(
                "validator_type", guardrails_validator.__class__.__name__
            )

            logger.debug(
                f"Initialized GuardrailsAdapter for {guardrails_validator.__class__.__name__}"
            )
        except Exception as e:
            if isinstance(e, ImportError):
                raise

            error_info = handle_error(
                e, f"GuardrailsAdapter:{guardrails_validator.__class__.__name__}"
            )
            raise AdapterError(
                f"Failed to initialize GuardrailsAdapter: {str(e)}", metadata=error_info
            ) from e

    def _convert_guardrails_result(self, gr_result: ValidationResult) -> RuleResult:
        """
        Convert a Guardrails validation result to a Sifaka rule result.

        Args:
            gr_result: The Guardrails validation result

        Returns:
            RuleResult: The converted Sifaka rule result
        """
        if isinstance(gr_result, PassResult):
            return RuleResult(
                passed=True,
                message="Validation passed",
                metadata={"guardrails_result": str(gr_result)},
            )
        else:
            return RuleResult(
                passed=False,
                message=(
                    str(gr_result.message) if hasattr(gr_result, "message") else "Validation failed"
                ),
                metadata={"guardrails_result": str(gr_result)},
            )

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
            # Handle empty text
            if not input_value:
                return RuleResult(
                    passed=False,
                    message="Empty text is not allowed",
                    metadata={"error": "empty_text"},
                )

            # Run validation
            gr_result = self.adaptee.validate(input_value, metadata=kwargs)

            # Convert result
            return self._convert_guardrails_result(gr_result)
        except Exception as e:
            if isinstance(e, (ValidationError, AdapterError)):
                raise

            error_info = handle_error(e, f"GuardrailsAdapter:{self.adaptee.__class__.__name__}")
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

        # Add guardrails-specific statistics
        stats.update(
            {
                "validator_type": self._state_manager.get_metadata("validator_type", "unknown"),
                "validator_class": self.adaptee.__class__.__name__,
            }
        )

        return stats


class GuardrailsRule(Rule, BaseComponent):
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

    # State management using standardized state manager
    _state_manager = PrivateAttr(default_factory=create_adapter_state)

    def __init__(
        self,
        guardrails_validator: GuardrailsValidator,
        rule_id: Optional[str] = None,
        name: Optional[str] = None,
        description: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ):
        """
        Initialize with a Guardrails validator.

        Args:
            guardrails_validator: The Guardrails validator to use
            rule_id: Optional unique identifier for the rule
            name: Optional human-readable name for the rule
            description: Optional description of the rule's purpose
            config: Optional additional configuration
            **kwargs: Additional keyword arguments

        Raises:
            ImportError: If Guardrails is not installed
        """
        super().__init__()

        if not GUARDRAILS_AVAILABLE:
            raise ImportError(
                "Guardrails is not installed. Please install it with 'pip install guardrails-ai'"
            )

        # Initialize state
        state = self._state_manager.get_state()
        state.adaptee = guardrails_validator
        state.initialized = True
        state.execution_count = 0
        state.error_count = 0
        state.last_execution_time = None
        state.avg_execution_time = 0
        state.cache = {}
        state.config_cache = {"config": config or {}}

        # Set metadata
        self._state_manager.set_metadata("component_type", "rule")
        self._state_manager.set_metadata("creation_time", time.time())
        self._state_manager.set_metadata("rule_id", rule_id or "guardrails_rule")
        self._state_manager.set_metadata("name", name or "Guardrails Rule")
        self._state_manager.set_metadata(
            "description", description or "Rule using Guardrails validator"
        )
        self._state_manager.set_metadata("validator_type", guardrails_validator.__class__.__name__)

    def _convert_guardrails_result(self, gr_result: ValidationResult) -> RuleResult:
        """
        Convert a Guardrails validation result to a Sifaka rule result.

        Args:
            gr_result: The Guardrails validation result

        Returns:
            RuleResult: The converted Sifaka rule result
        """
        if isinstance(gr_result, PassResult):
            return RuleResult(
                passed=True,
                message="Validation passed",
                metadata={"guardrails_result": str(gr_result)},
            )
        else:
            return RuleResult(
                passed=False,
                message=(
                    str(gr_result.message) if hasattr(gr_result, "message") else "Validation failed"
                ),
                metadata={"guardrails_result": str(gr_result)},
            )

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
        # Get state
        state = self._state_manager.get_state()

        # Track execution
        state.execution_count += 1
        start_time = time.time()

        try:
            # Check cache
            cache_key = self._get_cache_key(text, kwargs)
            if cache_key and cache_key in state.cache:
                logger.debug(f"Cache hit for guardrails rule")
                return state.cache[cache_key]

            # Handle empty text
            if not text:
                result = RuleResult(
                    passed=False,
                    message="Empty text is not allowed",
                    metadata={"error": "empty_text"},
                )
            else:
                # Run validation
                gr_result = state.adaptee.validate(text, metadata=kwargs)
                result = self._convert_guardrails_result(gr_result)

            # Cache result if enabled
            if cache_key:
                state.cache[cache_key] = result

            return result
        except Exception as e:
            # Track error
            state.error_count += 1
            logger.error(f"Validation error: {str(e)}")
            raise ValidationError(f"Guardrails validation failed: {str(e)}")
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

    def _get_cache_key(self, input_value: str, kwargs: Dict[str, Any]) -> Optional[str]:
        """
        Generate a cache key for the input value and kwargs.

        Args:
            input_value: The input text
            kwargs: Additional parameters

        Returns:
            Optional[str]: Cache key or None if caching is disabled
        """
        # Simple string hash for text inputs
        return f"{hash(input_value)}:{hash(str(kwargs))}"

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about rule usage.

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
            "rule_id": self._state_manager.get_metadata("rule_id", "unknown"),
            "validator_type": self._state_manager.get_metadata("validator_type", "unknown"),
        }

    def clear_cache(self) -> None:
        """Clear the rule result cache."""
        self._state_manager.get_state().cache = {}
        logger.debug("Rule cache cleared")


def create_guardrails_adapter(
    guardrails_validator: GuardrailsValidator,
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
        # Check if Guardrails is available
        if not GUARDRAILS_AVAILABLE:
            raise ImportError(
                "Guardrails is not installed. Please install it with 'pip install guardrails-ai'"
            )

        # Create adapter instance
        adapter = GuardrailsAdapter(guardrails_validator)

        # Set name and description if provided
        if name:
            adapter._state_manager.set_metadata("name", name)
        if description:
            adapter._state_manager.set_metadata("description", description)

        # Initialize if requested
        if initialize:
            adapter.warm_up()

        logger.debug(f"Created GuardrailsAdapter for {guardrails_validator.__class__.__name__}")
        return adapter
    except Exception as e:
        # Handle errors
        if isinstance(e, (ImportError, AdapterError)):
            raise

        # Convert other errors to AdapterError
        error_info = handle_error(
            e, f"GuardrailsAdapterFactory:{guardrails_validator.__class__.__name__}"
        )
        raise AdapterError(
            f"Failed to create GuardrailsAdapter: {str(e)}", metadata=error_info
        ) from e


def create_guardrails_rule(
    guardrails_validator: GuardrailsValidator,
    rule_id: Optional[str] = None,
    name: Optional[str] = None,
    description: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> GuardrailsRule:
    """
    Create a GuardrailsRule instance.

    ## Overview
    This function creates a rule that uses a Guardrails validator for validation.

    Args:
        guardrails_validator: The Guardrails validator to use
        rule_id: Optional unique identifier for the rule
        name: Optional human-readable name for the rule
        description: Optional description of the rule's purpose
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
