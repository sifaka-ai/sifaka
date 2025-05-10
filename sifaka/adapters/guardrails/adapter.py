"""
Guardrails Adapter

Adapter for using Guardrails validators as rules in the Sifaka framework.

## Overview
This module provides adapters for using validators from the Guardrails library as Sifaka rules.
It enables seamless integration between Guardrails' validation capabilities and Sifaka's
rule system, allowing for sophisticated content validation.

## Components
1. **GuardrailsValidatable Protocol**: Defines the expected interface for Guardrails validators
2. **GuardrailsValidatorAdapter**: Adapts Guardrails validators to work as Sifaka validators
3. **GuardrailsRule**: Rule that uses a Guardrails validator for validation
4. **Factory Functions**: Simple creation patterns for Guardrails-based rules

## Usage Examples
```python
from guardrails.hub import RegexMatch
from sifaka.adapters.guardrails import create_guardrails_rule

# Create a Guardrails validator
regex_validator = RegexMatch(regex=r"\\d{3}-\\d{3}-\\d{4}")

# Create a Sifaka rule using the Guardrails validator
phone_rule = create_guardrails_rule(
    guardrails_validator=regex_validator,
    rule_id="phone_number_format",
    name="Phone Number Format",
    description="Validates that text contains a properly formatted phone number"
)

# Use it in a Sifaka chain
result = chain.run("What's a good phone number format?")
```

## Error Handling
- ImportError: Raised when Guardrails is not installed
- ValidationError: Raised when validation fails
- TypeError: Raised when input types are incompatible

## Configuration
- guardrails_validator: The Guardrails validator to use
- rule_id: Unique identifier for the rule
- name: Human-readable name of the rule
- description: Description of the rule's purpose
"""

from typing import Any, Dict, Optional, Union, Protocol, runtime_checkable
import time

from pydantic import PrivateAttr

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
from sifaka.rules.base import BaseValidator, RuleResult, Rule, RuleConfig
from sifaka.adapters.base import Adaptable, BaseAdapter
from sifaka.utils.state import StateManager
from sifaka.utils.logging import get_logger

logger = get_logger(__name__)


@runtime_checkable
class GuardrailsValidatable(Protocol):
    """Protocol for components that can be validated by Guardrails."""

    def validate(self, value: str, metadata: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """Validate a value."""
        ...


class GuardrailsValidatorAdapter(BaseValidator[str], BaseComponent):
    """
    Adapter for using Guardrails validators as Sifaka validators.

    This adapter converts Guardrails validators into a format compatible
    with Sifaka's validation system, handling the conversion between
    Guardrails' validation results and Sifaka's RuleResult format.
    """

    # State management
    _state = PrivateAttr(default_factory=StateManager)

    def __init__(
        self,
        guardrails_validator: GuardrailsValidator,
        name: str = "guardrails_validator",
        description: str = "Adapter for Guardrails validator",
        config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize with a Guardrails validator."""
        super().__init__()

        if not GUARDRAILS_AVAILABLE:
            raise ImportError(
                "Guardrails is not installed. Please install it with 'pip install guardrails-ai'"
            )

        self._state.update("validator", guardrails_validator)
        self._state.update("name", name)
        self._state.update("description", description)
        self._state.update("config", config or {})
        self._state.update("initialized", True)
        self._state.update("execution_count", 0)
        self._state.update("result_cache", {})

        # Set metadata
        self._state.set_metadata("component_type", "validator")
        self._state.set_metadata("creation_time", time.time())

    def _convert_guardrails_result(self, gr_result: ValidationResult) -> RuleResult:
        """Convert a Guardrails validation result to a Sifaka rule result."""
        if isinstance(gr_result, PassResult):
            return RuleResult(
                passed=True, message="Validation passed", details={"guardrails_result": gr_result}
            )
        else:
            return RuleResult(
                passed=False,
                message=(
                    str(gr_result.message) if hasattr(gr_result, "message") else "Validation failed"
                ),
                details={"guardrails_result": gr_result},
            )

    def validate(self, output: str, **kwargs) -> RuleResult:
        """Validate text using the Guardrails validator."""
        # Track execution count
        execution_count = self._state.get("execution_count", 0)
        self._state.update("execution_count", execution_count + 1)

        # Check cache
        cache = self._state.get("result_cache", {})
        if output in cache:
            self._state.set_metadata("cache_hit", True)
            return cache[output]

        # Mark as cache miss
        self._state.set_metadata("cache_hit", False)

        # Record start time
        start_time = time.time()

        try:
            # Get validator from state
            validator = self._state.get("validator")

            # Handle empty text
            if not output:
                result = RuleResult(
                    passed=False,
                    message="Empty text is not allowed",
                    details={"error": "empty_text"},
                )
            else:
                # Run validation
                gr_result = validator.validate(output, metadata=kwargs)
                result = self._convert_guardrails_result(gr_result)

            # Record execution time
            end_time = time.time()
            exec_time = end_time - start_time

            # Update average execution time
            avg_time = self._state.get_metadata("avg_execution_time", 0)
            count = self._state.get("execution_count", 1)
            new_avg = ((avg_time * (count - 1)) + exec_time) / count
            self._state.set_metadata("avg_execution_time", new_avg)

            # Update max execution time if needed
            max_time = self._state.get_metadata("max_execution_time", 0)
            if exec_time > max_time:
                self._state.set_metadata("max_execution_time", exec_time)

            # Cache result
            cache[output] = result
            self._state.update("result_cache", cache)

            return result

        except Exception as e:
            # Track error
            error_count = self._state.get_metadata("error_count", 0)
            self._state.set_metadata("error_count", error_count + 1)
            logger.error(f"Validation error: {str(e)}")
            raise

    @property
    def validation_type(self) -> type:
        """Get the type of values this validator can validate."""
        return str

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about validator usage."""
        return {
            "execution_count": self._state.get("execution_count", 0),
            "cache_size": len(self._state.get("result_cache", {})),
            "avg_execution_time": self._state.get_metadata("avg_execution_time", 0),
            "max_execution_time": self._state.get_metadata("max_execution_time", 0),
            "error_count": self._state.get_metadata("error_count", 0),
        }

    def clear_cache(self) -> None:
        """Clear the validator result cache."""
        self._state.update("result_cache", {})
        logger.debug("Validator cache cleared")


class GuardrailsRule(Rule, BaseComponent):
    """
    Rule that uses a Guardrails validator for validation.

    This rule wraps a Guardrails validator and provides a Sifaka-compatible
    interface for validation.
    """

    # State management
    _state = PrivateAttr(default_factory=StateManager)

    def __init__(
        self,
        guardrails_validator: GuardrailsValidator,
        rule_id: Optional[str] = None,
        name: Optional[str] = None,
        description: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        """Initialize with a Guardrails validator."""
        super().__init__()

        if not GUARDRAILS_AVAILABLE:
            raise ImportError(
                "Guardrails is not installed. Please install it with 'pip install guardrails-ai'"
            )

        self._state.update("validator", guardrails_validator)
        self._state.update("rule_id", rule_id or "guardrails_rule")
        self._state.update("name", name or "Guardrails Rule")
        self._state.update("description", description or "Rule using Guardrails validator")
        self._state.update("config", config or {})
        self._state.update("initialized", True)
        self._state.update("execution_count", 0)
        self._state.update("result_cache", {})

        # Set metadata
        self._state.set_metadata("component_type", "rule")
        self._state.set_metadata("creation_time", time.time())

    def validate(self, text: str, **kwargs) -> RuleResult:
        """Validate text using the Guardrails validator."""
        # Track execution count
        execution_count = self._state.get("execution_count", 0)
        self._state.update("execution_count", execution_count + 1)

        # Check cache
        cache = self._state.get("result_cache", {})
        if text in cache:
            self._state.set_metadata("cache_hit", True)
            return cache[text]

        # Mark as cache miss
        self._state.set_metadata("cache_hit", False)

        # Record start time
        start_time = time.time()

        try:
            # Get validator from state
            validator = self._state.get("validator")

            # Handle empty text
            if not text:
                result = RuleResult(
                    passed=False,
                    message="Empty text is not allowed",
                    details={"error": "empty_text"},
                )
            else:
                # Run validation
                gr_result = validator.validate(text, metadata=kwargs)
                result = self._convert_guardrails_result(gr_result)

            # Record execution time
            end_time = time.time()
            exec_time = end_time - start_time

            # Update average execution time
            avg_time = self._state.get_metadata("avg_execution_time", 0)
            count = self._state.get("execution_count", 1)
            new_avg = ((avg_time * (count - 1)) + exec_time) / count
            self._state.set_metadata("avg_execution_time", new_avg)

            # Update max execution time if needed
            max_time = self._state.get_metadata("max_execution_time", 0)
            if exec_time > max_time:
                self._state.set_metadata("max_execution_time", exec_time)

            # Cache result
            cache[text] = result
            self._state.update("result_cache", cache)

            return result

        except Exception as e:
            # Track error
            error_count = self._state.get_metadata("error_count", 0)
            self._state.set_metadata("error_count", error_count + 1)
            logger.error(f"Validation error: {str(e)}")
            raise

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about rule usage."""
        return {
            "execution_count": self._state.get("execution_count", 0),
            "cache_size": len(self._state.get("result_cache", {})),
            "avg_execution_time": self._state.get_metadata("avg_execution_time", 0),
            "max_execution_time": self._state.get_metadata("max_execution_time", 0),
            "error_count": self._state.get_metadata("error_count", 0),
        }

    def clear_cache(self) -> None:
        """Clear the rule result cache."""
        self._state.update("result_cache", {})
        logger.debug("Rule cache cleared")


def create_guardrails_rule(
    guardrails_validator: GuardrailsValidator,
    rule_id: Optional[str] = None,
    name: Optional[str] = None,
    description: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> GuardrailsRule:
    """Create a GuardrailsRule instance."""
    return GuardrailsRule(
        guardrails_validator=guardrails_validator,
        rule_id=rule_id,
        name=name,
        description=description,
        config=config,
        **kwargs,
    )
