"""
GuardrailsAI validator for Sifaka.

This module provides a validator that uses GuardrailsAI to validate text.
It adapts GuardrailsAI's Guard to the Sifaka validator interface, allowing
GuardrailsAI validators to be used in Sifaka validation chains.

GuardrailsAI is a framework for validating and structuring data from language models.
For more information, see https://www.guardrailsai.com/docs

Note: This validator requires the GuardrailsAI API key to be set in the
GUARDRAILS_API_KEY environment variable or passed directly to the validator.
"""

import importlib
import logging
import os
from typing import Any, Dict, List, Optional, Union, Callable

from sifaka.results import ValidationResult
from sifaka.errors import ValidationError
from sifaka.registry import register_validator

# Set up logging
logger = logging.getLogger(__name__)


class GuardrailsValidator:
    """
    Validator that uses GuardrailsAI for validation.

    This validator adapts GuardrailsAI's Guard to the Sifaka validator interface,
    allowing GuardrailsAI validators to be used in Sifaka validation chains.

    Attributes:
        guard: The GuardrailsAI Guard instance.
        validators: List of GuardrailsAI validators to use.
        validator_args: Dictionary of arguments for each validator.
        name: The name of the validator.
        description: The description of the validator.
    """

    def __init__(
        self,
        guard=None,
        validators: Optional[List[str]] = None,
        validator_args: Optional[Dict[str, Dict[str, Any]]] = None,
        api_key: Optional[str] = None,
        name: str = "guardrails_validator",
        description: str = "Validates text using GuardrailsAI",
    ):
        """
        Initialize the GuardrailsAI validator.

        Args:
            guard: Optional pre-configured GuardrailsAI Guard instance.
            validators: Optional list of GuardrailsAI validator names to use.
            validator_args: Optional dictionary of arguments for each validator.
            api_key: Optional GuardrailsAI API key. If not provided, will use the key from
                     the GuardrailsAI configuration or the GUARDRAILS_API_KEY environment variable.
            name: The name of the validator.
            description: The description of the validator.

        Raises:
            ImportError: If guardrails-ai is not installed.
            ValidationError: If both guard and validators are provided.
        """
        self._name = name
        self._description = description
        self._guard = None
        self._guardrails = None
        self._initialized = False
        self._api_key = api_key or os.environ.get("GUARDRAILS_API_KEY")

        # Validate inputs
        if guard is not None and validators is not None:
            raise ValidationError(
                "Cannot provide both a guard and validators. Choose one approach."
            )

        # Store configuration for lazy initialization
        self._guard_config = guard
        self._validators_config = validators
        self._validator_args_config = validator_args or {}

    @property
    def name(self) -> str:
        """Get the validator name."""
        return self._name

    @property
    def description(self) -> str:
        """Get the validator description."""
        return self._description

    def _load_guardrails(self):
        """
        Load the guardrails-ai library.

        Returns:
            The guardrails module.

        Raises:
            ImportError: If guardrails-ai is not installed.
        """
        try:
            guardrails = importlib.import_module("guardrails")
            return guardrails
        except ImportError:
            raise ImportError(
                "guardrails-ai package is required for GuardrailsValidator. "
                "Install it with: pip install guardrails-ai"
            )

    def _initialize(self) -> None:
        """
        Initialize the GuardrailsAI validator.

        This method loads the guardrails-ai library and creates a Guard instance
        if one was not provided during initialization.

        Raises:
            ImportError: If guardrails-ai is not installed.
            ValidationError: If initialization fails.
        """
        if self._initialized:
            return

        try:
            # Load guardrails
            self._guardrails = self._load_guardrails()

            # If a guard was provided, use it
            if self._guard_config is not None:
                self._guard = self._guard_config
            # Otherwise, create a new guard with the specified validators
            elif self._validators_config is not None:
                # Create a new guard
                self._guard = self._guardrails.Guard()

                # Install and add each validator
                for validator_name in self._validators_config:
                    # Get arguments for this validator
                    args = self._validator_args_config.get(validator_name, {})

                    # Install the validator
                    # Set API key in environment if provided
                    old_api_key = os.environ.get("GUARDRAILS_API_KEY")
                    if self._api_key:
                        os.environ["GUARDRAILS_API_KEY"] = self._api_key

                    try:
                        validator_module = self._guardrails.install(
                            f"hub://guardrails/{validator_name}"
                        )
                    finally:
                        # Restore original API key
                        if self._api_key:
                            if old_api_key:
                                os.environ["GUARDRAILS_API_KEY"] = old_api_key
                            else:
                                os.environ.pop("GUARDRAILS_API_KEY", None)

                    # Get the validator class
                    validator_class = getattr(validator_module, self._to_camel_case(validator_name))

                    # Add the validator to the guard
                    self._guard = self._guard.use(validator_class, **args)
            else:
                # Create an empty guard
                self._guard = self._guardrails.Guard()

            self._initialized = True

        except Exception as e:
            raise ValidationError(f"Failed to initialize GuardrailsAI: {str(e)}")

    def _to_camel_case(self, snake_str: str) -> str:
        """
        Convert a snake_case string to CamelCase.

        Args:
            snake_str: The snake_case string to convert.

        Returns:
            The CamelCase string.
        """
        components = snake_str.split("_")
        return "".join(x.title() for x in components)

    def validate(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """
        Validate text using GuardrailsAI.

        Args:
            text: The text to validate.
            metadata: Optional metadata to pass to GuardrailsAI validators.

        Returns:
            A ValidationResult indicating whether the text passed validation.

        Raises:
            ValidationError: If validation fails due to an error.
        """
        try:
            # Initialize if needed
            self._initialize()

            # Handle empty text
            if not text:
                return ValidationResult(
                    passed=False,
                    message="Input text is empty",
                    details={"input_length": 0},
                )

            # Validate the text using GuardrailsAI
            result = self._guard.parse(text, metadata=metadata)

            # Convert GuardrailsAI result to Sifaka ValidationResult
            passed = result.validation_passed

            # Extract validation details
            details = {
                "input_length": len(text),
                "validation_passed": passed,
            }

            # Add validation errors if any
            if hasattr(result, "validation_errors") and result.validation_errors:
                details["validation_errors"] = result.validation_errors

            # Add fix details if any
            if hasattr(result, "fixed_output") and result.fixed_output:
                details["fixed_output"] = result.fixed_output

            # Create message
            if passed:
                message = "Text passed GuardrailsAI validation"
            else:
                error_messages = []
                if hasattr(result, "validation_errors") and result.validation_errors:
                    for error in result.validation_errors:
                        if isinstance(error, dict) and "message" in error:
                            error_messages.append(error["message"])
                        elif hasattr(error, "message"):
                            error_messages.append(error.message)
                        else:
                            error_messages.append(str(error))

                if error_messages:
                    message = f"Text failed GuardrailsAI validation: {'; '.join(error_messages)}"
                else:
                    message = "Text failed GuardrailsAI validation"

            return ValidationResult(
                passed=passed,
                message=message,
                details=details,
            )

        except Exception as e:
            logger.exception("GuardrailsAI validation failed")
            raise ValidationError(f"GuardrailsAI validation failed: {str(e)}")


@register_validator("guardrails")
def create_guardrails_validator(
    guard=None,
    validators: Optional[List[str]] = None,
    validator_args: Optional[Dict[str, Dict[str, Any]]] = None,
    api_key: Optional[str] = None,
    name: str = "guardrails_validator",
    description: str = "Validates text using GuardrailsAI",
    **options: Any,
) -> GuardrailsValidator:
    """
    Create a GuardrailsAI validator.

    This factory function creates a GuardrailsValidator with the specified parameters.
    It is registered with the registry system for dependency injection.

    Args:
        guard: Optional pre-configured GuardrailsAI Guard instance.
        validators: Optional list of GuardrailsAI validator names to use.
        validator_args: Optional dictionary of arguments for each validator.
        api_key: Optional GuardrailsAI API key. If not provided, will use the
                 GUARDRAILS_API_KEY environment variable.
        name: The name of the validator.
        description: The description of the validator.
        **options: Additional options (ignored).

    Returns:
        A GuardrailsValidator instance.

    Raises:
        ValidationError: If the configuration is invalid.
    """
    return GuardrailsValidator(
        guard=guard,
        validators=validators,
        validator_args=validator_args,
        api_key=api_key,
        name=name,
        description=description,
    )


def guardrails_validator(
    guard=None,
    validators: Optional[List[str]] = None,
    validator_args: Optional[Dict[str, Dict[str, Any]]] = None,
    api_key: Optional[str] = None,
    name: str = "guardrails_validator",
    description: str = "Validates text using GuardrailsAI",
) -> GuardrailsValidator:
    """
    Create a GuardrailsAI validator.

    This is a convenience function for creating a GuardrailsValidator.

    Args:
        guard: Optional pre-configured GuardrailsAI Guard instance.
        validators: Optional list of GuardrailsAI validator names to use.
        validator_args: Optional dictionary of arguments for each validator.
        api_key: Optional GuardrailsAI API key. If not provided, will use the
                 GUARDRAILS_API_KEY environment variable.
        name: The name of the validator.
        description: The description of the validator.

    Returns:
        A GuardrailsValidator instance.

    Raises:
        ValidationError: If the configuration is invalid.
    """
    return GuardrailsValidator(
        guard=guard,
        validators=validators,
        validator_args=validator_args,
        api_key=api_key,
        name=name,
        description=description,
    )
