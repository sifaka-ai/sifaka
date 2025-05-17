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
import time
from typing import Any, Dict, List, Optional

from sifaka.results import ValidationResult as SifakaValidationResult
from sifaka.errors import ValidationError
from sifaka.registry import register_validator
from sifaka.validators.base import BaseValidator
from sifaka.utils.error_handling import validation_context, log_error

# Set up logging
logger = logging.getLogger(__name__)


class GuardrailsValidator(BaseValidator):
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
        guard: Any = None,
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
        # Initialize the base validator with a name
        super().__init__(name=name)

        # Log initialization attempt
        logger.debug(
            f"Initializing GuardrailsValidator with "
            f"guard={guard is not None}, "
            f"validators={validators}, "
            f"api_key={'provided' if api_key else 'from env'}"
        )

        try:
            self._description = description
            self._guard = None
            self._guardrails = None
            self._initialized = False
            self._api_key = api_key or os.environ.get("GUARDRAILS_API_KEY")

            # Check if API key is available
            if not self._api_key:
                logger.warning(
                    "No GuardrailsAI API key provided or found in environment. "
                    "Some validators may not work without an API key."
                )

            # Validate inputs with improved error handling
            if guard is not None and validators is not None:
                logger.error("Cannot provide both a guard and validators")
                raise ValidationError(
                    message="Cannot provide both a guard and validators. Choose one approach.",
                    component="GuardrailsValidator",
                    operation="initialization",
                    suggestions=[
                        "Provide either a pre-configured guard OR a list of validators, not both",
                        "For simple use cases, provide a list of validators",
                        "For complex use cases, configure a guard separately and provide it",
                    ],
                    metadata={
                        "has_guard": guard is not None,
                        "validators": validators,
                        "has_api_key": self._api_key is not None,
                    },
                )

            # Store configuration for lazy initialization
            self._guard_config = guard
            self._validators_config = validators
            self._validator_args_config = validator_args or {}

            logger.debug(f"Successfully initialized GuardrailsValidator configuration: {self.name}")

        except Exception as e:
            # Log the error
            log_error(e, logger, component="GuardrailsValidator", operation="initialization")

            # Re-raise as ValidationError with more context if not already a ValidationError
            if not isinstance(e, ValidationError):
                raise ValidationError(
                    message=f"Failed to initialize GuardrailsValidator: {str(e)}",
                    component="GuardrailsValidator",
                    operation="initialization",
                    suggestions=[
                        "Check if the guard or validators are properly configured",
                        "Verify that the API key is valid if provided",
                        "Check the error message for details",
                    ],
                    metadata={
                        "has_guard": guard is not None,
                        "validators": validators,
                        "has_api_key": api_key is not None,
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                    },
                )
            raise

    @property
    def name(self) -> str:
        """Get the validator name."""
        return self._name

    @property
    def description(self) -> str:
        """Get the validator description."""
        return self._description

    def _load_guardrails(self) -> Any:
        """
        Load the guardrails-ai library.

        Returns:
            The guardrails module.

        Raises:
            ImportError: If guardrails-ai is not installed.
        """
        start_time = time.time()

        logger.debug("Loading guardrails-ai library")

        try:
            guardrails = importlib.import_module("guardrails")

            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds

            logger.debug(f"Successfully loaded guardrails-ai library in {processing_time:.2f}ms")

            return guardrails

        except ImportError as e:
            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds

            logger.error(
                f"Failed to load guardrails-ai library in {processing_time:.2f}ms: {str(e)}"
            )

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
        start_time = time.time()

        # Skip if already initialized
        if self._initialized:
            logger.debug(f"{self.name}: Already initialized, skipping")
            return

        logger.debug(f"{self.name}: Initializing GuardrailsAI validator")

        with validation_context(
            validator_name=self.name,
            operation="initialize",
            message_prefix="Failed to initialize GuardrailsAI validator",
            suggestions=[
                "Check if guardrails-ai is installed",
                "Verify that the API key is valid if provided",
                "Check that the validators are properly configured",
            ],
            metadata={
                "has_guard_config": self._guard_config is not None,
                "validators_config": self._validators_config,
                "has_api_key": self._api_key is not None,
            },
        ):
            # Load guardrails
            self._guardrails = self._load_guardrails()

            # If a guard was provided, use it
            if self._guard_config is not None:
                logger.debug(f"{self.name}: Using provided guard")
                self._guard = self._guard_config
            # Otherwise, create a new guard with the specified validators
            elif self._validators_config is not None:
                logger.debug(
                    f"{self.name}: Creating new guard with {len(self._validators_config)} validators"
                )

                # Create a new guard
                if self._guardrails is not None:
                    self._guard = self._guardrails.Guard()
                else:
                    logger.warning(
                        f"{self.name}: Guardrails module not loaded, cannot create guard"
                    )

                # Install and add each validator
                for validator_name in self._validators_config:
                    validator_start_time = time.time()

                    # Get arguments for this validator
                    args = self._validator_args_config.get(validator_name, {})

                    logger.debug(
                        f"{self.name}: Installing validator '{validator_name}' with args: {args}"
                    )

                    # Install the validator
                    # Set API key in environment if provided
                    old_api_key = os.environ.get("GUARDRAILS_API_KEY")
                    if self._api_key:
                        os.environ["GUARDRAILS_API_KEY"] = self._api_key

                    try:
                        if self._guardrails is not None and self._guard is not None:
                            validator_module = self._guardrails.install(
                                f"hub://guardrails/{validator_name}"
                            )

                            # Get the validator class
                            validator_class = getattr(
                                validator_module, self._to_camel_case(validator_name)
                            )

                            # Add the validator to the guard
                            self._guard = self._guard.use(validator_class, **args)
                        else:
                            logger.warning(
                                f"{self.name}: Guardrails module or guard not initialized, cannot install validator"
                            )

                        # Calculate validator processing time
                        validator_time = (time.time() - validator_start_time) * 1000  # ms

                        logger.debug(
                            f"{self.name}: Successfully installed validator '{validator_name}' "
                            f"in {validator_time:.2f}ms"
                        )

                    except Exception as e:
                        # Calculate validator processing time
                        validator_time = (time.time() - validator_start_time) * 1000  # ms

                        logger.error(
                            f"{self.name}: Failed to install validator '{validator_name}' "
                            f"in {validator_time:.2f}ms: {str(e)}"
                        )

                        raise ValidationError(
                            message=f"Failed to install validator '{validator_name}': {str(e)}",
                            component="GuardrailsValidator",
                            operation="initialize_validator",
                            suggestions=[
                                "Check if the validator name is correct",
                                "Verify that the API key is valid",
                                "Check if the validator is available in the GuardrailsAI hub",
                            ],
                            metadata={
                                "validator_name": validator_name,
                                "args": args,
                                "error_type": type(e).__name__,
                                "error_message": str(e),
                            },
                        )

                    finally:
                        # Restore original API key
                        if self._api_key:
                            if old_api_key:
                                os.environ["GUARDRAILS_API_KEY"] = old_api_key
                            else:
                                os.environ.pop("GUARDRAILS_API_KEY", None)
                else:
                    # Create an empty guard
                    logger.debug(f"{self.name}: Creating empty guard (no validators specified)")
                    if self._guardrails is not None:
                        self._guard = self._guardrails.Guard()
                    else:
                        logger.warning(
                            f"{self.name}: Guardrails module not loaded, cannot create guard"
                        )

            self._initialized = True

            # Calculate total processing time
            processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds

            logger.debug(
                f"{self.name}: Successfully initialized GuardrailsAI validator in {processing_time:.2f}ms"
            )

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

    def _validate(
        self, text: str, metadata: Optional[Dict[str, Any]] = None
    ) -> SifakaValidationResult:
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
        start_time = time.time()

        logger.debug(f"{self.name}: Validating text of length {len(text)}")

        try:
            # Initialize if needed
            self._initialize()

            # Handle empty text
            if not text:
                logger.debug(f"{self.name}: Empty text provided, returning fail result")
                return SifakaValidationResult(
                    passed=False,
                    message="Input text is empty",
                    details={"input_length": 0},
                    score=0.0,
                    issues=["Input text is empty"],
                    suggestions=["Provide non-empty text for validation"],
                )

            # Validate the text using GuardrailsAI
            with validation_context(
                validator_name=self.name,
                operation="validate",
                message_prefix="Failed to validate text with GuardrailsAI",
                suggestions=[
                    "Check if the guard is properly configured",
                    "Verify that the API key is valid if required",
                    "Check the error message for details",
                ],
                metadata={
                    "text_length": len(text),
                    "has_metadata": metadata is not None,
                    "initialized": self._initialized,
                },
            ):
                if self._guard is None:
                    logger.warning(f"{self.name}: Guard is not initialized, returning failure")
                    return SifakaValidationResult(
                        passed=False,
                        message="Guard is not initialized",
                        details={"input_length": len(text)},
                        score=0.0,
                        issues=["Guard is not initialized"],
                        suggestions=["Check if GuardrailsAI is properly installed and configured"],
                    )

                result = self._guard.parse(text, metadata=metadata)

                # Convert GuardrailsAI result to ValidationResult
                passed = getattr(result, "validation_passed", False)

                # Extract validation details
                details = {
                    "input_length": len(text),
                    "validation_passed": passed,
                    "validator_name": self.name,
                }

            # Add validation errors if any
            error_messages = []
            try:
                if hasattr(result, "validation_errors") and result.validation_errors:
                    details["validation_errors"] = result.validation_errors

                    # Extract error messages
                    for error in result.validation_errors:
                        if isinstance(error, dict) and "message" in error:
                            error_messages.append(error["message"])
                        elif hasattr(error, "message"):
                            error_messages.append(error.message)
                        else:
                            error_messages.append(str(error))
            except Exception as e:
                logger.warning(f"{self.name}: Error extracting validation errors: {str(e)}")

            # Add fix details if any
            if hasattr(result, "fixed_output") and result.fixed_output:
                details["fixed_output"] = result.fixed_output

            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds
            details["processing_time_ms"] = processing_time

            # Create issues and suggestions
            issues = []
            suggestions = []

            if not passed:
                if error_messages:
                    issues = error_messages
                else:
                    issues = ["Text failed GuardrailsAI validation"]

                suggestions = ["Review the validation errors and modify the text accordingly"]

                if hasattr(result, "fixed_output") and result.fixed_output:
                    suggestions.append("Consider using the fixed output provided by GuardrailsAI")

            # Calculate score based on validation result
            score = 1.0 if passed else 0.0

            # Create message
            if passed:
                message = "Text passed GuardrailsAI validation"
                logger.debug(f"{self.name}: Validation passed in {processing_time:.2f}ms")
            else:
                if error_messages:
                    message = f"Text failed GuardrailsAI validation: {'; '.join(error_messages)}"
                else:
                    message = "Text failed GuardrailsAI validation"

                logger.debug(
                    f"{self.name}: Validation failed with {len(error_messages)} errors in {processing_time:.2f}ms"
                )

            return SifakaValidationResult(
                passed=passed,
                message=message,
                details=details,
                score=score,
                issues=issues,
                suggestions=suggestions,
            )

        except Exception as e:
            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds

            # Log the error
            log_error(e, logger, component="GuardrailsValidator", operation="validate")

            # Raise as ValidationError with more context
            raise ValidationError(
                message=f"GuardrailsAI validation failed: {str(e)}",
                component="GuardrailsValidator",
                operation="validate",
                suggestions=[
                    "Check if the guard is properly configured",
                    "Verify that the API key is valid if required",
                    "Check the error message for details",
                ],
                metadata={
                    "text_length": len(text) if text else 0,
                    "has_metadata": metadata is not None,
                    "initialized": self._initialized,
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "processing_time_ms": processing_time,
                },
            )


@register_validator("guardrails")
def create_guardrails_validator(
    guard: Any = None,
    validators: Optional[List[str]] = None,
    validator_args: Optional[Dict[str, Dict[str, Any]]] = None,
    api_key: Optional[str] = None,
    name: str = "guardrails_validator",
    description: str = "Validates text using GuardrailsAI",
    **_options: Any,  # Unused options
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
        **_options: Additional options (ignored).

    Returns:
        A GuardrailsValidator instance.

    Raises:
        ValidationError: If the configuration is invalid.
    """
    start_time = time.time()

    try:
        # Log factory function call
        logger.debug(
            f"Creating GuardrailsValidator with "
            f"guard={guard is not None}, "
            f"validators={validators}, "
            f"api_key={'provided' if api_key else 'from env'}, "
            f"name={name}"
        )

        # Validate inputs
        if guard is not None and validators is not None:
            logger.error("Cannot provide both a guard and validators")
            raise ValidationError(
                message="Cannot provide both a guard and validators. Choose one approach.",
                component="GuardrailsValidatorFactory",
                operation="create_validator",
                suggestions=[
                    "Provide either a pre-configured guard OR a list of validators, not both",
                    "For simple use cases, provide a list of validators",
                    "For complex use cases, configure a guard separately and provide it",
                ],
                metadata={
                    "has_guard": guard is not None,
                    "validators": validators,
                    "has_api_key": api_key is not None,
                },
            )

        # Create the validator
        validator = GuardrailsValidator(
            guard=guard,
            validators=validators,
            validator_args=validator_args,
            api_key=api_key,
            name=name,
            description=description,
        )

        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds

        # Log successful creation
        logger.debug(
            f"Successfully created GuardrailsValidator: {validator.name} in {processing_time:.2f}ms"
        )

        return validator

    except Exception as e:
        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds

        # Log the error
        log_error(e, logger, component="GuardrailsValidatorFactory", operation="create_validator")

        # Re-raise as ValidationError with more context if not already a ValidationError
        if not isinstance(e, ValidationError):
            raise ValidationError(
                message=f"Failed to create GuardrailsValidator: {str(e)}",
                component="GuardrailsValidatorFactory",
                operation="create_validator",
                suggestions=[
                    "Check if the guard or validators are properly configured",
                    "Verify that the API key is valid if provided",
                    "Check the error message for details",
                ],
                metadata={
                    "has_guard": guard is not None,
                    "validators": validators,
                    "has_api_key": api_key is not None,
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "processing_time_ms": processing_time,
                },
            )
        raise


def guardrails_validator(
    guard: Any = None,
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
    start_time = time.time()

    try:
        # Log function call
        logger.debug(
            f"Creating GuardrailsValidator with "
            f"guard={guard is not None}, "
            f"validators={validators}, "
            f"api_key={'provided' if api_key else 'from env'}, "
            f"name={name}"
        )

        # Validate inputs
        if guard is not None and validators is not None:
            logger.error("Cannot provide both a guard and validators")
            raise ValidationError(
                message="Cannot provide both a guard and validators. Choose one approach.",
                component="GuardrailsValidatorFunction",
                operation="guardrails_validator",
                suggestions=[
                    "Provide either a pre-configured guard OR a list of validators, not both",
                    "For simple use cases, provide a list of validators",
                    "For complex use cases, configure a guard separately and provide it",
                ],
                metadata={
                    "has_guard": guard is not None,
                    "validators": validators,
                    "has_api_key": api_key is not None,
                },
            )

        # Create the validator
        validator = GuardrailsValidator(
            guard=guard,
            validators=validators,
            validator_args=validator_args,
            api_key=api_key,
            name=name,
            description=description,
        )

        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds

        # Log successful creation
        logger.debug(
            f"Successfully created GuardrailsValidator: {validator.name} in {processing_time:.2f}ms"
        )

        return validator

    except Exception as e:
        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds

        # Log the error
        log_error(
            e, logger, component="GuardrailsValidatorFunction", operation="guardrails_validator"
        )

        # Re-raise as ValidationError with more context if not already a ValidationError
        if not isinstance(e, ValidationError):
            raise ValidationError(
                message=f"Failed to create GuardrailsValidator: {str(e)}",
                component="GuardrailsValidatorFunction",
                operation="guardrails_validator",
                suggestions=[
                    "Check if the guard or validators are properly configured",
                    "Verify that the API key is valid if provided",
                    "Check the error message for details",
                ],
                metadata={
                    "has_guard": guard is not None,
                    "validators": validators,
                    "has_api_key": api_key is not None,
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "processing_time_ms": processing_time,
                },
            )
        raise
