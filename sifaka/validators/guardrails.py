"""GuardrailsAI validator for Sifaka.

This module provides a GuardrailsValidator that integrates with GuardrailsAI
for advanced text validation including PII detection, toxicity checking,
and other safety measures.

The GuardrailsValidator requires the guardrails-ai library and an API key
to function properly.
"""

import os
import time
from typing import Any, Dict, List, Optional

from sifaka.core.thought import Thought, ValidationResult
from sifaka.utils.error_handling import ValidationError
from sifaka.utils.logging import get_logger
from sifaka.validators.shared import BaseValidator

# Configure logger
logger = get_logger(__name__)


class GuardrailsValidator(BaseValidator):
    """Validator that uses GuardrailsAI for validation.

    This validator integrates with GuardrailsAI to provide advanced text validation
    including PII detection, toxicity checking, and other safety measures.

    Attributes:
        guard: The GuardrailsAI Guard instance.
        validators: List of GuardrailsAI validators to use.
        validator_args: Dictionary of arguments for each validator.
        api_key: The GuardrailsAI API key.
        name: The name of the validator.
    """

    def __init__(
        self,
        guard: Optional[Any] = None,
        validators: Optional[List[str]] = None,
        validator_args: Optional[Dict[str, Dict[str, Any]]] = None,
        api_key: Optional[str] = None,
        name: str = "GuardrailsValidator",
    ):
        """Initialize the validator.

        Args:
            guard: Pre-configured GuardrailsAI Guard instance.
            validators: List of GuardrailsAI validator names to use.
            validator_args: Dictionary of arguments for each validator.
            api_key: The GuardrailsAI API key (or use GUARDRAILS_API_KEY env var).
            name: The name of the validator.

        Raises:
            ValidationError: If GuardrailsAI is not available or configuration is invalid.
        """
        super().__init__(name)
        self.validators = validators or []
        self.validator_args = validator_args or {}

        # Get API key from parameter or environment
        self.api_key = api_key or os.getenv("GUARDRAILS_API_KEY")

        # Initialize GuardrailsAI
        self._guard = None
        self._guardrails = None
        self._initialized = False

        if guard is not None:
            # Use pre-configured guard
            self._guard = guard
            self._initialized = True
        else:
            # Initialize guard with validators
            self._initialize_guardrails()

    def _initialize_guardrails(self) -> None:
        """Initialize GuardrailsAI components."""
        try:
            import guardrails as gd

            self._guardrails = gd

            # Check if API key is available
            if not self.api_key:
                logger.warning(
                    f"{self.name}: No GuardrailsAI API key found. "
                    "Set GUARDRAILS_API_KEY environment variable or pass api_key parameter."
                )
                # Some validators might work without API key

            # Create guard with validators
            if self.validators:
                guard_validators = []
                for validator_name in self.validators:
                    try:
                        # Try to get validator class from hub first, then from validators
                        validator_class = None

                        # Try guardrails.hub first (for installed validators)
                        try:
                            from guardrails.hub import GuardrailsPII

                            if validator_name == "GuardrailsPII":
                                validator_class = GuardrailsPII
                        except ImportError:
                            pass

                        # If not found in hub, try gd.validators (for built-in validators)
                        if validator_class is None:
                            try:
                                validator_class = getattr(gd.validators, validator_name)
                            except AttributeError:
                                pass

                        # If still not found, raise error
                        if validator_class is None:
                            raise AttributeError(f"Validator {validator_name} not found")

                        # Get arguments for this validator
                        args = self.validator_args.get(validator_name, {})

                        # Create validator instance
                        validator_instance = validator_class(**args)
                        guard_validators.append(validator_instance)

                        logger.debug(f"{self.name}: Added validator {validator_name}")

                    except AttributeError:
                        logger.error(f"{self.name}: Unknown validator: {validator_name}")
                        raise ValidationError(
                            message=f"Unknown GuardrailsAI validator: {validator_name}",
                            component="GuardrailsValidator",
                            operation="initialization",
                            suggestions=[
                                "Check the validator name spelling",
                                "Ensure the validator is available in your GuardrailsAI version",
                                "Install the validator: guardrails hub install hub://guardrails/guardrails_pii",
                            ],
                        )
                    except Exception as e:
                        logger.error(
                            f"{self.name}: Failed to create validator {validator_name}: {e}"
                        )
                        raise ValidationError(
                            message=f"Failed to create validator {validator_name}: {str(e)}",
                            component="GuardrailsValidator",
                            operation="initialization",
                            suggestions=[
                                "Check validator arguments",
                                "Ensure all required parameters are provided",
                            ],
                        )

                # Create guard
                self._guard = gd.Guard()
                for validator in guard_validators:
                    self._guard.use(validator)
                logger.debug(f"{self.name}: Created guard with {len(guard_validators)} validators")
            else:
                # Create empty guard
                self._guard = gd.Guard()
                logger.debug(f"{self.name}: Created empty guard")

            self._initialized = True

        except ImportError:
            logger.error(f"{self.name}: GuardrailsAI library not available")
            raise ValidationError(
                message="GuardrailsAI library not available",
                component="GuardrailsValidator",
                operation="initialization",
                suggestions=[
                    "Install GuardrailsAI: pip install guardrails-ai",
                    "Ensure the library is properly installed",
                ],
            )
        except Exception as e:
            logger.error(f"{self.name}: Failed to initialize GuardrailsAI: {e}")
            raise ValidationError(
                message=f"Failed to initialize GuardrailsAI: {str(e)}",
                component="GuardrailsValidator",
                operation="initialization",
                suggestions=[
                    "Check GuardrailsAI installation",
                    "Verify API key configuration",
                    "Check validator configuration",
                ],
            )

    def _validate_content(self, thought: Thought) -> ValidationResult:
        """Validate text using GuardrailsAI.

        Args:
            thought: The Thought container with the text to validate.

        Returns:
            A ValidationResult with information about whether the validation passed,
            any issues found, and suggestions for improvement.
        """
        start_time = time.time()

        # Check if GuardrailsAI is initialized
        if not self._initialized:
            return self.create_validation_result(
                passed=False,
                message="GuardrailsAI not properly initialized",
                issues=["GuardrailsAI validator is not initialized"],
                suggestions=["Check GuardrailsAI configuration and API key"],
            )

        try:
            # Check if guard is available
            if self._guard is None:
                return self.create_validation_result(
                    passed=False,
                    message="GuardrailsAI guard is not available",
                    issues=["GuardrailsAI guard is not configured"],
                    suggestions=["Configure GuardrailsAI guard or install guardrails-ai"],
                )

            # Validate with GuardrailsAI
            result = self._guard.validate(thought.text)

            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000

            # Check validation result
            if result.validation_passed:
                logger.debug(f"{self.name}: Validation passed in {processing_time:.2f}ms")
                return self.create_validation_result(
                    passed=True,
                    message="Text passed GuardrailsAI validation",
                    score=1.0,
                )
            else:
                # Extract issues from GuardrailsAI result
                issues = []
                suggestions = []

                if hasattr(result, "error_spans") and result.error_spans:
                    for error_span in result.error_spans:
                        issues.append(f"Validation error: {error_span.reason}")
                        suggestions.append(f"Fix issue: {error_span.reason}")

                if hasattr(result, "errors") and result.errors:
                    for error in result.errors:
                        issues.append(f"Validation error: {str(error)}")
                        suggestions.append("Address the validation error")

                if not issues:
                    issues = ["Text failed GuardrailsAI validation"]
                    suggestions = ["Modify text to meet validation requirements"]

                logger.debug(
                    f"{self.name}: Validation failed with {len(issues)} issues in {processing_time:.2f}ms"
                )

                return self.create_validation_result(
                    passed=False,
                    message="Text failed GuardrailsAI validation",
                    score=0.0,
                    issues=issues,
                    suggestions=suggestions,
                )

        except Exception as e:
            logger.error(f"{self.name}: GuardrailsAI validation failed: {e}")
            return self.create_error_result(e, self.name, "guardrails_validation")

    async def _validate_async(self, thought: Thought) -> ValidationResult:
        """Validate text using GuardrailsAI asynchronously.

        This is the internal async implementation that provides the same functionality
        as the sync validate method but can be called concurrently with other validators.

        Args:
            thought: The Thought container with the text to validate.

        Returns:
            A ValidationResult with information about whether the validation passed,
            any issues found, and suggestions for improvement.
        """
        # GuardrailsAI validation involves API calls, so this could benefit from true async
        # For now, we'll call the sync version, but this could be improved with async HTTP
        return self.validate(thought)


def create_guardrails_validator(
    guard: Optional[Any] = None,
    validators: Optional[List[str]] = None,
    validator_args: Optional[Dict[str, Dict[str, Any]]] = None,
    api_key: Optional[str] = None,
    name: str = "GuardrailsValidator",
) -> GuardrailsValidator:
    """Create a GuardrailsAI validator.

    Args:
        guard: Pre-configured GuardrailsAI Guard instance.
        validators: List of GuardrailsAI validator names to use.
        validator_args: Dictionary of arguments for each validator.
        api_key: The GuardrailsAI API key (or use GUARDRAILS_API_KEY env var).
        name: The name of the validator.

    Returns:
        A GuardrailsValidator instance.
    """
    return GuardrailsValidator(
        guard=guard,
        validators=validators,
        validator_args=validator_args,
        api_key=api_key,
        name=name,
    )


def guardrails_validator(
    validators: List[str],
    validator_args: Optional[Dict[str, Dict[str, Any]]] = None,
    api_key: Optional[str] = None,
) -> GuardrailsValidator:
    """Create a GuardrailsAI validator.

    This is a convenience function for creating a GuardrailsValidator.

    Args:
        validators: List of GuardrailsAI validator names to use.
        validator_args: Dictionary of arguments for each validator.
        api_key: The GuardrailsAI API key (or use GUARDRAILS_API_KEY env var).

    Returns:
        A GuardrailsValidator instance.
    """
    return create_guardrails_validator(
        validators=validators,
        validator_args=validator_args,
        api_key=api_key,
        name="GuardrailsAIValidator",
    )
