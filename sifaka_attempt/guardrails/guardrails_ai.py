"""
Guardrails-AI integration for Sifaka.

This module provides integration with the guardrails-ai library for adding
safety and compliance features to text processing pipelines.
"""

import importlib
from typing import Any, Dict, List, Optional, Union, cast

from ..di import inject
from . import Guardrail, GuardrailResult


class GuardrailsAIAdapter:
    """
    Adapter for the guardrails-ai library.

    This class provides a wrapper around guardrails-ai validators to make them
    compatible with Sifaka's guardrails system.
    """

    def __init__(self):
        """Initialize the adapter and check if guardrails-ai is available."""
        self.guardrails_available = self._check_guardrails_available()

    def _check_guardrails_available(self) -> bool:
        """Check if guardrails-ai is available."""
        try:
            importlib.import_module("guardrails")
            return True
        except ImportError:
            return False


class GuardrailsValidator(Guardrail):
    """
    Guardrail that uses a guardrails-ai validator.

    This guardrail wraps a guardrails-ai validator to provide safety and
    compliance features for text processing pipelines.
    """

    def __init__(
        self,
        validator_name: str,
        validator_config: Optional[Dict[str, Any]] = None,
        on_fail_action: str = "warn",  # "warn", "filter", or "exception"
    ):
        """
        Initialize with a guardrails-ai validator.

        Args:
            validator_name: Name of the guardrails-ai validator to use
            validator_config: Optional configuration for the validator
            on_fail_action: Action to take when validation fails

        Raises:
            ImportError: If guardrails-ai is not installed
            ValueError: If validator_name is invalid
        """
        # Check if guardrails-ai is available
        try:
            guardrails = importlib.import_module("guardrails")
        except ImportError:
            raise ImportError(
                "guardrails-ai is required for GuardrailsValidator. "
                "Install it with: pip install guardrails-ai"
            )

        # Initialize validator
        try:
            validator_class = getattr(guardrails.validators, validator_name)
            self.validator = validator_class(**(validator_config or {}))
        except (AttributeError, TypeError) as e:
            raise ValueError(f"Invalid guardrails-ai validator: {validator_name}") from e

        self.on_fail_action = on_fail_action

    def process(self, text: str) -> str:
        """
        Process text using the guardrails-ai validator.

        Args:
            text: The text to process

        Returns:
            The processed text (may be modified if on_fail_action is "filter")
        """
        result = self.run(text)
        return result.processed_text

    def validate(self, text: str) -> bool:
        """
        Validate text using the guardrails-ai validator.

        Args:
            text: The text to validate

        Returns:
            True if the text passes validation, False otherwise
        """
        result = self.run(text)
        return result.passed

    def run(self, text: str) -> GuardrailResult:
        """
        Run the guardrails-ai validator on the text.

        Args:
            text: The text to process

        Returns:
            GuardrailResult with the processed text and validation information
        """
        if not text:
            return GuardrailResult(
                processed_text="",
                passed=True,
                message="Empty text processed",
            )

        try:
            # Run the validator
            result = self.validator.validate(text)

            # Check if validation passed
            if hasattr(result, "__class__") and result.__class__.__name__ == "PassResult":
                return GuardrailResult(
                    processed_text=text,
                    passed=True,
                    message="Text passed validation",
                )
            else:
                # Handle validation failure based on on_fail_action
                issue = str(getattr(result, "message", "Validation failed"))
                if self.on_fail_action == "filter":
                    # Try to get fixed value
                    fixed = getattr(result, "fixed_value", text)
                    return GuardrailResult(
                        processed_text=fixed,
                        passed=False,
                        message=f"Text failed validation: {issue}",
                        issues=[issue],
                    )
                elif self.on_fail_action == "exception":
                    raise ValueError(f"Guardrail validation failed: {issue}")
                else:  # "warn"
                    return GuardrailResult(
                        processed_text=text,
                        passed=False,
                        message=f"Text failed validation: {issue}",
                        issues=[issue],
                    )
        except Exception as e:
            if self.on_fail_action == "exception":
                raise
            return GuardrailResult(
                processed_text=text,
                passed=False,
                message=f"Error in validation: {str(e)}",
                issues=[str(e)],
                metadata={"error": str(e), "error_type": type(e).__name__},
            )


class GuardrailsRail(Guardrail):
    """
    Guardrail that uses a guardrails-ai Guard.

    This guardrail wraps a guardrails-ai Guard to provide safety and
    compliance features for text processing pipelines.
    """

    def __init__(
        self,
        rail_spec: str,
        on_fail_action: str = "warn",  # "warn", "filter", or "exception"
    ):
        """
        Initialize with a guardrails-ai Rail specification.

        Args:
            rail_spec: Rail specification in guardrails-ai format
            on_fail_action: Action to take when validation fails

        Raises:
            ImportError: If guardrails-ai is not installed
            ValueError: If rail_spec is invalid
        """
        # Check if guardrails-ai is available
        try:
            guardrails = importlib.import_module("guardrails")
        except ImportError:
            raise ImportError(
                "guardrails-ai is required for GuardrailsRail. "
                "Install it with: pip install guardrails-ai"
            )

        # Initialize Guard
        try:
            self.guard = guardrails.Guard.from_rail_string(rail_spec)
        except Exception as e:
            raise ValueError(f"Invalid guardrails-ai rail specification: {str(e)}") from e

        self.on_fail_action = on_fail_action

    def process(self, text: str) -> str:
        """
        Process text using the guardrails-ai Guard.

        Args:
            text: The text to process

        Returns:
            The processed text (may be modified if on_fail_action is "filter")
        """
        result = self.run(text)
        return result.processed_text

    def validate(self, text: str) -> bool:
        """
        Validate text using the guardrails-ai Guard.

        Args:
            text: The text to validate

        Returns:
            True if the text passes validation, False otherwise
        """
        result = self.run(text)
        return result.passed

    def run(self, text: str) -> GuardrailResult:
        """
        Run the guardrails-ai Guard on the text.

        Args:
            text: The text to process

        Returns:
            GuardrailResult with the processed text and validation information
        """
        if not text:
            return GuardrailResult(
                processed_text="",
                passed=True,
                message="Empty text processed",
            )

        try:
            # Run the Guard
            # Note: guards return the validated output or raise ValidationError
            validated = self.guard(text)
            return GuardrailResult(
                processed_text=validated,
                passed=True,
                message="Text passed validation",
            )
        except Exception as e:
            if self.on_fail_action == "exception":
                raise
            elif self.on_fail_action == "filter":
                # Try to get a modified version if available
                validated = text  # Default to original
                return GuardrailResult(
                    processed_text=validated,
                    passed=False,
                    message=f"Text failed validation: {str(e)}",
                    issues=[str(e)],
                    metadata={"error": str(e), "error_type": type(e).__name__},
                )
            else:  # "warn"
                return GuardrailResult(
                    processed_text=text,
                    passed=False,
                    message=f"Text failed validation: {str(e)}",
                    issues=[str(e)],
                    metadata={"error": str(e), "error_type": type(e).__name__},
                )
