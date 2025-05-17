"""
Base validator class for Sifaka.

This module provides a base validator class that implements consistent error handling
and validation patterns for all validators in the Sifaka framework.
"""

import time
import logging
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable, TypeVar, Generic

from sifaka.results import ValidationResult as SifakaValidationResult
from sifaka.errors import ValidationError
from sifaka.utils.error_handling import validation_context, log_error
from sifaka.interfaces import Validator, ValidationResult
from sifaka.utils.validation_wrapper import wrap_validation_result

# Configure logger
logger = logging.getLogger(__name__)

# Type variable for generic validators
T = TypeVar("T")


@runtime_checkable
class ValidatorProtocol(Protocol):
    """Protocol for validators that check if text meets specific criteria."""

    def validate(self, text: str) -> ValidationResult:
        """
        Validate text against specific criteria.

        Args:
            text: The text to validate

        Returns:
            A ValidationResult with validation details
        """
        ...

    @property
    def name(self) -> str:
        """
        Get the name of the validator.

        Returns:
            The name of the validator
        """
        ...


class BaseValidator(Validator):
    """
    Base class for all validators in Sifaka.

    This class provides common functionality for validators, including:
    - Consistent error handling
    - Validation result creation
    - Empty text handling
    - Performance tracking

    All validators should inherit from this class to ensure consistent behavior.

    Attributes:
        name: The name of the validator
    """

    def __init__(self, name: Optional[str] = None, **options: Any):
        """
        Initialize the validator.

        Args:
            name: Optional name for the validator
            **options: Additional options for the validator
        """
        self._name = name or self.__class__.__name__
        self._options = options

    @property
    def name(self) -> str:
        """
        Get the name of the validator.

        Returns:
            The name of the validator
        """
        return self._name

    def configure(self, **options: Any) -> None:
        """
        Configure the validator with new options.

        This method allows updating the validator's configuration after initialization.
        Subclasses should override this method to handle specific configuration options.

        Args:
            **options: Configuration options to apply to the validator
        """
        # Update options
        self._options.update(options)
        logger.debug(f"Configured validator '{self.name}' with options: {options}")

    def validate(self, text: str) -> ValidationResult:
        """
        Validate text against specific criteria.

        This method should be overridden by subclasses to implement
        specific validation logic. The base implementation handles
        empty text and provides consistent error handling.

        Args:
            text: The text to validate

        Returns:
            A ValidationResult with validation details

        Raises:
            ValidationError: If validation fails due to an error
        """
        start_time = time.time()

        try:
            # Handle empty text
            if not text or not text.strip():
                return self._create_empty_text_result()

            # Perform validation
            with validation_context(validator_name=self.name):
                result = self._validate(text)

            # Add processing time
            if hasattr(result, "details") and result.details is not None:
                result.details["processing_time_ms"] = (time.time() - start_time) * 1000

            # Wrap the result to ensure it conforms to the ValidationResult protocol
            return wrap_validation_result(result)

        except Exception as e:
            # Log the error
            log_error(e, logger, component="Validator", operation=f"{self.name}.validate")

            # Create an error result
            return wrap_validation_result(self._create_error_result(e, start_time))

    def _validate(self, text: str) -> SifakaValidationResult:
        """
        Implement specific validation logic.

        This method should be overridden by subclasses to implement
        specific validation logic.

        Args:
            text: The text to validate

        Returns:
            A SifakaValidationResult with validation details

        Raises:
            NotImplementedError: If not overridden by subclasses
        """
        raise NotImplementedError("Subclasses must implement _validate method")

    def _create_empty_text_result(self) -> ValidationResult:
        """
        Create a validation result for empty text.

        Returns:
            A ValidationResult for empty text
        """
        return wrap_validation_result(
            SifakaValidationResult(
                passed=False,
                message="Empty text is not valid",
                score=0.0,
                issues=["Text is empty"],
                suggestions=["Provide non-empty content"],
                _details={
                    "validator_name": self.name,
                    "error_type": "EmptyText",
                },
            )
        )

    def _create_error_result(self, error: Exception, start_time: float) -> ValidationResult:
        """
        Create a validation result for an error.

        Args:
            error: The exception that occurred
            start_time: The time when validation started

        Returns:
            A ValidationResult for the error
        """
        processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds

        return wrap_validation_result(
            SifakaValidationResult(
                passed=False,
                message=f"Validation error: {str(error)}",
                score=0.0,
                issues=[f"Validation error: {str(error)}"],
                suggestions=["Check input format or try again"],
                _details={
                    "validator_name": self.name,
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                    "processing_time_ms": processing_time,
                },
            )
        )


def safe_validate(validator: ValidatorProtocol, text: str) -> ValidationResult:
    """
    Safely validate text with a validator.

    This utility function wraps validator.validate with error handling
    to ensure that validation errors are properly handled and don't
    propagate to the caller.

    Args:
        validator: The validator to use
        text: The text to validate

    Returns:
        A ValidationResult with validation details
    """
    start_time = time.time()

    try:
        return validator.validate(text)
    except Exception as e:
        # Log the error
        log_error(e, logger, component="Validator", operation=f"{validator.name}.validate")

        # Create an error result
        processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds

        return wrap_validation_result(
            SifakaValidationResult(
                passed=False,
                message=f"Validation error: {str(e)}",
                score=0.0,
                issues=[f"Validation error: {str(e)}"],
                suggestions=["Check input format or try again"],
                _details={
                    "validator_name": getattr(validator, "name", validator.__class__.__name__),
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "processing_time_ms": processing_time,
                },
            )
        )
