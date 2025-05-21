"""
Base validator class for Sifaka.

This module provides a base validator class that implements consistent error handling
and validation patterns for all validators in the Sifaka framework.
"""

import logging
import time
from abc import abstractmethod
from typing import Any, Dict, Optional

from sifaka.core.interfaces import Validator
from sifaka.core.thought import Thought

# Configure logger
logger = logging.getLogger(__name__)


class BaseValidator(Validator):
    """
    Base class for all validators in Sifaka.

    This class provides common functionality for validators, including:
    - Consistent error handling
    - Validation result creation
    - Empty text handling
    - Performance tracking

    All validators should inherit from this class to ensure consistent behavior.
    Subclasses only need to implement the _validate method to provide specific
    validation logic.

    Attributes:
        _name (str): The name of the validator.
        _options (Dict[str, Any]): Additional options for the validator.
    """

    def __init__(self, name: Optional[str] = None, **options: Any):
        """
        Initialize the validator.

        Args:
            name: The name of the validator. If not provided, uses the class name.
            **options: Additional options for the validator.
        """
        self._name = name or self.__class__.__name__
        self._options = options

        # Log initialization
        logger.debug(f"Initialized {self.name} validator with options: {options}")

    @property
    def name(self) -> str:
        """Return the name of the validator."""
        return self._name

    def validate(self, thought: Thought) -> bool:
        """
        Validate the text in the thought.

        This method handles the validation process, including:
        - Checking for empty text
        - Setting up error handling
        - Calling the _validate method to perform specific validation logic
        - Adding the validation result to the thought

        Subclasses should not override this method. Instead, they should
        override the _validate method to implement specific validation logic.

        Args:
            thought: The thought containing the text to validate.

        Returns:
            True if the text passes validation, False otherwise.
        """
        start_time = time.time()
        text = thought.text

        try:
            # Handle empty text
            if not text or not text.strip():
                thought.add_validation_result(
                    validator_name=self.name,
                    passed=False,
                    score=0.0,
                    details={
                        "validator_name": self.name,
                        "error_type": "EmptyText",
                    },
                    message="Empty text is not valid",
                )
                return False

            # Perform validation
            logger.debug(f"{self.name}: Validating text of length {len(text)}")
            result = self._validate(text)

            # Add processing time to details
            details = result.get("details", {})
            details["processing_time_ms"] = (time.time() - start_time) * 1000
            result["details"] = details

            # Add validation result to thought
            thought.add_validation_result(
                validator_name=self.name,
                passed=result["passed"],
                score=result.get("score", 1.0 if result["passed"] else 0.0),
                details=result.get("details", {}),
                message=result.get("message", "Validation completed"),
            )

            return result["passed"]

        except Exception as e:
            logger.error(f"Error in {self.name}: {str(e)}")

            # Add error validation result to thought
            thought.add_validation_result(
                validator_name=self.name,
                passed=False,
                score=0.0,
                details={
                    "validator_name": self.name,
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "processing_time_ms": (time.time() - start_time) * 1000,
                },
                message=f"Validation error: {str(e)}",
            )
            return False

    @abstractmethod
    def _validate(self, text: str) -> Dict[str, Any]:
        """
        Implement specific validation logic.

        This method should be overridden by subclasses to implement
        specific validation logic. It is called by the validate method
        after handling empty text and setting up error handling.

        Args:
            text: The text to validate.

        Returns:
            A dictionary with validation results, including at least:
            - passed: Whether the validation passed
            - message: A message describing the validation result
            - score: A score between 0.0 and 1.0 (optional)
            - details: Additional details about the validation (optional)

        Raises:
            NotImplementedError: If not overridden by subclasses.
        """
        raise NotImplementedError("Subclasses must implement _validate method")


def safe_validate(validator: Validator, thought: Thought) -> bool:
    """
    Safely validate text using a validator.

    This function wraps the validator's validate method with error handling
    to ensure that validation errors don't crash the application.

    Args:
        validator: The validator to use.
        thought: The thought containing the text to validate.

    Returns:
        True if the text passes validation, False otherwise.
    """
    start_time = time.time()

    try:
        return validator.validate(thought)
    except Exception as e:
        # Log the error
        logger.error(f"Error in {validator.name}: {str(e)}")

        # Add error validation result to thought
        thought.add_validation_result(
            validator_name=getattr(validator, "name", validator.__class__.__name__),
            passed=False,
            score=0.0,
            details={
                "error_type": type(e).__name__,
                "error_message": str(e),
                "processing_time_ms": (time.time() - start_time) * 1000,
            },
            message=f"Validation error: {str(e)}",
        )
        return False
