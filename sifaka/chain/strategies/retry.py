"""
Retry strategy module for Sifaka.

This module provides the RetryStrategy class and implementations which are
responsible for implementing retry logic.
"""

import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Generic, Optional, TypeVar

from ..interfaces.strategy import RetryStrategyProtocol
from ...critics import CriticCore
from ...generation import Generator
from ...models.base import ModelProvider
from ...validation import ValidationResult
from ..formatters.result import ResultFormatter
from ..managers.prompt import PromptManager
from ..managers.validation import ValidationManager
from ..result import ChainResult
from ...utils.logging import get_logger

logger = get_logger(__name__)

OutputType = TypeVar("OutputType")


class RetryStrategy(RetryStrategyProtocol, Generic[OutputType], ABC):
    """
    Abstract base class for retry strategies.

    This class is responsible for implementing retry logic for chains.
    It implements the RetryStrategyProtocol interface.
    """

    @abstractmethod
    def run(
        self,
        prompt: str,
        generator: Generator[OutputType],
        validation_manager: ValidationManager[OutputType],
        prompt_manager: PromptManager,
        result_formatter: ResultFormatter[OutputType],
        critic: Optional[CriticCore] = None,
    ) -> ChainResult[OutputType]:
        """
        Run with retries.

        Args:
            prompt: The prompt to process
            generator: The generator to use
            validation_manager: The validation manager to use
            prompt_manager: The prompt manager to use
            result_formatter: The result formatter to use
            critic: Optional critic to use

        Returns:
            The chain result

        Raises:
            ValueError: If validation fails after max attempts
        """
        pass

    def execute(self, operation: Any, *args: Any, **kwargs: Any) -> Any:
        """
        Execute an operation with retries.

        This method implements the RetryStrategyProtocol.execute method.

        Args:
            operation: The operation to execute
            *args: Positional arguments for the operation
            **kwargs: Keyword arguments for the operation

        Returns:
            The result of the operation

        Raises:
            RuntimeError: If the operation fails after all retries
        """
        attempt = 0
        while attempt < self._max_attempts:
            try:
                result = operation(*args, **kwargs)
                if not self.should_retry(attempt, result):
                    return result
            except Exception as e:
                if not self.should_retry(attempt, None, e):
                    raise RuntimeError(f"Operation failed after {attempt + 1} attempts") from e

            attempt += 1

        raise RuntimeError(f"Operation failed after {self._max_attempts} attempts")

    def should_retry(self, attempt: int, result: Any, error: Optional[Exception] = None) -> bool:
        """
        Check if an operation should be retried.

        This method implements the RetryStrategyProtocol.should_retry method.

        Args:
            attempt: The current attempt number
            result: The result of the operation
            error: The error that occurred, if any

        Returns:
            True if the operation should be retried, False otherwise
        """
        # If we've reached the maximum number of attempts, don't retry
        if attempt >= self._max_attempts - 1:
            return False

        # If there was an error, retry
        if error is not None:
            return True

        # If the result is None, retry
        if result is None:
            return True

        # Otherwise, don't retry
        return False


class SimpleRetryStrategy(RetryStrategy[OutputType]):
    """
    Simple retry strategy that retries a fixed number of times.

    This strategy retries a fixed number of times with feedback from
    validation failures.
    """

    def __init__(self, max_attempts: int = 3):
        """
        Initialize a SimpleRetryStrategy instance.

        Args:
            max_attempts: Maximum number of attempts
        """
        self._max_attempts = max_attempts

    @property
    def max_attempts(self) -> int:
        """
        Get the maximum number of attempts.

        Returns:
            The maximum number of attempts
        """
        return self._max_attempts

    def run(
        self,
        prompt: str,
        generator: Generator[OutputType],
        validation_manager: ValidationManager[OutputType],
        prompt_manager: PromptManager,
        result_formatter: ResultFormatter[OutputType],
        critic: Optional[CriticCore] = None,
    ) -> ChainResult[OutputType]:
        """
        Run with retries.

        Args:
            prompt: The prompt to process
            generator: The generator to use
            validation_manager: The validation manager to use
            prompt_manager: The prompt manager to use
            result_formatter: The result formatter to use
            critic: Optional critic to use

        Returns:
            The chain result

        Raises:
            ValueError: If validation fails after max attempts
            RuntimeError: If an unexpected condition occurs
        """
        attempts = 0
        current_prompt = prompt
        last_critique_details = None

        while attempts < self._max_attempts:
            # Generate output
            output = generator.generate(current_prompt)

            # Validate output
            validation_result = validation_manager.validate(output)

            # If validation passed, return result
            if validation_result.all_passed:
                return result_formatter.format_result(
                    output=output,
                    validation_result=validation_result,
                    critique_details=last_critique_details,
                )

            # If we're on the last attempt, raise error
            if attempts >= self._max_attempts - 1:
                error_messages = validation_manager.get_error_messages(validation_result)
                raise ValueError(
                    f"Validation failed after {attempts + 1} attempts. Errors:\n"
                    + "\n".join(error_messages)
                )

            # If we have a critic, use it to improve the output
            critique_details = None
            if critic:
                # Get feedback from the critic
                critique = critic.critique(output)

                # Process critique details
                if hasattr(critique, "__dict__"):
                    critique_details = critique.__dict__
                elif isinstance(critique, dict):
                    critique_details = critique

                # Store critique details for the result
                if critique_details:
                    last_critique_details = critique_details

                    # Get feedback from the critic
                    feedback = result_formatter.format_feedback_from_critique(critique_details)

                    # Update prompt with feedback
                    current_prompt = prompt_manager.create_prompt_with_feedback(prompt, feedback)
            else:
                # No critic, use validation result for feedback
                feedback = result_formatter.format_feedback_from_validation(validation_result)
                current_prompt = prompt_manager.create_prompt_with_feedback(prompt, feedback)

            attempts += 1

        # Should never reach here due to while loop condition
        raise RuntimeError("Unexpected end of retry execution")


class BackoffRetryStrategy(RetryStrategy[OutputType]):
    """
    Backoff retry strategy that increases the wait time between retries.

    This strategy increases the wait time between retries using an
    exponential backoff algorithm.
    """

    def __init__(
        self,
        max_attempts: int = 3,
        initial_backoff: float = 1.0,
        backoff_factor: float = 2.0,
        max_backoff: float = 60.0,
    ):
        """
        Initialize a BackoffRetryStrategy instance.

        Args:
            max_attempts: Maximum number of attempts
            initial_backoff: Initial backoff time in seconds
            backoff_factor: Factor to multiply backoff by each attempt
            max_backoff: Maximum backoff time in seconds
        """
        self._max_attempts = max_attempts
        self._initial_backoff = initial_backoff
        self._backoff_factor = backoff_factor
        self._max_backoff = max_backoff

    @property
    def max_attempts(self) -> int:
        """
        Get the maximum number of attempts.

        Returns:
            The maximum number of attempts
        """
        return self._max_attempts

    def run(
        self,
        prompt: str,
        generator: Generator[OutputType],
        validation_manager: ValidationManager[OutputType],
        prompt_manager: PromptManager,
        result_formatter: ResultFormatter[OutputType],
        critic: Optional[CriticCore] = None,
    ) -> ChainResult[OutputType]:
        """
        Run with retries.

        Args:
            prompt: The prompt to process
            generator: The generator to use
            validation_manager: The validation manager to use
            prompt_manager: The prompt manager to use
            result_formatter: The result formatter to use
            critic: Optional critic to use

        Returns:
            The chain result

        Raises:
            ValueError: If validation fails after max attempts
            RuntimeError: If an unexpected condition occurs
        """
        attempts = 0
        current_prompt = prompt
        last_critique_details = None
        backoff = self._initial_backoff

        while attempts < self._max_attempts:
            # Generate output
            output = generator.generate(current_prompt)

            # Validate output
            validation_result = validation_manager.validate(output)

            # If validation passed, return result
            if validation_result.all_passed:
                return result_formatter.format_result(
                    output=output,
                    validation_result=validation_result,
                    critique_details=last_critique_details,
                )

            # If we're on the last attempt, raise error
            if attempts >= self._max_attempts - 1:
                error_messages = validation_manager.get_error_messages(validation_result)
                raise ValueError(
                    f"Validation failed after {attempts + 1} attempts. Errors:\n"
                    + "\n".join(error_messages)
                )

            # If we have a critic, use it to improve the output
            critique_details = None
            if critic:
                # Get feedback from the critic
                critique = critic.critique(output)

                # Process critique details
                if hasattr(critique, "__dict__"):
                    critique_details = critique.__dict__
                elif isinstance(critique, dict):
                    critique_details = critique

                # Store critique details for the result
                if critique_details:
                    last_critique_details = critique_details

                    # Get feedback from the critic
                    feedback = result_formatter.format_feedback_from_critique(critique_details)

                    # Update prompt with feedback
                    current_prompt = prompt_manager.create_prompt_with_feedback(prompt, feedback)
            else:
                # No critic, use validation result for feedback
                feedback = result_formatter.format_feedback_from_validation(validation_result)
                current_prompt = prompt_manager.create_prompt_with_feedback(prompt, feedback)

            # Wait before next attempt
            logger.info(f"Waiting {backoff:.2f} seconds before next attempt")
            time.sleep(backoff)

            # Update backoff for next attempt
            backoff = min(backoff * self._backoff_factor, self._max_backoff)
            attempts += 1

        # Should never reach here due to while loop condition
        raise RuntimeError("Unexpected end of retry execution")
