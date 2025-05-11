"""
Retry Manager Module

This module provides a retry manager for the Sifaka chain system.
It handles retry logic for chain execution.

## Components
1. **RetryManager**: Manages retry logic for chain execution

## Usage Examples
```python
from sifaka.chain.managers.retry import RetryManager
from sifaka.utils.state import StateManager

# Create retry manager
retry_manager = RetryManager(
    state_manager=StateManager(),
    max_attempts=3
)

# Execute with retries
result = retry_manager.execute_with_retries(
    generate_func=lambda: model.generate(prompt),
    validate_func=lambda output: [validator.validate(output) for validator in validators],
    improve_func=lambda output, results: improver.improve(output, results) if improver else output,
    prompt=prompt
)
```
"""

from typing import Callable, List, Optional, Any
import time

from ...utils.state import StateManager
from ...utils.logging import get_logger
from ..interfaces import ValidationResult
from ..result import ChainResult
from ..errors import ChainError

# Configure logger
logger = get_logger(__name__)


class RetryManager:
    """Manages retry logic for chain execution."""

    def __init__(
        self,
        state_manager: StateManager,
        max_attempts: int = 3,
    ):
        """
        Initialize the retry manager.

        Args:
            state_manager: State manager for state management
            max_attempts: Maximum number of retry attempts
        """
        self._state_manager = state_manager
        self._max_attempts = max_attempts

        # Set metadata
        self._state_manager.set_metadata("component_type", "retry_manager")
        self._state_manager.set_metadata("creation_time", time.time())
        self._state_manager.set_metadata("max_attempts", max_attempts)

    @property
    def max_attempts(self) -> int:
        """Get the maximum number of retry attempts."""
        return self._max_attempts

    def execute_with_retries(
        self,
        generate_func: Callable[[], str],
        validate_func: Callable[[str], List[ValidationResult]],
        improve_func: Callable[[str, List[ValidationResult]], str],
        prompt: str,
        create_result_func: Callable[[str, str, List[ValidationResult], int], ChainResult],
    ) -> ChainResult:
        """
        Execute with retries.

        Args:
            generate_func: Function to generate output
            validate_func: Function to validate output
            improve_func: Function to improve output
            prompt: The prompt to process
            create_result_func: Function to create the result

        Returns:
            The chain result

        Raises:
            ChainError: If execution fails after max attempts
        """
        for attempt in range(1, self._max_attempts + 1):
            # Update attempt counter
            self._state_manager.update("attempt", attempt)
            logger.debug(f"Attempt {attempt}/{self._max_attempts}")

            # Generate output
            output = generate_func()
            self._state_manager.update("output", output)

            # Validate output
            validation_results = validate_func(output)
            self._state_manager.update("validation_results", validation_results)

            # Check if all validations passed
            all_passed = all(r.passed for r in validation_results)
            self._state_manager.update("all_passed", all_passed)

            # If all validations passed, return result
            if all_passed:
                return create_result_func(prompt, output, validation_results, attempt)

            # If not last attempt, improve output and retry
            if attempt < self._max_attempts:
                output = improve_func(output, validation_results)
                self._state_manager.update("output", output)
            else:
                # Last attempt, return result even if validations failed
                return create_result_func(prompt, output, validation_results, attempt)

        # Should never reach here, but just in case
        raise ChainError(f"Execution failed after {self._max_attempts} attempts")

    def get_retry_stats(self) -> dict:
        """
        Get retry statistics.

        Returns:
            Dictionary with retry statistics
        """
        return {
            "max_attempts": self._max_attempts,
            "current_attempt": self._state_manager.get("attempt", 0),
            "all_passed": self._state_manager.get("all_passed", False),
        }
