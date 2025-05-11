"""
Retry Manager Module

This module provides a retry manager for the Sifaka chain system.
It handles retry logic for chain execution.

## Overview
The retry manager implements a standardized approach to handling retry logic
in the chain execution flow. It coordinates the generation, validation, and
improvement steps, automatically retrying when validations fail. This ensures
that the chain system can recover from temporary failures and improve outputs
that don't meet validation criteria.

## Components
1. **RetryManager**: Manages retry logic for chain execution with configurable
   maximum attempts and state tracking

## Usage Examples
```python
from sifaka.chain.managers.retry import RetryManager
from sifaka.utils.state import StateManager
from sifaka.chain.result import ChainResult

# Create retry manager
retry_manager = RetryManager(
    state_manager=StateManager(),
    max_attempts=3
)

# Define functions for the retry flow
def generate():
    return model.generate(prompt)

def validate(output):
    return [validator.validate(output) for validator in validators]

def improve(output, results):
    return improver.improve(output, results) if improver else output

def create_result(prompt, output, results, attempt):
    return ChainResult(
        prompt=prompt,
        output=output,
        validation_results=results,
        attempt_count=attempt
    )

# Execute with retries
result = retry_manager.execute_with_retries(
    generate_func=generate,
    validate_func=validate,
    improve_func=improve,
    prompt=prompt,
    create_result_func=create_result
)

# Get retry statistics
stats = retry_manager.get_retry_stats()
print(f"Attempts: {stats['current_attempt']}/{stats['max_attempts']}")
print(f"All validations passed: {stats['all_passed']}")
```

## Error Handling
The retry manager handles errors gracefully:
- Raises ChainError if execution fails after maximum attempts
- Tracks validation results for each attempt
- Returns the best result even if validations fail on the last attempt

## Configuration
The retry manager can be configured with the following options:
- max_attempts: Maximum number of retry attempts (default: 3)
"""

from typing import Callable, List, Optional, Any
import time

from ...utils.state import StateManager
from ...utils.logging import get_logger
from ...utils.errors import ChainError
from ..interfaces import ValidationResult
from ..result import ChainResult

# Configure logger
logger = get_logger(__name__)


class RetryManager:
    """
    Manages retry logic for chain execution.

    This class provides a standardized way to handle retry logic in the chain
    execution flow. It coordinates the generation, validation, and improvement
    steps, automatically retrying when validations fail. This ensures that the
    chain system can recover from temporary failures and improve outputs that
    don't meet validation criteria.

    ## Architecture
    The RetryManager uses a functional approach to retry logic, accepting
    callback functions for generation, validation, improvement, and result
    creation. This design allows for flexible integration with different
    components while maintaining a consistent retry flow.

    ## Lifecycle
    1. **Initialization**: Set up retry manager with state manager and configuration
    2. **Execution**: Execute the generation-validation-improvement flow with retries
    3. **Result Handling**: Return the final result after successful validation or max attempts
    4. **Statistics Tracking**: Track retry statistics for monitoring and debugging

    ## Error Handling
    The RetryManager handles errors gracefully:
    - Raises ChainError if execution fails after maximum attempts
    - Tracks validation results for each attempt
    - Returns the best result even if validations fail on the last attempt

    ## Examples
    ```python
    # Create retry manager
    retry_manager = RetryManager(
        state_manager=StateManager(),
        max_attempts=3
    )

    # Execute with retries
    result = retry_manager.execute_with_retries(
        generate_func=lambda: model.generate(prompt),
        validate_func=lambda output: [validator.validate(output) for validator in validators],
        improve_func=lambda output, results: improver.improve(output, results),
        prompt=prompt,
        create_result_func=create_result
    )
    ```
    """

    def __init__(
        self,
        state_manager: StateManager,
        max_attempts: int = 3,
    ):
        """
        Initialize the retry manager.

        This method initializes the retry manager with the provided state manager
        and configuration options. It sets up the initial state and metadata.

        Args:
            state_manager (StateManager): State manager for state management
            max_attempts (int, optional): Maximum number of retry attempts. Defaults to 3.

        Raises:
            None: This method does not raise exceptions

        Example:
            ```python
            from sifaka.chain.managers.retry import RetryManager
            from sifaka.utils.state import StateManager

            # Create retry manager
            retry_manager = RetryManager(
                state_manager=StateManager(),
                max_attempts=3
            )
            ```
        """
        self._state_manager = state_manager
        self._max_attempts = max_attempts

        # Set metadata
        self._state_manager.set_metadata("component_type", "retry_manager")
        self._state_manager.set_metadata("creation_time", time.time())
        self._state_manager.set_metadata("max_attempts", max_attempts)

    @property
    def max_attempts(self) -> int:
        """
        Get the maximum number of retry attempts.

        Returns:
            int: The maximum number of retry attempts

        Example:
            ```python
            print(f"Maximum retry attempts: {retry_manager.max_attempts}")
            ```
        """
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

        This method executes the generation-validation-improvement flow with retries.
        It calls the generate function to create output, validates it using the validate
        function, and if validation fails, it calls the improve function to refine the
        output and tries again. This process repeats until either validation passes or
        the maximum number of attempts is reached.

        Args:
            generate_func (Callable[[], str]): Function to generate output
            validate_func (Callable[[str], List[ValidationResult]]): Function to validate output
            improve_func (Callable[[str, List[ValidationResult]], str]): Function to improve output
            prompt (str): The prompt to process
            create_result_func (Callable[[str, str, List[ValidationResult], int], ChainResult]):
                Function to create the result

        Returns:
            ChainResult: The chain result after successful validation or max attempts

        Raises:
            ChainError: If execution fails after max attempts

        Example:
            ```python
            result = retry_manager.execute_with_retries(
                generate_func=lambda: model.generate(prompt),
                validate_func=lambda output: [validator.validate(output) for validator in validators],
                improve_func=lambda output, results: improver.improve(output, results),
                prompt=prompt,
                create_result_func=lambda p, o, r, a: ChainResult(
                    prompt=p, output=o, validation_results=r, attempt_count=a
                )
            )
            ```
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

        This method returns a dictionary with various retry statistics,
        including the maximum number of attempts, the current attempt,
        and whether all validations passed.

        Returns:
            dict: Dictionary with retry statistics including:
                - max_attempts: Maximum number of retry attempts
                - current_attempt: Current attempt number
                - all_passed: Whether all validations passed

        Example:
            ```python
            stats = retry_manager.get_retry_stats()
            print(f"Attempts: {stats['current_attempt']}/{stats['max_attempts']}")
            print(f"All validations passed: {stats['all_passed']}")
            ```
        """
        return {
            "max_attempts": self._max_attempts,
            "current_attempt": self._state_manager.get("attempt", 0),
            "all_passed": self._state_manager.get("all_passed", False),
        }
