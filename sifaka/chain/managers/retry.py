from typing import Any

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
from sifaka.core.results import ChainResult

# Create retry manager
retry_manager = RetryManager(max_attempts=3)

# Define functions for the retry flow
def generate():
    return model.generate(prompt)

def validate(output):
    return [validator.validate(output) for validator in validators]

def improve(output, results):
    return improver.improve(output, results)

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

from typing import Callable, Dict, List, Optional, Any, Union
import time
from pydantic import PrivateAttr
from ...utils.state import StateManager, create_manager_state
from ...utils.logging import get_logger
from ...utils.errors import ChainError
from sifaka.interfaces.chain.models import ValidationResult
from sifaka.models.result import GenerationResult
from ...core.results import ChainResult

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
    retry_manager = RetryManager(max_attempts=3)

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

    _state_manager: StateManager = PrivateAttr(default_factory=create_manager_state)

    def __init__(self, max_attempts: int = 3, state_manager: Optional[StateManager] = None) -> None:
        """
        Initialize the retry manager.

        This method initializes the retry manager with the provided configuration
        options. It sets up the initial state and metadata.

        Args:
            max_attempts (int, optional): Maximum number of retry attempts. Defaults to 3.
            state_manager (Optional[StateManager], optional): State manager for state management.
                If None, a new state manager will be created. Defaults to None.

        Raises:
            None: This method does not raise exceptions

        Example:
            ```python
            from sifaka.chain.managers.retry import RetryManager

            # Create retry manager
            retry_manager = RetryManager(max_attempts=3)
            ```
        """
        self._max_attempts = max_attempts

        # Support both dependency injection and auto-creation patterns
        if state_manager is not None:
            object.__setattr__(self, "_state_manager", state_manager)

        self._initialize_state()

    def _initialize_state(self) -> None:
        """Initialize the retry manager state."""
        # Call super to ensure proper initialization of base state
        super()._initialize_state()

        self._state_manager.update("initialized", True)
        self._state_manager.set_metadata("component_type", "retry_manager")
        self._state_manager.set_metadata("creation_time", time.time())
        self._state_manager.set_metadata("max_attempts", self._max_attempts)

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
        generate_func: Callable[[], Union[str, GenerationResult]],
        validate_func: Callable[[Union[str, GenerationResult]], List[ValidationResult]],
        improve_func: Callable[
            [Union[str, GenerationResult], List[ValidationResult]], Union[str, GenerationResult]
        ],
        prompt: str,
        create_result_func: Callable[
            [str, Union[str, GenerationResult], List[ValidationResult], int], ChainResult
        ],
    ) -> ChainResult:
        """
        Execute with retries.

        This method executes the generation-validation-improvement flow with retries.
        It calls the generate function to create output, validates it using the validate
        function, and if validation fails, it calls the improve function to refine the
        output and tries again. This process repeats until either validation passes or
        the maximum number of attempts is reached.

        Args:
            generate_func (Callable[[], Union[str, GenerationResult]]): Function to generate output
            validate_func (Callable[[Union[str, GenerationResult]], List[ValidationResult]]): Function to validate output
            improve_func (Callable[[Union[str, GenerationResult], List[ValidationResult]], Union[str, GenerationResult]]): Function to improve output
            prompt (str): The prompt to process
            create_result_func (Callable[[str, Union[str, GenerationResult], List[ValidationResult], int], ChainResult]):
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
            self._state_manager.update("attempt", attempt)
            logger.debug(f"Attempt {attempt}/{self._max_attempts}")
            output = generate_func()
            self._state_manager.update("output", output)
            validation_results = validate_func(output)
            self._state_manager.update("validation_results", validation_results)
            all_passed = all(r.passed for r in validation_results)
            self._state_manager.update("all_passed", all_passed)
            if all_passed:
                return create_result_func(prompt, output, validation_results, attempt)
            if attempt < self._max_attempts:
                improved_output = improve_func(output, validation_results)
                output = improved_output
                self._state_manager.update("output", output)
            else:
                return create_result_func(prompt, output, validation_results, attempt)
        raise ChainError(f"Execution failed after {self._max_attempts} attempts")

    def get_retry_stats(self) -> Dict[str, Any]:
        """
        Get retry statistics.

        This method returns a dictionary with various retry statistics,
        including the maximum number of attempts, the current attempt,
        and whether all validations passed.

        Returns:
            Dict[str, Any]: Dictionary with retry statistics including:
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
