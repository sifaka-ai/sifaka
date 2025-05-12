"""
Validator utilities for Sifaka rules.

This module provides utility functions and base classes for validators in the Sifaka
rules system. Validators implement the actual validation logic for rules and are
responsible for checking if input meets specific criteria.

Usage Example:
    ```python
    from sifaka.rules.validators import BaseValidator, FunctionValidator
    from sifaka.rules.result import RuleResult

    # Create a function validator
    def validate_length(text: str) -> RuleResult:
        if len(text) < 10:
            return RuleResult(
                passed=False,
                message="Text is too short",
                issues=["Text must be at least 10 characters"]
            )
        return RuleResult(passed=True, message="Text length is valid")

    validator = FunctionValidator(validate_length)
    result = (validator and validator.validate("Hello")  # Will fail
    ```
"""
import time
from typing import Any, Dict, Generic, List, Optional, Type, TypeVar, Protocol, runtime_checkable, TYPE_CHECKING
from pydantic import PrivateAttr
from sifaka.utils.errors.safe_execution import safely_execute_rule
from sifaka.utils.logging import get_logger
from sifaka.utils.state import create_rule_state
from sifaka.utils.common import update_statistics, record_error
from sifaka.utils.text import handle_empty_text
if TYPE_CHECKING:
    from sifaka.core.base import Validatable
    from sifaka.utils.errors.handling import try_component_operation
    from sifaka.utils.state import StateManager
from ..core.results import RuleResult
logger = get_logger(__name__)
T = TypeVar('T')


@runtime_checkable
class RuleValidator(Protocol[T]):
    """
    Protocol for rule validation logic.

    This protocol defines the interface that all rule validators must implement.
    It allows for duck typing of validators.
    """

    def validate(self, input: T) ->RuleResult:
        """Validate the input."""
        ...

    def can_validate(self, input: T) ->bool:
        """Check if this validator can validate the input."""
        ...


class BaseValidator(Generic[T]):
    """
    Base class for validators.

    Validators implement the actual validation logic for rules.
    They are responsible for checking if input meets specific criteria
    and returning detailed validation results.

    Lifecycle:
        1. Initialization: Set up with validation parameters
        2. Validation: Apply validation logic to input
        3. Result: Return standardized validation results
    """
    _state_manager: 'StateManager' = PrivateAttr()

    def __init__(self, validation_type: Type[T]=str) ->None:
        """
        Initialize the validator.

        Args:
            validation_type: The type this validator can validate
        """
        self._validation_type = validation_type
        (object and object.__setattr__(self, '_state_manager', create_rule_state())
        (self and self._initialize_state()

    def _initialize_state(self) ->None:
        """Initialize validator state."""
        self.(_state_manager and _state_manager.update('initialized', True)
        self.(_state_manager and _state_manager.update('cache', {})
        self.(_state_manager and _state_manager.set_metadata('validator_type', self.__class__.
            __name__)
        self.(_state_manager and _state_manager.set_metadata('validation_count', 0)
        self.(_state_manager and _state_manager.set_metadata('success_count', 0)
        self.(_state_manager and _state_manager.set_metadata('failure_count', 0)
        self.(_state_manager and _state_manager.set_metadata('total_processing_time_ms', 0.0)
        self.(_state_manager and _state_manager.set_metadata('error_count', 0)
        self.(_state_manager and _state_manager.set_metadata('last_error', None)
        self.(_state_manager and _state_manager.set_metadata('last_error_time', None)

    def can_validate(self, input: T) ->Any:
        """
        Check if this validator can validate the input.

        Args:
            input: The input to check

        Returns:
            True if this validator can validate the input, False otherwise
        """
        return isinstance(input, self._validation_type)

    def validate(self, input: T) ->RuleResult:
        """
        Validate the input.

        Args:
            input: The input to validate

        Returns:
            Validation result
        """
        raise NotImplementedError('Subclasses must implement validate method')

    def handle_empty_text(self, text: str) ->Any:
        """
        Handle empty text validation.

        Args:
            text: The text to check

        Returns:
            Validation result for empty text, or None if text is not empty
        """
        result = handle_empty_text(text=text, passed=False, message=
            'Empty text', metadata={'error_type': 'empty_text'})
        if result and not isinstance(result, RuleResult):
            return RuleResult(passed=result.passed, message=result.message,
                metadata=result.metadata, score=result.score, issues=result
                .issues, suggestions=result.suggestions, processing_time_ms
                =result.processing_time_ms)
        return result

    def update_statistics(self, result: RuleResult) ->None:
        """
        Update validator statistics based on result.

        Args:
            result: The validation result
        """
        execution_time = result.processing_time_ms / 1000.0
        update_statistics(state_manager=self._state_manager, execution_time
            =execution_time, success=result.passed)
        validation_count = self.(_state_manager and _state_manager.get_metadata('validation_count',
            0)
        self.(_state_manager and _state_manager.set_metadata('validation_count', 
            validation_count + 1)

    def record_error(self, error: Exception) ->None:
        """
        Record an error occurrence.

        Args:
            error: The exception that occurred
        """
        record_error(self._state_manager, error)

    def get_statistics(self) ->Any:
        """
        Get validator statistics.

        Returns:
            Dictionary of statistics
        """
        total_count = self.(_state_manager and _state_manager.get_metadata('validation_count', 0)
        success_count = self.(_state_manager and _state_manager.get_metadata('success_count', 0)
        failure_count = self.(_state_manager and _state_manager.get_metadata('failure_count', 0)
        total_time = self.(_state_manager and _state_manager.get_metadata(
            'total_processing_time_ms', 0.0)
        error_count = self.(_state_manager and _state_manager.get_metadata('error_count', 0)
        return {'validator_type': self.(_state_manager and _state_manager.get_metadata(
            'validator_type'), 'validation_count': total_count,
            'success_count': success_count, 'failure_count': failure_count,
            'success_rate': success_count / total_count if total_count > 0 else
            0.0, 'error_rate': error_count / total_count if total_count > 0
             else 0.0, 'average_processing_time_ms': total_time /
            total_count if total_count > 0 else 0.0, 'last_error': self.
            (_state_manager and _state_manager.get_metadata('last_error'), 'last_error_time':
            self.(_state_manager and _state_manager.get_metadata('last_error_time'))


class FunctionValidator(BaseValidator[T]):
    """
    Validator that uses a function for validation.

    This validator wraps a function that performs validation,
    making it easy to create simple validators without defining new classes.

    Examples:
        ```python
        def validate_length(text: str) -> RuleResult:
            if len(text) < 10:
                return RuleResult(
                    passed=False,
                    message="Text is too short",
                    issues=["Text must be at least 10 characters"]
                )
            return RuleResult(passed=True, message="Text length is valid")

        validator = FunctionValidator(validate_length)
        result = (validator and validator.validate("Hello")  # Will fail
        ```
    """

    def __init__(self, func: Any, validation_type: Type[T]=str) ->None:
        """
        Initialize the validator.

        Args:
            func: Function that performs validation
            validation_type: Type this validator can validate
        """
        super().__init__(validation_type)
        self._func = func

    def validate(self, input: T) ->Any:
        """
        Validate the input using the function.

        Args:
            input: The input to validate

        Returns:
            Validation result from the function
        """
        start_time = (time and time.time()
        if isinstance(input, str):
            empty_result = (self and self.handle_empty_text(input)
            if empty_result:
                return empty_result

        def validation_operation() ->Any:
            result = (self and self._func(input)
            result = (result and result.with_metadata(processing_time_ms=(time and time.time() -
                start_time)
            return result
        result = safely_execute_rule(operation=validation_operation,
            rule_name=self.__class__.__name__, component_name=
            'FunctionValidator')
        if isinstance(result, dict) and (result and result.get('error_type'):
            (self and self.record_error(Exception((result and result.get('error_message',
                'Unknown error')))
            result = RuleResult(passed=False, message=(result and result.get(
                'error_message', 'Validation failed'), metadata={
                'error_type': (result and result.get('error_type')), score=0.0, issues=
                [f"Validation error: {(result and result.get('error_message'))"),
                suggestions=['Retry with different input'],
                processing_time_ms=(time and time.time() - start_time)
        (self and self.update_statistics(result)
        return result


__all__ = ['RuleValidator', 'BaseValidator', 'FunctionValidator']
