"""
Validator Adapter Module

This module provides the ValidatorAdapter class for adapting existing rules
to the Validator interface from the chain system.
"""

import time
from typing import Any, List, Optional, Union, cast
from sifaka.interfaces.chain.components import Validator
from sifaka.interfaces.chain.models import ValidationResult
from sifaka.utils.errors.component import ValidationError
from sifaka.utils.errors.results import ErrorResult
from sifaka.utils.errors.safe_execution import safely_execute_component_operation
from sifaka.utils.state import create_adapter_state


class ValidatorAdapter(Validator):
    """
    Adapter for existing rules.

    This adapter implements the Validator interface for existing rules,
    using the standardized state management pattern.

    ## Architecture
    The ValidatorAdapter follows the adapter pattern to wrap existing rules:
    - Implements the Validator interface from chain.interfaces
    - Uses standardized state management with _state_manager
    - Delegates to the wrapped rule
    - Handles different rule interfaces (validate, process, run)
    - Converts rule results to ValidationResult objects
    - Provides consistent error handling and statistics tracking

    ## Lifecycle
    1. **Initialization**: Adapter is created with a rule
    2. **State Setup**: State manager is initialized with adapter state
    3. **Operation**: Adapter delegates to the rule
    4. **Cleanup**: Resources are released when no longer needed

    ## Error Handling
    - ValidationError: Raised when validation fails
    - Tracks error statistics in state manager
    - Provides detailed error messages with component information

    Attributes:
        _validator (Any): The wrapped rule
        _name (str): The name of the adapter
        _description (str): The description of the adapter
        _state_manager (StateManager): The state manager for the adapter
    """

    def __init__(
        self, validator: Any, name: Optional[str] = None, description: Optional[str] = None
    ) -> None:
        """
        Initialize the validator adapter.

        Args:
            validator: The rule to adapt
            name: Optional name for the adapter
            description: Optional description for the adapter
        """
        self._validator = validator
        self._name = name or f"{type(validator).__name__}Adapter"
        self._description = description or f"Adapter for {type(validator).__name__}"
        self._state_manager = create_adapter_state()
        self._initialize_state()

    def _initialize_state(self) -> None:
        """Initialize adapter state."""
        self._state_manager.update("adaptee", self._validator)
        self._state_manager.update("initialized", True)
        self._state_manager.update("cache", {})
        self._state_manager.set_metadata("component_type", "validator_adapter")
        self._state_manager.set_metadata("adaptee_type", type(self._validator).__name__)
        self._state_manager.set_metadata("creation_time", time.time())

    @property
    def name(self) -> str:
        """Get adapter name."""
        return self._name

    @property
    def description(self) -> str:
        """Get adapter description."""
        return self._description

    def validate(self, output: str) -> ValidationResult:
        """
        Validate an output.

        Args:
            output: The output to validate

        Returns:
            The validation result

        Raises:
            ValidationError: If validation fails
        """
        if not self._state_manager.get("initialized", False):
            self._initialize_state()

        start_time = time.time()
        try:

            def validate_operation() -> ValidationResult:
                if hasattr(self._validator, "validate"):
                    result = self._validator.validate(output)
                elif hasattr(self._validator, "process"):
                    result = self._validator.process(output)
                elif hasattr(self._validator, "run"):
                    result = self._validator.run(output)
                else:
                    raise ValidationError(
                        f"Unsupported validator: {type(self._validator).__name__}"
                    )
                return self._convert_result(result)

            result_or_error = safely_execute_component_operation(
                operation=validate_operation,
                component_name=self.name,
                component_type="Validator",
                error_class=ValidationError,
            )

            if isinstance(result_or_error, ErrorResult):
                # Convert ErrorResult to ValidationResult with failed status
                result = ValidationResult(
                    passed=False,
                    message=f"Validation error: {result_or_error.error_message}",
                    score=0.0,
                    issues=[result_or_error.error_message],
                    suggestions=[],
                    metadata=result_or_error.metadata,
                )
            else:
                result = result_or_error

            # Ensure result is a ValidationResult
            validation_result = self._convert_result(result)

            end_time = time.time()
            execution_time = end_time - start_time
            validation_count = self._state_manager.get_metadata("validation_count", 0)
            self._state_manager.set_metadata("validation_count", validation_count + 1)

            if validation_result.passed:
                success_count = self._state_manager.get_metadata("success_count", 0)
                self._state_manager.set_metadata("success_count", success_count + 1)
            else:
                failure_count = self._state_manager.get_metadata("failure_count", 0)
                self._state_manager.set_metadata("failure_count", failure_count + 1)

            avg_time = self._state_manager.get_metadata("avg_execution_time", 0)
            new_avg = (avg_time * validation_count + execution_time) / (validation_count + 1)
            self._state_manager.set_metadata("avg_execution_time", new_avg)
            max_time = self._state_manager.get_metadata("max_execution_time", 0)

            if execution_time > max_time:
                self._state_manager.set_metadata("max_execution_time", execution_time)

            if self._state_manager.get("cache_enabled", True):
                cache = self._state_manager.get("cache", {})
                cache_key = f"{output[:50]}_{len(output)}"
                cache[cache_key] = validation_result
                self._state_manager.update("cache", cache)

            return validation_result

        except Exception as e:
            error_count = self._state_manager.get_metadata("error_count", 0)
            self._state_manager.set_metadata("error_count", error_count + 1)
            self._state_manager.set_metadata("last_error", str(e))
            self._state_manager.set_metadata("last_error_time", time.time())

            if isinstance(e, ValidationError):
                raise e
            raise ValidationError(f"Validation failed: {str(e)}")

    def _convert_result(self, result: Union[Any, ErrorResult]) -> ValidationResult:
        """
        Convert a rule result to a ValidationResult.

        Args:
            result: The rule result to convert

        Returns:
            The converted ValidationResult
        """
        # Handle the case where result is already a ValidationResult
        if isinstance(result, ValidationResult):
            return result

        # Handle the case where result is an ErrorResult
        if isinstance(result, ErrorResult):
            return ValidationResult(
                passed=False,
                message=result.error_message,
                score=0.0,
                issues=[result.error_message],
                suggestions=[],
                metadata={"error_type": result.error_type, **result.metadata},
            )

        # Handle other result types
        passed = getattr(result, "passed", False)
        message = getattr(result, "message", "")
        score = getattr(result, "score", 0.0)
        issues = getattr(result, "issues", [])
        suggestions = getattr(result, "suggestions", [])
        metadata = getattr(result, "metadata", {})

        return ValidationResult(
            passed=passed,
            message=message,
            score=score,
            issues=issues,
            suggestions=suggestions,
            metadata=metadata,
        )
