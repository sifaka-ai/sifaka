"""
Chain Engine Module

A module providing the core execution engine for the Sifaka chain system.

## Overview
This module provides the core execution engine for the Sifaka chain system,
coordinating the flow between components, handling retries, and managing state.
The Engine class is responsible for the actual execution of the chain, including
generating output, validating it, improving it if necessary, and creating the
final result.

## Components
- Engine: Core execution engine that coordinates the flow
- CacheManager: Manager for caching results
- RetryManager: Manager for handling retries
- Model: Interface for text generation models
- Validator: Interface for output validators
- Improver: Interface for output improvers
- Formatter: Interface for result formatters

## Usage Examples
```python
# Direct import of Engine class
from sifaka.chain.engine import Engine
# Import the configuration class
from sifaka.utils.config.chain import EngineConfig

# Create engine
engine = Engine(config=EngineConfig(max_attempts=3))

# Run engine
result = engine.run(
    prompt="Write a story",
    model=model,
    validators=validators,
    improver=improver,
    formatter=formatter
)

# Access result
print(f"Output: {result.output}")
print(f"All validations passed: {result.all_passed}")
print(f"Execution time: {result.execution_time:.2f}s")
```

## Error Handling
The module handles various error conditions:
- ChainError: Raised when chain execution fails
- ModelError: Raised when model generation fails
- ValidationError: Raised when validation fails
- ImproverError: Raised when improver refinement fails
- FormatterError: Raised when formatter formatting fails

## Execution Flow
1. Check cache for existing result
2. Generate output using model
3. Validate output using validators
4. Improve output using improver if validation fails
5. Format result using formatter
6. Cache result for future use
7. Return result
"""

from typing import List, Optional, Union, Callable, Any, Dict, Type, TypeVar, cast
import time
from pydantic import BaseModel, PrivateAttr, Field, ConfigDict

# Use new interface paths
from sifaka.interfaces import (
    ModelProtocol as Model,
    ValidatorProtocol as Validator,
    ImproverProtocol as Improver,
    FormatterProtocol as Formatter,
    ValidationResult,
)

from sifaka.utils.state import StateManager, create_engine_state
from sifaka.utils.logging import get_logger
from sifaka.core.results import ChainResult
from sifaka.utils.config.chain import EngineConfig
from sifaka.utils.errors import ChainError, safely_execute_chain
from sifaka.utils.errors.component import ModelError
from sifaka.utils.errors.results import ErrorResult
from sifaka.utils.mixins import InitializeStateMixin
from .managers.cache import CacheManager
from .managers.retry import RetryManager
from sifaka.models.result import GenerationResult

logger = get_logger(__name__)


class Engine(InitializeStateMixin, BaseModel):
    """
    Core execution engine for the Sifaka chain system.

    This class provides the core execution logic for the Sifaka chain system,
    coordinating the flow between components, handling retries, and managing state.
    It is responsible for generating output, validating it, improving it if necessary,
    and creating the final result.

    ## Architecture
    The Engine class follows a component-based architecture:
    - Uses Pydantic for data validation
    - Uses StateManager for state management
    - Uses CacheManager for result caching
    - Uses RetryManager for handling retries
    - Coordinates between Model, Validator, Improver, and Formatter components

    ## Lifecycle
    1. **Initialization**: Set up engine resources and configuration
    2. **Execution**: Run inputs through the flow
    3. **Result Handling**: Process and return results
    4. **State Management**: Manage engine state
    5. **Error Handling**: Handle and track errors
    6. **Execution Tracking**: Track execution statistics

    ## Examples
    ```python
    # Create engine
    engine = Engine(config=EngineConfig(max_attempts=3))

    # Run engine
    result = engine.run(
        prompt="Write a story",
        model=model,
        validators=validators,
        improver=improver,
        formatter=formatter
    )
    ```

    Attributes:
        config (EngineConfig): Engine configuration
    """

    _state_manager: StateManager = PrivateAttr(default_factory=create_engine_state)
    config: EngineConfig = EngineConfig()
    _cache_manager: CacheManager = PrivateAttr()
    _retry_manager: RetryManager = PrivateAttr()
    _execution_start_time: float = PrivateAttr(default=0.0)
    _execution_time: float = PrivateAttr(default=0.0)

    @property
    def execution_start_time(self) -> float:
        """Get the execution start time."""
        return self._execution_start_time

    @execution_start_time.setter
    def execution_start_time(self, value: float) -> None:
        """Set the execution start time."""
        self._execution_start_time = value

    @property
    def execution_time(self) -> float:
        """Get the execution time."""
        return self._execution_time

    @execution_time.setter
    def execution_time(self, value: float) -> None:
        """Set the execution time."""
        self._execution_time = value

    def __init__(
        self, config: Optional[EngineConfig] = None, state_manager: Optional[StateManager] = None
    ) -> None:
        """
        Initialize the engine.

        Args:
            config: Engine configuration
            state_manager: Optional state manager for state management. If None, a new one will be created.
        """
        super().__init__(config=config or EngineConfig())

        # Support both dependency injection and auto-creation patterns
        if state_manager is not None:
            object.__setattr__(self, "_state_manager", state_manager)

        self._initialize_state()

        # Create managers
        self._cache_manager = CacheManager(
            cache_enabled=self.config.params.get("cache_enabled", True),
            cache_size=self.config.params.get("cache_size", 100),
        )

        self._retry_manager = RetryManager(max_attempts=self.config.max_attempts)

    def _initialize_state(self) -> None:
        """Initialize the engine state."""
        # Check if super has _initialize_state method before calling it
        if hasattr(super(), "_initialize_state"):
            super()._initialize_state()

        self._state_manager.update("config", self.config)
        self._state_manager.update("initialized", True)
        self._state_manager.update("execution_count", 0)
        self._state_manager.update("cache", {})
        self._state_manager.set_metadata("component_type", self.__class__.__name__)
        self._state_manager.set_metadata("creation_time", time.time())

    def execute(self, prompt: str) -> ChainResult:
        """Execute the chain with the given prompt.

        Args:
            prompt: The prompt to use

        Returns:
            The chain result

        Raises:
            ChainError: If the chain execution fails
        """
        # Record the start time
        self.execution_start_time = time.time()

        # Create the retry manager
        retry_manager = RetryManager(
            max_attempts=self.config.max_attempts,
        )

        # Define the generate function to use with retry manager
        def generate_func() -> Union[str, GenerationResult]:
            return self._generate_output(prompt)

        # Define the validation function to use with retry manager
        def validate_func(output: Union[str, GenerationResult]) -> List[ValidationResult]:
            return self._validate_output(output)

        # Define the improve function to use with retry manager
        def improve_func(
            output: Union[str, GenerationResult], validation_results: List[ValidationResult]
        ) -> Union[str, GenerationResult]:
            # Cast to string before improving since the improver expects a string
            output_str = output.output if isinstance(output, GenerationResult) else output
            improved = self._improve_output(output_str, validation_results)
            return improved

        # Define the create result function to use with retry manager
        def create_result_func(
            prompt: str,
            output: Union[str, GenerationResult],
            validation_results: List[ValidationResult],
            attempt_count: int,
        ) -> ChainResult:
            return self._create_result(prompt, output, validation_results, attempt_count)

        # Execute with retries
        try:
            result = retry_manager.execute_with_retries(
                generate_func=generate_func,
                validate_func=validate_func,
                improve_func=improve_func,
                prompt=prompt,
                create_result_func=create_result_func,
            )

            # Record the execution time
            self.execution_time = time.time() - self.execution_start_time

            return result
        except Exception as e:
            raise ChainError(f"Chain execution failed: {str(e)}") from e

    def _generate_output(self, prompt: str) -> Union[str, GenerationResult]:
        """
        Generate output using the model.

        This method generates output using the model component, handling
        any errors that may occur during generation.

        Args:
            prompt (str): The prompt to generate from

        Returns:
            str: The generated output

        Raises:
            ModelError: If model generation fails

        Example:
            ```python
            output = engine._generate_output("Write a story about a robot") if engine else ""
            ```
        """
        model = self._state_manager.get("model")

        def generate_operation() -> Union[str, GenerationResult]:
            generated_output = model.generate(prompt)
            if not isinstance(generated_output, (str, GenerationResult)):
                raise ModelError(f"Model generated unexpected type: {type(generated_output)}")
            return generated_output

        additional_metadata = {"method": "generate", "prompt_length": len(prompt)}

        result = safely_execute_chain(
            generate_operation,
            "model",
            None,  # default_result
            "error",  # log_level
            True,  # include_traceback
            additional_metadata,
        )

        # Handle the case where the result is an ErrorResult
        if isinstance(result, ErrorResult):
            raise ModelError(f"Model generation failed: {result.error_message}")

        if isinstance(result, (str, GenerationResult)):
            return result
        # This should never happen, but we need to satisfy the type checker
        raise ModelError(f"Unexpected result type from generate_operation: {type(result)}")

    def _validate_output(self, output: Union[str, GenerationResult]) -> List[ValidationResult]:
        """Validate the output using all validators.

        Args:
            output: The output to validate

        Returns:
            The validation results
        """
        # Use the text for validation
        validation_text = output.output if isinstance(output, GenerationResult) else output

        # Validate using each validator
        results = []
        validators = self._state_manager.get("validators", [])
        for validator in validators:
            result = validator.validate(validation_text)
            # Convert from interface ValidationResult to core ValidationResult if needed
            if not isinstance(result, ValidationResult):
                # Convert the result
                core_result = ValidationResult(
                    passed=result.passed,
                    message=result.message,
                    score=result.score,
                    issues=result.issues,
                    suggestions=result.suggestions,
                    metadata=result.metadata,
                )
                results.append(core_result)
            else:
                results.append(result)

        # Log the validation results
        self._log_validation_results(results, validation_text)

        return results

    def _log_validation_results(self, results: List[ValidationResult], text: str) -> None:
        """Log validation results.

        Args:
            results: Validation results
            text: The validated text
        """
        logger = get_logger(self.__class__.__name__)
        for i, result in enumerate(results):
            status = "PASSED" if result.passed else "FAILED"
            logger.debug(f"Validation {i+1}: {status} - {result.message}")

    def _improve_output(self, output: str, validation_results: List[ValidationResult]) -> str:
        """Improve output based on validation results.

        Args:
            output: The output to improve
            validation_results: The validation results

        Returns:
            The improved output
        """
        # Skip improvement if all validations passed
        if all(result.passed for result in validation_results):
            return output

        # Skip improvement if no improver
        improver = self._state_manager.get("improver")
        if not improver:
            return output

        # Create improvement suggestions
        improvement_suggestions = []
        for result in validation_results:
            if not result.passed:
                improvement_suggestions.append(
                    {
                        "issue": result.message,
                        "suggestions": result.suggestions,
                    }
                )

        # Improve the output
        return improver.improve(output, improvement_suggestions)

    def _create_result(
        self,
        prompt: str,
        output: Union[str, GenerationResult],
        validation_results: List[ValidationResult],
        attempt_count: int,
    ) -> ChainResult:
        """
        Create a chain result from the execution results.

        Args:
            prompt: The prompt used
            output: The generated output
            validation_results: The validation results
            attempt_count: The number of attempts made

        Returns:
            The chain result
        """
        output_text = output.output if isinstance(output, GenerationResult) else output

        # Convert any validation results to core ValidationResults
        core_validation_results = []
        for result in validation_results:
            # First normalize the result to a standard dictionary format
            normalized_result = self._normalize_validation_result(result)

            # Then convert it to a core ValidationResult
            if not isinstance(result, ValidationResult):
                # Try to use the from_interface_validation_result if available
                try:
                    result_obj = ValidationResult.from_interface_validation_result(result)
                except (AttributeError, TypeError):
                    # If that fails, create a ValidationResult from the normalized dictionary
                    result_obj = ValidationResult(
                        passed=normalized_result["passed"],
                        message=normalized_result["message"],
                        score=normalized_result["score"],
                        issues=normalized_result["issues"],
                        suggestions=normalized_result["suggestions"],
                        metadata=normalized_result["metadata"],
                    )
                core_validation_results.append(result_obj)
            else:
                # If it's already a ValidationResult, just add it
                core_validation_results.append(result)

        # Create metadata
        metadata = {}
        if isinstance(output, GenerationResult):
            metadata.update(output.metadata)
            metadata.update(
                {
                    "prompt_tokens": output.prompt_tokens,
                    "completion_tokens": output.completion_tokens,
                    "total_tokens": output.total_tokens,
                }
            )

        # Create the result
        # If we have no validation results, create a default "passed" result
        if not core_validation_results:
            default_result = ValidationResult(
                passed=True,
                message="No validation performed",
                score=1.0,
                issues=[],
                suggestions=[],
                metadata={},
            )
            core_validation_results.append(default_result)

        # Create the message based on validation results
        if all(result.passed for result in core_validation_results):
            message = "Validation passed"
        else:
            message = "Validation failed: " + "; ".join(
                [r.message for r in core_validation_results if not r.passed]
            )

        return ChainResult(
            output=output_text,
            prompt=prompt,
            message=message,  # Add a message for the ChainResult
            validation_results=core_validation_results,
            passed=all(result.passed for result in core_validation_results),
            execution_time=self.execution_time,
            attempt_count=attempt_count,
            metadata=metadata,
        )

    def run(
        self,
        prompt: str,
        model: Optional[Any] = None,
        validators: Optional[List[Any]] = None,
        improver: Optional[Any] = None,
        formatter: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChainResult:
        """Run the chain with the given components.

        Args:
            prompt: The prompt to use
            model: Optional model to use (overrides the one in state)
            validators: Optional validators to use (override the ones in state)
            improver: Optional improver to use (overrides the one in state)
            formatter: Optional formatter to use (overrides the one in state)
            **kwargs: Additional run parameters

        Returns:
            The chain result

        Raises:
            ChainError: If the chain execution fails
        """
        # Update state with components if provided
        if model is not None:
            self._state_manager.update("model", model)
        if validators is not None:
            self._state_manager.update("validators", validators)
        if improver is not None:
            self._state_manager.update("improver", improver)
        if formatter is not None:
            self._state_manager.update("formatter", formatter)

        # Execute the chain
        return self.execute(prompt)

    def _normalize_validation_result(self, result: Any) -> Dict[str, Any]:
        """
        Normalize a validation result to a dictionary format expected by ChainResult.

        Args:
            result: The validation result to normalize

        Returns:
            A normalized dictionary representation of the validation result
        """
        # If result is already a dictionary, return it
        if isinstance(result, dict):
            # Ensure the dictionary has all required fields
            result.setdefault("passed", False)
            result.setdefault("message", "")
            result.setdefault("score", 0.0)
            result.setdefault("issues", [])
            result.setdefault("suggestions", [])
            result.setdefault("metadata", {})
            return result

        # For objects with attributes, convert to dictionary
        return {
            "passed": getattr(result, "passed", False),
            "message": getattr(result, "message", ""),
            "score": getattr(result, "score", 0.0),
            "issues": getattr(result, "issues", []),
            "suggestions": getattr(result, "suggestions", []),
            "metadata": getattr(result, "metadata", {}),
        }


class RetryManager:
    """Manages retry logic for the chain execution.

    This class handles the retry logic for the chain execution, including
    generation, validation, and improvement.

    Attributes:
        max_attempts: The maximum number of attempts to make
        fail_fast: Whether to fail fast if validation fails
    """

    def __init__(self, max_attempts: int = 3, fail_fast: bool = False):
        """Initialize the retry manager.

        Args:
            max_attempts: The maximum number of attempts to make
            fail_fast: Whether to fail fast if validation fails
        """
        self.max_attempts = max_attempts
        self.fail_fast = fail_fast

    def execute_with_retries(
        self,
        prompt: str,
        generate_func: Callable[[], Union[str, GenerationResult]],
        validate_func: Callable[[Union[str, GenerationResult]], List[ValidationResult]],
        improve_func: Callable[[str, List[ValidationResult]], str],
        create_result_func: Callable[
            [str, Union[str, GenerationResult], List[ValidationResult], int], ChainResult
        ],
    ) -> ChainResult:
        """Execute the chain with retries.

        Args:
            prompt: The prompt to use
            generate_func: Function to generate output
            validate_func: Function to validate output
            improve_func: Function to improve output
            create_result_func: Function to create result

        Returns:
            The chain result
        """
        attempts = 0
        output = None
        validation_results = []

        while attempts < self.max_attempts:
            attempts += 1

            # Generate output if this is the first attempt or the previous attempt failed
            if output is None or (
                not all(r.passed for r in validation_results) and attempts < self.max_attempts
            ):
                output = generate_func()

            # Validate the output
            validation_results = validate_func(output)

            # Check if validation passed
            if all(r.passed for r in validation_results):
                break

            # If fail fast is enabled, don't retry
            if self.fail_fast:
                break

            # If this is the last attempt, don't try to improve
            if attempts == self.max_attempts:
                break

            # Convert output to string for improvement if it's a GenerationResult
            output_text = output.output if isinstance(output, GenerationResult) else output

            # Improve the output
            improved_text = improve_func(output_text, validation_results)

            # Update the output
            if isinstance(output, GenerationResult):
                # Keep the same metadata but update the output
                output = GenerationResult(
                    output=improved_text,
                    model_name=output.model_name,
                    prompt_tokens=output.prompt_tokens,
                    completion_tokens=output.completion_tokens,
                    total_tokens=output.total_tokens,
                    metadata=output.metadata,
                )
            else:
                output = improved_text

        # Create the final result
        return create_result_func(prompt, output, validation_results, attempts)


from ..utils.mixins import InitializeStateMixin
