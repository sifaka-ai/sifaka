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
from sifaka.chain.engine import Engine
from sifaka.chain.config import EngineConfig

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

from typing import List, Optional, Union
import time
from pydantic import BaseModel, PrivateAttr
from sifaka.interfaces.chain.components import Model, Validator, Improver
from sifaka.interfaces.chain.components.formatter import ChainFormatter as Formatter
from sifaka.interfaces.chain.models import ValidationResult
from ..utils.state import StateManager, create_engine_state
from ..utils.logging import get_logger
from ..core.results import ChainResult
from ..utils.config import EngineConfig
from ..utils.errors import ChainError, safely_execute_chain
from ..utils.errors.component import ModelError
from ..utils.errors.results import ErrorResult
from .managers.cache import CacheManager
from .managers.retry import RetryManager
from sifaka.models.result import GenerationResult

logger = get_logger(__name__)


class Engine(BaseModel):
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
        # Call super to ensure proper initialization of base state
        super()._initialize_state()

        self._state_manager.update("config", self.config)
        self._state_manager.update("initialized", True)
        self._state_manager.update("execution_count", 0)
        self._state_manager.update("cache", {})
        self._state_manager.set_metadata("component_type", self.__class__.__name__)
        self._state_manager.set_metadata("creation_time", time.time())

    def run(
        self,
        prompt: str,
        model: Model,
        validators: List[Validator],
        improver: Optional[Optional[Improver]] = None,
        formatter: Optional[Optional[Formatter]] = None,
    ) -> ChainResult:
        """
        Run the engine on the given prompt.

        Args:
            prompt: The prompt to process
            model: The model to use for generation
            validators: The validators to use for validation
            improver: Optional improver for output improvement
            formatter: Optional formatter for result formatting

        Returns:
            The chain result

        Raises:
            ChainError: If chain execution fails
            ModelError: If model generation fails
            ValidationError: If validation fails
            ImproverError: If improver refinement fails
            FormatterError: If formatter formatting fails
        """

        def run_operation() -> ChainResult:
            execution_count = self._state_manager.get("execution_count", 0)
            self._state_manager.update("execution_count", execution_count + 1)
            if self._cache_manager.has_cached_result(prompt):
                cached_result = self._cache_manager.get_cached_result(prompt)
                if isinstance(cached_result, ChainResult):
                    return cached_result
                raise ChainError(f"Cached result is not a ChainResult: {type(cached_result)}")
            start_time = time.time()
            self._state_manager.set_metadata("execution_start_time", start_time)
            try:
                self._state_manager.update("model", model)
                self._state_manager.update("validators", validators)
                self._state_manager.update("improver", improver)
                self._state_manager.update("formatter", formatter)
                self._state_manager.update("prompt", prompt)
                self._state_manager.update("attempt", 0)
                self._state_manager.update("output", "")
                self._state_manager.update("validation_results", [])
                self._state_manager.update("all_passed", False)
                result = self._retry_manager.execute_with_retries(
                    generate_func=lambda: self._generate_output(prompt),
                    validate_func=lambda output: self._validate_output(output),
                    improve_func=lambda output, results: self._improve_output(output, results),
                    prompt=prompt,
                    create_result_func=self._create_result,
                )
                if not isinstance(result, ChainResult):
                    raise ChainError(f"Result is not a ChainResult: {type(result)}")
                self._cache_manager.cache_result(prompt, result)
                return result
            finally:
                end_time = time.time()
                execution_time = end_time - start_time
                self._state_manager.set_metadata("last_execution_time", execution_time)
                avg_time = self._state_manager.get_metadata("avg_execution_time", 0)
                count = execution_count + 1
                new_avg = (avg_time * (count - 1) + execution_time) / count
                self._state_manager.set_metadata("avg_execution_time", new_avg)
                max_time = self._state_manager.get_metadata("max_execution_time", 0)
                if execution_time > max_time:
                    self._state_manager.set_metadata("max_execution_time", execution_time)

        additional_metadata = {
            "prompt_length": len(prompt),
            "validator_count": len(validators),
            "has_improver": improver is not None,
            "has_formatter": formatter is not None,
        }

        try:
            # Use the safely_execute_chain function with the correct parameters
            result = safely_execute_chain(
                run_operation,
                self.__class__.__name__,
                None,  # default_result
                "error",  # log_level
                True,  # include_traceback
                additional_metadata,
            )
            # Handle the case where the result is an ErrorResult
            if isinstance(result, ErrorResult):
                logger.error(f"Engine execution error: {result.error_message}")
                raise ChainError(f"Engine execution failed: {result.error_message}")
            if isinstance(result, ChainResult):
                return result
            # This should never happen, but we need to satisfy the type checker
            raise ChainError("Unexpected result type from run_operation")
        except Exception as e:
            error_count = self._state_manager.get_metadata("error_count", 0)
            self._state_manager.set_metadata("error_count", error_count + 1)
            self._state_manager.set_metadata("last_error", str(e))
            self._state_manager.set_metadata("last_error_time", time.time())
            logger.error(f"Engine execution error: {str(e)}")
            raise ChainError(f"Engine execution failed: {str(e)}")

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
        """
        Validate output using validators.

        This method validates the generated output using the validator components,
        handling any errors that may occur during validation. It returns a list
        of validation results, one for each validator.

        Args:
            output (Union[str, GenerationResult]): The output to validate

        Returns:
            List[ValidationResult]: List of validation results, one for each validator

        Raises:
            ValidationError: If validation fails

        Example:
            ```python
            validation_results = (engine._validate_output("Generated text")
            all_passed = all(result.passed for result in validation_results)
            ```
        """
        validators = self._state_manager.get("validators", [])
        results: List[ValidationResult] = []
        for i, validator in enumerate(validators):

            def validate_operation() -> ValidationResult:
                validation_result = validator.validate(output)
                if not isinstance(validation_result, ValidationResult):
                    raise ValueError(
                        f"Validator returned unexpected type: {type(validation_result)}"
                    )
                return validation_result

            additional_metadata = {
                "method": "validate",
                "validator_type": validator.__class__.__name__,
                "output_length": (len(output.output) if hasattr(output, "output") else len(output)),
            }

            result = safely_execute_chain(
                validate_operation,
                f"validator_{i}",
                None,  # default_result
                "error",  # log_level
                True,  # include_traceback
                additional_metadata,
            )

            # If result is an ErrorResult, create a failed ValidationResult
            if isinstance(result, ErrorResult):
                validation_result = ValidationResult(
                    passed=False,
                    message=f"Validation error: {result.error_message}",
                    score=0.0,
                    issues=[result.error_message],
                    suggestions=["Fix the validator or try a different input"],
                    metadata=result.metadata,
                )
                results.append(validation_result)
            else:
                if isinstance(result, ValidationResult):
                    results.append(result)
                else:
                    # This should never happen, but we need to satisfy the type checker
                    validation_result = ValidationResult(
                        passed=False,
                        message=f"Unexpected result type from validate_operation",
                        score=0.0,
                        issues=["Unexpected result type"],
                        suggestions=["Fix the validator implementation"],
                        metadata={},
                    )
                    results.append(validation_result)

            if self.config.params.get("fail_fast", False) and not results[-1].passed:
                break

        return results

    def _improve_output(
        self, output: Union[str, GenerationResult], validation_results: List[ValidationResult]
    ) -> Union[str, GenerationResult]:
        """
        Improve output using the improver.

        This method improves the generated output using the improver component
        if one is available, handling any errors that may occur during improvement.
        It uses the validation results to guide the improvement process.

        Args:
            output (str): The output to improve
            validation_results (List[ValidationResult]): The validation results

        Returns:
            str: The improved output, or the original output if no improver is available

        Raises:
            ImproverError: If improvement fails

        Example:
            ```python
            improved_output = (engine._improve_output(
                "Generated text",
                validation_results
            )
            ```
        """
        improver = self._state_manager.get("improver")
        if not improver:
            return output
        output_text = output.output if hasattr(output, "output") else output

        def improve_operation() -> str:
            improved_output = improver.improve(output_text, validation_results)
            if not isinstance(improved_output, str):
                raise ValueError(f"Improver returned unexpected type: {type(improved_output)}")
            return improved_output

        additional_metadata = {
            "method": "improve",
            "improver_type": improver.__class__.__name__,
            "output_length": len(output_text),
            "validation_results_count": len(validation_results),
        }

        result = safely_execute_chain(
            improve_operation,
            "improver",
            None,  # default_result
            "error",  # log_level
            True,  # include_traceback
            additional_metadata,
        )

        # If result is an ErrorResult, return the original output
        if isinstance(result, ErrorResult):
            logger.warning(f"Improvement failed: {result.error_message}. Using original output.")
            return output

        if isinstance(result, str):
            improved_text = result
        else:
            # This should never happen, but we need to satisfy the type checker
            logger.warning("Unexpected result type from improve_operation. Using original output.")
            return output

        if hasattr(output, "output") and hasattr(output, "metadata"):
            return GenerationResult(
                output=improved_text,
                prompt_tokens=getattr(output, "prompt_tokens", 0),
                completion_tokens=getattr(output, "completion_tokens", 0),
                metadata=getattr(output, "metadata", {}),
            )
        return improved_text

    def _create_result(
        self,
        prompt: str,
        output: Union[str, GenerationResult],
        validation_results: List[ValidationResult],
        attempt_count: int,
    ) -> ChainResult:
        """
        Create a chain result.

        This method creates a ChainResult object from the given parameters,
        including execution time and metadata. It also applies formatting
        if a formatter is available.

        Args:
            prompt (str): The original prompt
            output (Union[str, GenerationResult]): The generated output
            validation_results (List[ValidationResult]): The validation results
            attempt_count (int): The number of attempts made

        Returns:
            ChainResult: The chain result object

        Example:
            ```python
            result = (engine._create_result(
                prompt="Write a story",
                output="Generated text",
                validation_results=validation_results,
                attempt_count=2
            )
            ```
        """
        start_time = self._state_manager.get_metadata("execution_start_time", 0)
        execution_time = time.time() - start_time if start_time else 0
        output_text = output.output if hasattr(output, "output") else output
        all_passed = (
            all(result.passed for result in validation_results) if validation_results else True
        )

        # Use the from_interface_validation_results method to handle the conversion
        result = ChainResult.from_interface_validation_results(
            output=output_text,
            validation_results=validation_results,
            prompt=prompt,
            execution_time=execution_time,
            attempt_count=attempt_count,
            passed=all_passed,
            message=(
                "Chain execution completed successfully"
                if all_passed
                else "Chain execution completed with validation failures"
            ),
            metadata={
                "engine_config": {
                    "max_attempts": self.config.max_attempts,
                    "params": self.config.params,
                },
                "execution_count": self._state_manager.get("execution_count"),
            },
        )

        formatter = self._state_manager.get("formatter")
        if formatter:

            def format_operation() -> ChainResult:
                formatted_result = formatter.format(output, validation_results)
                if not isinstance(formatted_result, ChainResult):
                    raise ValueError(
                        f"Formatter returned unexpected type: {type(formatted_result)}"
                    )
                return formatted_result

            try:
                additional_metadata = {
                    "method": "format",
                    "formatter_type": formatter.__class__.__name__,
                    "output_length": (
                        len(output.output) if hasattr(output, "output") else len(output)
                    ),
                    "validation_results_count": len(validation_results),
                }

                formatted_result = safely_execute_chain(
                    format_operation,
                    "formatter",
                    None,  # default_result
                    "error",  # log_level
                    True,  # include_traceback
                    additional_metadata,
                )
                if isinstance(formatted_result, ChainResult):
                    result = formatted_result
                elif isinstance(formatted_result, ErrorResult):
                    logger.warning(
                        f"Formatting failed: {formatted_result.error_message}. Using original result."
                    )
            except Exception as e:
                logger.warning(f"Result formatting failed: {str(e)}")
        if self.config.params.get("cache_enabled", True):
            cache = self._state_manager.get("result_cache", {})
            cache_size = self.config.params.get("cache_size", 100)
            if len(cache) >= cache_size:
                oldest_key = next(iter(cache))
                del cache[oldest_key]
            cache[prompt] = result
            self._state_manager.update("result_cache", cache)
        return result
