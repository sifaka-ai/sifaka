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
from sifaka.utils.state import StateManager

# Create engine
engine = Engine(
    state_manager=StateManager(),
    config=EngineConfig(max_attempts=3)
)

# Run engine
result = engine.run(
    prompt="Write a story",
    model=model,
    validators=validators,
    improver=improver,
    formatter=formatter
) if engine else ""

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

from typing import List, Optional
import time
from pydantic import BaseModel, PrivateAttr
from sifaka.interfaces.chain.components import Model, Validator, Improver
from sifaka.interfaces.chain.components.formatter import ChainFormatter as Formatter
from sifaka.interfaces.chain.models import ValidationResult
from ..utils.state import StateManager
from ..utils.logging import get_logger
from ..core.results import ChainResult
from ..utils.config import EngineConfig
from ..utils.errors import ChainError, safely_execute_chain
from .managers.cache import CacheManager
from .managers.retry import RetryManager

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
    engine = Engine(
        state_manager=StateManager(),
        config=EngineConfig(max_attempts=3)
    )

    # Run engine
    result = engine.run(
        prompt="Write a story",
        model=model,
        validators=validators,
        improver=improver,
        formatter=formatter
    ) if engine else ""
    ```

    Attributes:
        config (EngineConfig): Engine configuration
    """

    _state_manager: StateManager = PrivateAttr()
    config: EngineConfig = EngineConfig()
    _cache_manager: CacheManager = PrivateAttr()
    _retry_manager: RetryManager = PrivateAttr()

    def __init__(self, state_manager: StateManager, config: Optional[EngineConfig] = None) -> None:
        """
        Initialize the engine.

        Args:
            state_manager: State manager for state management
            config: Engine configuration
        """
        super().__init__(config=config or EngineConfig())
        self._state_manager = state_manager
        self._cache_manager = CacheManager(
            state_manager=state_manager,
            cache_enabled=self.config.params.get("cache_enabled", True),
            cache_size=self.config.params.get("cache_size", 100),
        )
        self._retry_manager = RetryManager(
            state_manager=state_manager, max_attempts=self.config.max_attempts
        )
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

        def run_operation():
            execution_count = self._state_manager.get("execution_count", 0)
            self._state_manager.update("execution_count", execution_count + 1)
            if self._cache_manager.has_cached_result(prompt):
                return self._cache_manager.get_cached_result(prompt)
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

        try:
            return safely_execute_chain(
                operation=run_operation,
                component_name=self.__class__.__name__,
                additional_metadata={
                    "prompt_length": len(prompt),
                    "validator_count": len(validators),
                    "has_improver": improver is not None,
                    "has_formatter": formatter is not None,
                },
            )
        except Exception as e:
            error_count = self._state_manager.get_metadata("error_count", 0)
            self._state_manager.set_metadata("error_count", error_count + 1)
            self._state_manager.set_metadata("last_error", str(e))
            self._state_manager.set_metadata("last_error_time", time.time())
            logger.error(f"Engine execution error: {str(e)}")
            raise ChainError(f"Engine execution failed: {str(e)}")

    def _generate_output(self, prompt: str) -> str:
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

        def generate_operation():
            return model.generate(prompt)

        return safely_execute_chain(
            operation=generate_operation,
            component_name="model",
            additional_metadata={"method": "generate", "prompt_length": len(prompt)},
        )

    def _validate_output(self, output: str) -> List[ValidationResult]:
        """
        Validate output using validators.

        This method validates the generated output using the validator components,
        handling any errors that may occur during validation. It returns a list
        of validation results, one for each validator.

        Args:
            output (str): The output to validate

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
        results = []
        for i, validator in enumerate(validators):

            def validate_operation():
                return validator.validate(output)

            result = safely_execute_chain(
                operation=validate_operation,
                component_name=f"validator_{i}",
                additional_metadata={
                    "method": "validate",
                    "validator_type": validator.__class__.__name__,
                    "output_length": (
                        len(output.output) if hasattr(output, "output") else len(output)
                    ),
                },
            )
            results.append(result)
            if self.config.params.get("fail_fast", False) and not result.passed:
                break
        return results

    def _improve_output(self, output: str, validation_results: List[ValidationResult]) -> str:
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

        def improve_operation():
            return improver.improve(output_text, validation_results)

        improved_text = safely_execute_chain(
            operation=improve_operation,
            component_name="improver",
            additional_metadata={
                "method": "improve",
                "improver_type": improver.__class__.__name__,
                "output_length": len(output_text),
                "validation_results_count": len(validation_results),
            },
        )
        if hasattr(output, "output") and hasattr(output, "metadata"):
            from sifaka.models.result import GenerationResult

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
        output: str,
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
            output (str): The generated output
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

        result = ChainResult(
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

            def format_operation():
                return formatter.format(output, validation_results)

            try:
                formatted_result = safely_execute_chain(
                    operation=format_operation,
                    component_name="formatter",
                    additional_metadata={
                        "method": "format",
                        "formatter_type": formatter.__class__.__name__,
                        "output_length": (
                            len(output.output) if hasattr(output, "output") else len(output)
                        ),
                        "validation_results_count": len(validation_results),
                    },
                )
                if isinstance(formatted_result, ChainResult):
                    result = formatted_result
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
