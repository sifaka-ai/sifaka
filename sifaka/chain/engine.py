"""
Chain Engine Module

This module provides the core execution engine for the Sifaka chain system.
It coordinates the flow between components, handles retries, and manages state.

## Components
1. **Engine**: Core execution engine that coordinates the flow

## Usage Examples
```python
from sifaka.chain.engine import Engine
from sifaka.chain.config import EngineConfig
from sifaka.chain.state import StateTracker

# Create engine
engine = Engine(
    state_tracker=StateTracker(),
    config=EngineConfig(max_attempts=3)
)

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
```
"""

from typing import Any, Dict, List, Optional
import time

from .interfaces import Model, Validator, Improver, Formatter, ValidationResult
from ..utils.state import StateManager
from ..utils.logging import get_logger
from .result import ChainResult
from .config import EngineConfig
from ..utils.errors import (
    ChainError,
    ModelError,
    ValidationError,
    ImproverError,
    FormatterError,
)
from ..utils.error_patterns import safely_execute_chain as safely_execute
from .managers.cache import CacheManager
from .managers.retry import RetryManager

# Configure logger
logger = get_logger(__name__)


class Engine:
    """Core execution engine for the Sifaka chain system."""

    def __init__(
        self,
        state_manager: StateManager,
        config: Optional[EngineConfig] = None,
    ):
        """
        Initialize the engine.

        Args:
            state_manager: State manager for state management
            config: Engine configuration
        """
        self._state_manager = state_manager
        self._config = config or EngineConfig()

        # Create managers
        self._cache_manager = CacheManager(
            state_manager=state_manager,
            cache_enabled=self._config.params.get("cache_enabled", True),
            cache_size=self._config.params.get("cache_size", 100),
        )

        self._retry_manager = RetryManager(
            state_manager=state_manager,
            max_attempts=self._config.max_attempts,
        )

        # Initialize state
        self._state_manager.update("config", self._config)
        self._state_manager.update("initialized", True)
        self._state_manager.update("execution_count", 0)

        # Set metadata
        self._state_manager.set_metadata("component_type", "engine")
        self._state_manager.set_metadata("creation_time", time.time())

    def run(
        self,
        prompt: str,
        model: Model,
        validators: List[Validator],
        improver: Optional[Improver] = None,
        formatter: Optional[Formatter] = None,
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
        # Track execution count
        execution_count = self._state_manager.get("execution_count", 0)
        self._state_manager.update("execution_count", execution_count + 1)

        # Check cache
        if self._cache_manager.has_cached_result(prompt):
            return self._cache_manager.get_cached_result(prompt)

        # Record start time
        start_time = time.time()
        self._state_manager.set_metadata("execution_start_time", start_time)

        try:
            # Store components in state
            self._state_manager.update("model", model)
            self._state_manager.update("validators", validators)
            self._state_manager.update("improver", improver)
            self._state_manager.update("formatter", formatter)
            self._state_manager.update("prompt", prompt)

            # Initialize execution state
            self._state_manager.update("attempt", 0)
            self._state_manager.update("output", "")
            self._state_manager.update("validation_results", [])
            self._state_manager.update("all_passed", False)

            # Execute with retries
            result = self._retry_manager.execute_with_retries(
                generate_func=lambda: self._generate_output(prompt),
                validate_func=lambda output: self._validate_output(output),
                improve_func=lambda output, results: self._improve_output(output, results),
                prompt=prompt,
                create_result_func=self._create_result,
            )

            # Cache result
            self._cache_manager.cache_result(prompt, result)

            return result

        except Exception as e:
            # Track error
            error_count = self._state_manager.get_metadata("error_count", 0)
            self._state_manager.set_metadata("error_count", error_count + 1)
            self._state_manager.set_metadata("last_error", str(e))
            self._state_manager.set_metadata("last_error_time", time.time())

            # Log error
            logger.error(f"Engine execution error: {str(e)}")

            # Raise as chain error
            raise ChainError(f"Engine execution failed: {str(e)}")

        finally:
            # Record execution time
            end_time = time.time()
            execution_time = end_time - start_time
            self._state_manager.set_metadata("last_execution_time", execution_time)

            # Update average execution time
            avg_time = self._state_manager.get_metadata("avg_execution_time", 0)
            count = execution_count + 1
            new_avg = (avg_time * (count - 1) + execution_time) / count
            self._state_manager.set_metadata("avg_execution_time", new_avg)

            # Update max execution time
            max_time = self._state_manager.get_metadata("max_execution_time", 0)
            if execution_time > max_time:
                self._state_manager.set_metadata("max_execution_time", execution_time)

    def _generate_output(self, prompt: str) -> str:
        """
        Generate output using the model.

        Args:
            prompt: The prompt to generate from

        Returns:
            The generated output

        Raises:
            ModelError: If model generation fails
        """
        model = self._state_manager.get("model")

        def generate_operation():
            return model.generate(prompt)

        return safely_execute(
            operation=generate_operation,
            component_name="model",
            component_type="Model",
            error_class=ModelError,
        )

    def _validate_output(self, output: str) -> List[ValidationResult]:
        """
        Validate output using validators.

        Args:
            output: The output to validate

        Returns:
            List of validation results

        Raises:
            ValidationError: If validation fails
        """
        validators = self._state_manager.get("validators", [])
        results = []

        for i, validator in enumerate(validators):

            def validate_operation():
                return validator.validate(output)

            result = safely_execute(
                operation=validate_operation,
                component_name=f"validator_{i}",
                component_type="Validator",
                error_class=ValidationError,
            )

            results.append(result)

            # If fail_fast is enabled and validation failed, stop
            if self._config.params.get("fail_fast", False) and not result.passed:
                break

        return results

    def _improve_output(self, output: str, validation_results: List[ValidationResult]) -> str:
        """
        Improve output using the improver.

        Args:
            output: The output to improve
            validation_results: The validation results

        Returns:
            The improved output

        Raises:
            ImproverError: If improvement fails
        """
        improver = self._state_manager.get("improver")

        if not improver:
            return output

        def improve_operation():
            return improver.improve(output, validation_results)

        return safely_execute(
            operation=improve_operation,
            component_name="improver",
            component_type="Improver",
            error_class=ImproverError,
        )

    def _create_result(
        self,
        prompt: str,
        output: str,
        validation_results: List[ValidationResult],
        attempt_count: int,
    ) -> ChainResult:
        """
        Create a chain result.

        Args:
            prompt: The original prompt
            output: The generated output
            validation_results: The validation results
            attempt_count: The number of attempts

        Returns:
            The chain result
        """
        # Calculate execution time
        start_time = self._state_manager.get_metadata("execution_start_time", 0)
        execution_time = time.time() - start_time if start_time else 0

        # Create result
        result = ChainResult(
            output=output,
            validation_results=validation_results,
            prompt=prompt,
            execution_time=execution_time,
            attempt_count=attempt_count,
            metadata={
                "engine_config": self._config.model_dump(),
                "execution_count": self._state_manager.get("execution_count"),
            },
        )

        # Format result if formatter is available
        formatter = self._state_manager.get("formatter")
        if formatter:

            def format_operation():
                return formatter.format(output, validation_results)

            try:
                formatted_result = safely_execute(
                    operation=format_operation,
                    component_name="formatter",
                    component_type="Formatter",
                    error_class=FormatterError,
                )

                # If formatter returns a ChainResult, use it
                if isinstance(formatted_result, ChainResult):
                    result = formatted_result
            except Exception as e:
                # Log error but continue with default result
                logger.warning(f"Result formatting failed: {str(e)}")

        # Cache result if caching is enabled
        if self._config.params.get("cache_enabled", True):
            cache = self._state_manager.get("result_cache", {})
            cache_size = self._config.params.get("cache_size", 100)

            # If cache is full, remove oldest entry
            if len(cache) >= cache_size:
                oldest_key = next(iter(cache))
                del cache[oldest_key]

            # Add result to cache
            cache[prompt] = result
            self._state_manager.update("result_cache", cache)

        return result
