"""
Chain Engine Module

This module provides the core execution engine for the Sifaka chain system.
It coordinates the flow between components, handles retries, and manages state.

## Components
1. **Engine**: Core execution engine that coordinates the flow

## Usage Examples
```python
from sifaka.chain.v2.engine import Engine
from sifaka.chain.v2.config import EngineConfig
from sifaka.chain.v2.state import StateTracker

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
import logging

from .interfaces import Model, Validator, Improver, Formatter, ValidationResult
from .state import StateTracker
from .result import ChainResult
from .config import EngineConfig
from .errors import (
    ChainError, ModelError, ValidationError, ImproverError, FormatterError, safely_execute
)

# Configure logger
logger = logging.getLogger(__name__)


class Engine:
    """Core execution engine for the Sifaka chain system."""
    
    def __init__(
        self,
        state_tracker: StateTracker,
        config: Optional[EngineConfig] = None,
    ):
        """
        Initialize the engine.
        
        Args:
            state_tracker: State tracker for state management
            config: Engine configuration
        """
        self._state_tracker = state_tracker
        self._config = config or EngineConfig()
        
        # Initialize state
        self._state_tracker.update("config", self._config)
        self._state_tracker.update("initialized", True)
        self._state_tracker.update("execution_count", 0)
        self._state_tracker.update("result_cache", {})
        
        # Set metadata
        self._state_tracker.set_metadata("component_type", "engine")
        self._state_tracker.set_metadata("creation_time", time.time())
    
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
        execution_count = self._state_tracker.get("execution_count", 0)
        self._state_tracker.update("execution_count", execution_count + 1)
        
        # Check cache
        cache = self._state_tracker.get("result_cache", {})
        if prompt in cache:
            self._state_tracker.set_metadata("cache_hit", True)
            return cache[prompt]
        
        # Record start time
        start_time = time.time()
        
        try:
            # Create execution snapshot
            snapshot_id = self._state_tracker.create_snapshot(f"execution_{execution_count}")
            
            # Store components in state
            self._state_tracker.update("model", model)
            self._state_tracker.update("validators", validators)
            self._state_tracker.update("improver", improver)
            self._state_tracker.update("formatter", formatter)
            self._state_tracker.update("prompt", prompt)
            
            # Initialize execution state
            self._state_tracker.update("attempt", 0)
            self._state_tracker.update("output", "")
            self._state_tracker.update("validation_results", [])
            self._state_tracker.update("all_passed", False)
            
            # Execute with retries
            return self._execute_with_retries(prompt)
        
        except Exception as e:
            # Track error
            error_count = self._state_tracker.get_metadata("error_count", 0)
            self._state_tracker.set_metadata("error_count", error_count + 1)
            self._state_tracker.set_metadata("last_error", str(e))
            self._state_tracker.set_metadata("last_error_time", time.time())
            
            # Log error
            logger.error(f"Engine execution error: {str(e)}")
            
            # Raise as chain error
            raise ChainError(f"Engine execution failed: {str(e)}")
        
        finally:
            # Record execution time
            end_time = time.time()
            execution_time = end_time - start_time
            self._state_tracker.set_metadata("last_execution_time", execution_time)
            
            # Update average execution time
            avg_time = self._state_tracker.get_metadata("avg_execution_time", 0)
            count = execution_count + 1
            new_avg = (avg_time * (count - 1) + execution_time) / count
            self._state_tracker.set_metadata("avg_execution_time", new_avg)
            
            # Update max execution time
            max_time = self._state_tracker.get_metadata("max_execution_time", 0)
            if execution_time > max_time:
                self._state_tracker.set_metadata("max_execution_time", execution_time)
    
    def _execute_with_retries(self, prompt: str) -> ChainResult:
        """
        Execute the chain with retries.
        
        Args:
            prompt: The prompt to process
            
        Returns:
            The chain result
            
        Raises:
            ChainError: If chain execution fails after max attempts
        """
        max_attempts = self._config.max_attempts
        
        for attempt in range(1, max_attempts + 1):
            # Update attempt counter
            self._state_tracker.update("attempt", attempt)
            
            # Generate output
            output = self._generate_output(prompt)
            self._state_tracker.update("output", output)
            
            # Validate output
            validation_results = self._validate_output(output)
            self._state_tracker.update("validation_results", validation_results)
            
            # Check if all validations passed
            all_passed = all(r.passed for r in validation_results)
            self._state_tracker.update("all_passed", all_passed)
            
            # If all validations passed or no improver, return result
            if all_passed or not self._state_tracker.get("improver"):
                return self._create_result(prompt, output, validation_results, attempt)
            
            # If not last attempt, improve output and retry
            if attempt < max_attempts:
                output = self._improve_output(output, validation_results)
                self._state_tracker.update("output", output)
            else:
                # Last attempt, return result even if validations failed
                return self._create_result(prompt, output, validation_results, attempt)
        
        # Should never reach here, but just in case
        raise ChainError(f"Chain execution failed after {max_attempts} attempts")
    
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
        model = self._state_tracker.get("model")
        
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
        validators = self._state_tracker.get("validators", [])
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
        improver = self._state_tracker.get("improver")
        
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
        start_time = self._state_tracker.get_metadata("execution_start_time", 0)
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
                "execution_count": self._state_tracker.get("execution_count"),
            },
        )
        
        # Format result if formatter is available
        formatter = self._state_tracker.get("formatter")
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
            cache = self._state_tracker.get("result_cache", {})
            cache_size = self._config.params.get("cache_size", 100)
            
            # If cache is full, remove oldest entry
            if len(cache) >= cache_size:
                oldest_key = next(iter(cache))
                del cache[oldest_key]
            
            # Add result to cache
            cache[prompt] = result
            self._state_tracker.update("result_cache", cache)
        
        return result
