"""
Chain Adapters Module

This module provides adapter classes for integrating existing Sifaka components
with the chain system. These adapters implement the interfaces while
delegating to the existing components.

## Adapter Classes
1. **ModelAdapter**: Adapts existing model providers to the Model interface
2. **ValidatorAdapter**: Adapts existing rules to the Validator interface
3. **ImproverAdapter**: Adapts existing critics to the Improver interface
4. **FormatterAdapter**: Adapts existing formatters to the Formatter interface

## Usage Examples
```python
from sifaka.chain.adapters import ModelAdapter, ValidatorAdapter, ImproverAdapter
from sifaka.models import OpenAIProvider
from sifaka.rules import create_length_rule
from sifaka.critics import create_prompt_critic

# Create components
model_provider = OpenAIProvider("gpt-3.5-turbo")
rule = create_length_rule(min_chars=10, max_chars=1000)
critic = create_prompt_critic(
    llm_provider=model_provider,
    system_prompt="You are an expert editor that improves text."
)

# Create adapters
model = ModelAdapter(model_provider)
validator = ValidatorAdapter(rule)
improver = ImproverAdapter(critic)

# Use adapters
output = model.generate("Write a short story")
validation_result = validator.validate(output)
if not validation_result.passed:
    improved_output = improver.improve(output, [validation_result])
```
"""

from typing import Any, List
import asyncio
import time

from .interfaces import Model, Validator, Improver, Formatter, ValidationResult
from ..utils.errors import (
    ModelError,
    ValidationError,
    ImproverError,
    FormatterError,
    safely_execute_chain as safely_execute,
)


class ModelAdapter(Model):
    """
    Adapter for existing model providers.

    This adapter implements the Model interface for existing model providers,
    using the standardized state management pattern.
    """

    def __init__(self, model: Any, name: str = None, description: str = None):
        """
        Initialize the model adapter.

        Args:
            model: The model provider to adapt
            name: Optional name for the adapter
            description: Optional description for the adapter
        """
        from ..utils.state import create_state_manager, AdapterState

        # Store model
        self._model = model

        # Set name and description
        self._name = name or f"{type(model).__name__}Adapter"
        self._description = description or f"Adapter for {type(model).__name__}"

        # Create state manager
        self._state_manager = create_state_manager(AdapterState)

        # Initialize state
        self._initialize_state()

    def _initialize_state(self) -> None:
        """Initialize adapter state."""
        # Update state with initial values
        self._state_manager.update("adaptee", self._model)
        self._state_manager.update("initialized", True)
        self._state_manager.update("cache", {})

        # Set metadata
        self._state_manager.set_metadata("component_type", "model_adapter")
        self._state_manager.set_metadata("adaptee_type", type(self._model).__name__)
        self._state_manager.set_metadata("creation_time", time.time())

    @property
    def name(self) -> str:
        """Get adapter name."""
        return self._name

    @property
    def description(self) -> str:
        """Get adapter description."""
        return self._description

    def generate(self, prompt: str) -> str:
        """
        Generate text from a prompt.

        Args:
            prompt: The prompt to generate text from

        Returns:
            The generated text

        Raises:
            ModelError: If text generation fails
        """
        # Ensure adapter is initialized
        if not self._state_manager.get("initialized", False):
            self._initialize_state()

        # Record start time
        start_time = time.time()

        try:

            def generate_operation():
                # Check for different model provider interfaces
                if hasattr(self._model, "invoke"):
                    return self._model.invoke(prompt)
                elif hasattr(self._model, "generate"):
                    return self._model.generate(prompt)
                elif hasattr(self._model, "run"):
                    return self._model.run(prompt)
                elif hasattr(self._model, "process"):
                    return self._model.process(prompt)
                else:
                    raise ModelError(f"Unsupported model provider: {type(self._model).__name__}")

            # Execute operation safely
            result = safely_execute(
                operation=generate_operation,
                component_name=self.name,
                component_type="Model",
                error_class=ModelError,
            )

            # Update statistics
            end_time = time.time()
            execution_time = end_time - start_time

            # Update generation count
            generation_count = self._state_manager.get_metadata("generation_count", 0)
            self._state_manager.set_metadata("generation_count", generation_count + 1)

            # Update average execution time
            avg_time = self._state_manager.get_metadata("avg_execution_time", 0)
            new_avg = ((avg_time * generation_count) + execution_time) / (generation_count + 1)
            self._state_manager.set_metadata("avg_execution_time", new_avg)

            # Update max execution time if needed
            max_time = self._state_manager.get_metadata("max_execution_time", 0)
            if execution_time > max_time:
                self._state_manager.set_metadata("max_execution_time", execution_time)

            return result

        except Exception as e:
            # Update error statistics
            error_count = self._state_manager.get_metadata("error_count", 0)
            self._state_manager.set_metadata("error_count", error_count + 1)
            self._state_manager.set_metadata("last_error", str(e))
            self._state_manager.set_metadata("last_error_time", time.time())

            # Re-raise as ModelError
            if isinstance(e, ModelError):
                raise e
            raise ModelError(f"Model generation failed: {str(e)}")

    async def generate_async(self, prompt: str) -> str:
        """
        Generate text asynchronously.

        Args:
            prompt: The prompt to generate text from

        Returns:
            The generated text

        Raises:
            ModelError: If text generation fails
        """
        # Ensure adapter is initialized
        if not self._state_manager.get("initialized", False):
            self._initialize_state()

        # Record start time
        start_time = time.time()

        try:
            # Check if model has async methods
            if hasattr(self._model, "invoke_async"):
                result = await self._model.invoke_async(prompt)
            elif hasattr(self._model, "generate_async"):
                result = await self._model.generate_async(prompt)
            elif hasattr(self._model, "run_async"):
                result = await self._model.run_async(prompt)
            elif hasattr(self._model, "process_async"):
                result = await self._model.process_async(prompt)
            else:
                # Fall back to running synchronous method in executor
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, self.generate, prompt)
                # Return early since statistics are updated in the synchronous method
                return result

            # Update statistics
            end_time = time.time()
            execution_time = end_time - start_time

            # Update generation count
            generation_count = self._state_manager.get_metadata("generation_count", 0)
            self._state_manager.set_metadata("generation_count", generation_count + 1)

            # Update average execution time
            avg_time = self._state_manager.get_metadata("avg_execution_time", 0)
            new_avg = ((avg_time * generation_count) + execution_time) / (generation_count + 1)
            self._state_manager.set_metadata("avg_execution_time", new_avg)

            # Update max execution time if needed
            max_time = self._state_manager.get_metadata("max_execution_time", 0)
            if execution_time > max_time:
                self._state_manager.set_metadata("max_execution_time", execution_time)

            return result

        except Exception as e:
            # Update error statistics
            error_count = self._state_manager.get_metadata("error_count", 0)
            self._state_manager.set_metadata("error_count", error_count + 1)
            self._state_manager.set_metadata("last_error", str(e))
            self._state_manager.set_metadata("last_error_time", time.time())

            # Re-raise as ModelError
            if isinstance(e, ModelError):
                raise e
            raise ModelError(f"Async model generation failed: {str(e)}")


class ValidatorAdapter(Validator):
    """
    Adapter for existing rules.

    This adapter implements the Validator interface for existing rules,
    using the standardized state management pattern.
    """

    def __init__(self, validator: Any, name: str = None, description: str = None):
        """
        Initialize the validator adapter.

        Args:
            validator: The rule to adapt
            name: Optional name for the adapter
            description: Optional description for the adapter
        """
        from ..utils.state import create_state_manager, AdapterState

        # Store validator
        self._validator = validator

        # Set name and description
        self._name = name or f"{type(validator).__name__}Adapter"
        self._description = description or f"Adapter for {type(validator).__name__}"

        # Create state manager
        self._state_manager = create_state_manager(AdapterState)

        # Initialize state
        self._initialize_state()

    def _initialize_state(self) -> None:
        """Initialize adapter state."""
        # Update state with initial values
        self._state_manager.update("adaptee", self._validator)
        self._state_manager.update("initialized", True)
        self._state_manager.update("cache", {})

        # Set metadata
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
        # Ensure adapter is initialized
        if not self._state_manager.get("initialized", False):
            self._initialize_state()

        # Record start time
        start_time = time.time()

        try:

            def validate_operation():
                # Check for different rule interfaces
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

                # Convert result to ValidationResult
                return self._convert_result(result)

            # Execute operation safely
            result = safely_execute(
                operation=validate_operation,
                component_name=self.name,
                component_type="Validator",
                error_class=ValidationError,
            )

            # Update statistics
            end_time = time.time()
            execution_time = end_time - start_time

            # Update validation count
            validation_count = self._state_manager.get_metadata("validation_count", 0)
            self._state_manager.set_metadata("validation_count", validation_count + 1)

            # Update success/failure counts
            if result.passed:
                success_count = self._state_manager.get_metadata("success_count", 0)
                self._state_manager.set_metadata("success_count", success_count + 1)
            else:
                failure_count = self._state_manager.get_metadata("failure_count", 0)
                self._state_manager.set_metadata("failure_count", failure_count + 1)

            # Update average execution time
            avg_time = self._state_manager.get_metadata("avg_execution_time", 0)
            new_avg = ((avg_time * validation_count) + execution_time) / (validation_count + 1)
            self._state_manager.set_metadata("avg_execution_time", new_avg)

            # Update max execution time if needed
            max_time = self._state_manager.get_metadata("max_execution_time", 0)
            if execution_time > max_time:
                self._state_manager.set_metadata("max_execution_time", execution_time)

            # Cache result if caching is enabled
            if self._state_manager.get("cache_enabled", True):
                cache = self._state_manager.get("cache", {})
                cache_key = f"{output[:50]}_{len(output)}"
                cache[cache_key] = result
                self._state_manager.update("cache", cache)

            return result

        except Exception as e:
            # Update error statistics
            error_count = self._state_manager.get_metadata("error_count", 0)
            self._state_manager.set_metadata("error_count", error_count + 1)
            self._state_manager.set_metadata("last_error", str(e))
            self._state_manager.set_metadata("last_error_time", time.time())

            # Re-raise as ValidationError
            if isinstance(e, ValidationError):
                raise e
            raise ValidationError(f"Validation failed: {str(e)}")

    async def validate_async(self, output: str) -> ValidationResult:
        """
        Validate an output asynchronously.

        Args:
            output: The output to validate

        Returns:
            The validation result

        Raises:
            ValidationError: If validation fails
        """
        # Ensure adapter is initialized
        if not self._state_manager.get("initialized", False):
            self._initialize_state()

        # Record start time
        start_time = time.time()

        try:
            # Check if validator has async methods
            if hasattr(self._validator, "validate_async"):
                result = await self._validator.validate_async(output)
                result = self._convert_result(result)
            elif hasattr(self._validator, "process_async"):
                result = await self._validator.process_async(output)
                result = self._convert_result(result)
            elif hasattr(self._validator, "run_async"):
                result = await self._validator.run_async(output)
                result = self._convert_result(result)
            else:
                # Fall back to running synchronous method in executor
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, self.validate, output)
                # Return early since statistics are updated in the synchronous method
                return result

            # Update statistics
            end_time = time.time()
            execution_time = end_time - start_time

            # Update validation count
            validation_count = self._state_manager.get_metadata("validation_count", 0)
            self._state_manager.set_metadata("validation_count", validation_count + 1)

            # Update success/failure counts
            if result.passed:
                success_count = self._state_manager.get_metadata("success_count", 0)
                self._state_manager.set_metadata("success_count", success_count + 1)
            else:
                failure_count = self._state_manager.get_metadata("failure_count", 0)
                self._state_manager.set_metadata("failure_count", failure_count + 1)

            # Update average execution time
            avg_time = self._state_manager.get_metadata("avg_execution_time", 0)
            new_avg = ((avg_time * validation_count) + execution_time) / (validation_count + 1)
            self._state_manager.set_metadata("avg_execution_time", new_avg)

            # Update max execution time if needed
            max_time = self._state_manager.get_metadata("max_execution_time", 0)
            if execution_time > max_time:
                self._state_manager.set_metadata("max_execution_time", execution_time)

            # Cache result if caching is enabled
            if self._state_manager.get("cache_enabled", True):
                cache = self._state_manager.get("cache", {})
                cache_key = f"{output[:50]}_{len(output)}"
                cache[cache_key] = result
                self._state_manager.update("cache", cache)

            return result

        except Exception as e:
            # Update error statistics
            error_count = self._state_manager.get_metadata("error_count", 0)
            self._state_manager.set_metadata("error_count", error_count + 1)
            self._state_manager.set_metadata("last_error", str(e))
            self._state_manager.set_metadata("last_error_time", time.time())

            # Re-raise as ValidationError
            if isinstance(e, ValidationError):
                raise e
            raise ValidationError(f"Async validation failed: {str(e)}")

    def _convert_result(self, result: Any) -> ValidationResult:
        """
        Convert a rule result to a ValidationResult.

        Args:
            result: The rule result to convert

        Returns:
            The converted ValidationResult
        """
        # If already a ValidationResult, return as is
        if isinstance(result, ValidationResult):
            return result

        # Extract fields from result
        passed = getattr(result, "passed", False)
        message = getattr(result, "message", "")
        score = getattr(result, "score", 0.0)
        issues = getattr(result, "issues", [])
        suggestions = getattr(result, "suggestions", [])
        metadata = getattr(result, "metadata", {})

        # Create ValidationResult
        return ValidationResult(
            passed=passed,
            message=message,
            score=score,
            issues=issues,
            suggestions=suggestions,
            metadata=metadata,
        )


class ImproverAdapter(Improver):
    """
    Adapter for existing critics.

    This adapter implements the Improver interface for existing critics,
    using the standardized state management pattern.
    """

    def __init__(self, improver: Any, name: str = None, description: str = None):
        """
        Initialize the improver adapter.

        Args:
            improver: The critic to adapt
            name: Optional name for the adapter
            description: Optional description for the adapter
        """
        from ..utils.state import create_state_manager, AdapterState

        # Store improver
        self._improver = improver

        # Set name and description
        self._name = name or f"{type(improver).__name__}Adapter"
        self._description = description or f"Adapter for {type(improver).__name__}"

        # Create state manager
        self._state_manager = create_state_manager(AdapterState)

        # Initialize state
        self._initialize_state()

    def _initialize_state(self) -> None:
        """Initialize adapter state."""
        # Update state with initial values
        self._state_manager.update("adaptee", self._improver)
        self._state_manager.update("initialized", True)
        self._state_manager.update("cache", {})

        # Set metadata
        self._state_manager.set_metadata("component_type", "improver_adapter")
        self._state_manager.set_metadata("adaptee_type", type(self._improver).__name__)
        self._state_manager.set_metadata("creation_time", time.time())

    @property
    def name(self) -> str:
        """Get adapter name."""
        return self._name

    @property
    def description(self) -> str:
        """Get adapter description."""
        return self._description

    def improve(self, output: str, validation_results: List[ValidationResult]) -> str:
        """
        Improve an output based on validation results.

        Args:
            output: The output to improve
            validation_results: The validation results to use for improvement

        Returns:
            The improved output

        Raises:
            ImproverError: If improvement fails
        """
        # Ensure adapter is initialized
        if not self._state_manager.get("initialized", False):
            self._initialize_state()

        # Record start time
        start_time = time.time()

        try:

            def improve_operation():
                # Check for different critic interfaces
                if hasattr(self._improver, "improve"):
                    return self._improver.improve(output, validation_results)
                elif hasattr(self._improver, "refine"):
                    return self._improver.refine(output, validation_results)
                elif hasattr(self._improver, "process"):
                    return self._improver.process(output, validation_results)
                elif hasattr(self._improver, "run"):
                    return self._improver.run(output, validation_results)
                else:
                    raise ImproverError(f"Unsupported improver: {type(self._improver).__name__}")

            # Execute operation safely
            result = safely_execute(
                operation=improve_operation,
                component_name=self.name,
                component_type="Improver",
                error_class=ImproverError,
            )

            # Update statistics
            end_time = time.time()
            execution_time = end_time - start_time

            # Update improvement count
            improvement_count = self._state_manager.get_metadata("improvement_count", 0)
            self._state_manager.set_metadata("improvement_count", improvement_count + 1)

            # Update average execution time
            avg_time = self._state_manager.get_metadata("avg_execution_time", 0)
            new_avg = ((avg_time * improvement_count) + execution_time) / (improvement_count + 1)
            self._state_manager.set_metadata("avg_execution_time", new_avg)

            # Update max execution time if needed
            max_time = self._state_manager.get_metadata("max_execution_time", 0)
            if execution_time > max_time:
                self._state_manager.set_metadata("max_execution_time", execution_time)

            # Cache result if caching is enabled
            if self._state_manager.get("cache_enabled", True):
                cache = self._state_manager.get("cache", {})
                cache_key = f"{output[:50]}_{len(output)}"
                cache[cache_key] = result
                self._state_manager.update("cache", cache)

            return result

        except Exception as e:
            # Update error statistics
            error_count = self._state_manager.get_metadata("error_count", 0)
            self._state_manager.set_metadata("error_count", error_count + 1)
            self._state_manager.set_metadata("last_error", str(e))
            self._state_manager.set_metadata("last_error_time", time.time())

            # Re-raise as ImproverError
            if isinstance(e, ImproverError):
                raise e
            raise ImproverError(f"Improvement failed: {str(e)}")

    async def improve_async(self, output: str, validation_results: List[ValidationResult]) -> str:
        """
        Improve an output asynchronously.

        Args:
            output: The output to improve
            validation_results: The validation results to use for improvement

        Returns:
            The improved output

        Raises:
            ImproverError: If improvement fails
        """
        # Ensure adapter is initialized
        if not self._state_manager.get("initialized", False):
            self._initialize_state()

        # Record start time
        start_time = time.time()

        try:
            # Check if improver has async methods
            if hasattr(self._improver, "improve_async"):
                result = await self._improver.improve_async(output, validation_results)
            elif hasattr(self._improver, "refine_async"):
                result = await self._improver.refine_async(output, validation_results)
            elif hasattr(self._improver, "process_async"):
                result = await self._improver.process_async(output, validation_results)
            elif hasattr(self._improver, "run_async"):
                result = await self._improver.run_async(output, validation_results)
            else:
                # Fall back to running synchronous method in executor
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, self.improve, output, validation_results)
                # Return early since statistics are updated in the synchronous method
                return result

            # Update statistics
            end_time = time.time()
            execution_time = end_time - start_time

            # Update improvement count
            improvement_count = self._state_manager.get_metadata("improvement_count", 0)
            self._state_manager.set_metadata("improvement_count", improvement_count + 1)

            # Update average execution time
            avg_time = self._state_manager.get_metadata("avg_execution_time", 0)
            new_avg = ((avg_time * improvement_count) + execution_time) / (improvement_count + 1)
            self._state_manager.set_metadata("avg_execution_time", new_avg)

            # Update max execution time if needed
            max_time = self._state_manager.get_metadata("max_execution_time", 0)
            if execution_time > max_time:
                self._state_manager.set_metadata("max_execution_time", execution_time)

            # Cache result if caching is enabled
            if self._state_manager.get("cache_enabled", True):
                cache = self._state_manager.get("cache", {})
                cache_key = f"{output[:50]}_{len(output)}"
                cache[cache_key] = result
                self._state_manager.update("cache", cache)

            return result

        except Exception as e:
            # Update error statistics
            error_count = self._state_manager.get_metadata("error_count", 0)
            self._state_manager.set_metadata("error_count", error_count + 1)
            self._state_manager.set_metadata("last_error", str(e))
            self._state_manager.set_metadata("last_error_time", time.time())

            # Re-raise as ImproverError
            if isinstance(e, ImproverError):
                raise e
            raise ImproverError(f"Async improvement failed: {str(e)}")


class FormatterAdapter(Formatter):
    """
    Adapter for existing formatters.

    This adapter implements the Formatter interface for existing formatters,
    using the standardized state management pattern.
    """

    def __init__(self, formatter: Any, name: str = None, description: str = None):
        """
        Initialize the formatter adapter.

        Args:
            formatter: The formatter to adapt
            name: Optional name for the adapter
            description: Optional description for the adapter
        """
        from ..utils.state import create_state_manager, AdapterState

        # Store formatter
        self._formatter = formatter

        # Set name and description
        self._name = name or f"{type(formatter).__name__ if formatter else 'Default'}Adapter"
        self._description = (
            description
            or f"Adapter for {type(formatter).__name__ if formatter else 'default formatter'}"
        )

        # Create state manager
        self._state_manager = create_state_manager(AdapterState)

        # Initialize state
        self._initialize_state()

    def _initialize_state(self) -> None:
        """Initialize adapter state."""
        # Update state with initial values
        self._state_manager.update("adaptee", self._formatter)
        self._state_manager.update("initialized", True)
        self._state_manager.update("cache", {})

        # Set metadata
        self._state_manager.set_metadata("component_type", "formatter_adapter")
        self._state_manager.set_metadata(
            "adaptee_type", type(self._formatter).__name__ if self._formatter else "None"
        )
        self._state_manager.set_metadata("creation_time", time.time())

    @property
    def name(self) -> str:
        """Get adapter name."""
        return self._name

    @property
    def description(self) -> str:
        """Get adapter description."""
        return self._description

    def format(self, output: str, validation_results: List[ValidationResult]) -> Any:
        """
        Format a result.

        Args:
            output: The output to format
            validation_results: The validation results to include

        Returns:
            The formatted result

        Raises:
            FormatterError: If formatting fails
        """
        # Ensure adapter is initialized
        if not self._state_manager.get("initialized", False):
            self._initialize_state()

        # If no formatter is provided, return a default result
        if self._formatter is None:
            from ..chain.result import ChainResult

            return ChainResult(
                output=output,
                validation_results=validation_results,
                prompt="",
                execution_time=0.0,
                attempt_count=1,
                metadata={},
            )

        # Record start time
        start_time = time.time()

        try:

            def format_operation():
                # Check for different formatter interfaces
                if hasattr(self._formatter, "format"):
                    return self._formatter.format(output, validation_results)
                elif hasattr(self._formatter, "format_result"):
                    return self._formatter.format_result(output, validation_results)
                elif hasattr(self._formatter, "process"):
                    return self._formatter.process(output, validation_results)
                elif hasattr(self._formatter, "run"):
                    return self._formatter.run(output, validation_results)
                else:
                    raise FormatterError(f"Unsupported formatter: {type(self._formatter).__name__}")

            # Execute operation safely
            result = safely_execute(
                operation=format_operation,
                component_name=self.name,
                component_type="Formatter",
                error_class=FormatterError,
            )

            # Update statistics
            end_time = time.time()
            execution_time = end_time - start_time

            # Update format count
            format_count = self._state_manager.get_metadata("format_count", 0)
            self._state_manager.set_metadata("format_count", format_count + 1)

            # Update average execution time
            avg_time = self._state_manager.get_metadata("avg_execution_time", 0)
            new_avg = ((avg_time * format_count) + execution_time) / (format_count + 1)
            self._state_manager.set_metadata("avg_execution_time", new_avg)

            # Update max execution time if needed
            max_time = self._state_manager.get_metadata("max_execution_time", 0)
            if execution_time > max_time:
                self._state_manager.set_metadata("max_execution_time", execution_time)

            # Cache result if caching is enabled
            if self._state_manager.get("cache_enabled", True):
                cache = self._state_manager.get("cache", {})
                cache_key = f"{output[:50]}_{len(output)}"
                cache[cache_key] = result
                self._state_manager.update("cache", cache)

            return result

        except Exception as e:
            # Update error statistics
            error_count = self._state_manager.get_metadata("error_count", 0)
            self._state_manager.set_metadata("error_count", error_count + 1)
            self._state_manager.set_metadata("last_error", str(e))
            self._state_manager.set_metadata("last_error_time", time.time())

            # Re-raise as FormatterError
            if isinstance(e, FormatterError):
                raise e
            raise FormatterError(f"Formatting failed: {str(e)}")

    async def format_async(self, output: str, validation_results: List[ValidationResult]) -> Any:
        """
        Format a result asynchronously.

        Args:
            output: The output to format
            validation_results: The validation results to include

        Returns:
            The formatted result

        Raises:
            FormatterError: If formatting fails
        """
        # Ensure adapter is initialized
        if not self._state_manager.get("initialized", False):
            self._initialize_state()

        # If no formatter is provided, return a default result
        if self._formatter is None:
            from ..chain.result import ChainResult

            return ChainResult(
                output=output,
                validation_results=validation_results,
                prompt="",
                execution_time=0.0,
                attempt_count=1,
                metadata={},
            )

        # Record start time
        start_time = time.time()

        try:
            # Check if formatter has async methods
            if hasattr(self._formatter, "format_async"):
                result = await self._formatter.format_async(output, validation_results)
            elif hasattr(self._formatter, "format_result_async"):
                result = await self._formatter.format_result_async(output, validation_results)
            elif hasattr(self._formatter, "process_async"):
                result = await self._formatter.process_async(output, validation_results)
            elif hasattr(self._formatter, "run_async"):
                result = await self._formatter.run_async(output, validation_results)
            else:
                # Fall back to running synchronous method in executor
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, self.format, output, validation_results)
                # Return early since statistics are updated in the synchronous method
                return result

            # Update statistics
            end_time = time.time()
            execution_time = end_time - start_time

            # Update format count
            format_count = self._state_manager.get_metadata("format_count", 0)
            self._state_manager.set_metadata("format_count", format_count + 1)

            # Update average execution time
            avg_time = self._state_manager.get_metadata("avg_execution_time", 0)
            new_avg = ((avg_time * format_count) + execution_time) / (format_count + 1)
            self._state_manager.set_metadata("avg_execution_time", new_avg)

            # Update max execution time if needed
            max_time = self._state_manager.get_metadata("max_execution_time", 0)
            if execution_time > max_time:
                self._state_manager.set_metadata("max_execution_time", execution_time)

            # Cache result if caching is enabled
            if self._state_manager.get("cache_enabled", True):
                cache = self._state_manager.get("cache", {})
                cache_key = f"{output[:50]}_{len(output)}"
                cache[cache_key] = result
                self._state_manager.update("cache", cache)

            return result

        except Exception as e:
            # Update error statistics
            error_count = self._state_manager.get_metadata("error_count", 0)
            self._state_manager.set_metadata("error_count", error_count + 1)
            self._state_manager.set_metadata("last_error", str(e))
            self._state_manager.set_metadata("last_error_time", time.time())

            # Re-raise as FormatterError
            if isinstance(e, FormatterError):
                raise e
            raise FormatterError(f"Async formatting failed: {str(e)}")
