"""
Chain Interfaces Module

This module defines the core interfaces for the Sifaka chain system.
These interfaces establish a common contract for component behavior,
enabling better modularity, extensibility, and interoperability.

## Interface Hierarchy

1. **Model**: Interface for text generation models
2. **Validator**: Interface for output validators
3. **Improver**: Interface for output improvers
4. **Formatter**: Interface for result formatters
5. **Plugin**: Interface for plugins

## Usage Examples

```python
from typing import List
from sifaka.chain.interfaces import Model, Validator, ValidationResult

class SimpleModel(Model):
    def generate(self, prompt: str) -> str:
        return f"Generated: {prompt}"

class LengthValidator(Validator):
    def __init__(self, min_length: int, max_length: int):
        self.min_length = min_length
        self.max_length = max_length

    def validate(self, output: str) -> ValidationResult:
        length = len(output)
        if length < self.min_length:
            return ValidationResult(
                passed=False,
                message=f"Output too short: {length} < {self.min_length}",
                score=0.0
            )
        if length > self.max_length:
            return ValidationResult(
                passed=False,
                message=f"Output too long: {length} > {self.max_length}",
                score=0.0
            )
        return ValidationResult(
            passed=True,
            message="Length validation passed",
            score=1.0
        )
```
"""

from abc import abstractmethod
from typing import Any, Dict, List, Optional, Protocol, TypeVar, runtime_checkable
from pydantic import BaseModel, Field

# Type variables
T = TypeVar("T")


@runtime_checkable
class ChainComponent(Protocol):
    """Base interface for all chain components."""

    @property
    def name(self) -> str:
        """Get the component name."""
        ...

    @property
    def description(self) -> str:
        """Get the component description."""
        ...


class ValidationResult(BaseModel):
    """Result of a validation operation."""

    passed: bool = Field(description="Whether the validation passed")
    message: str = Field(description="Validation message")
    score: float = Field(default=0.0, ge=0.0, le=1.0, description="Validation score")
    issues: List[str] = Field(default_factory=list, description="List of issues found")
    suggestions: List[str] = Field(
        default_factory=list, description="List of improvement suggestions"
    )
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


@runtime_checkable
class Model(ChainComponent, Protocol):
    """Interface for text generation models."""

    @abstractmethod
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
        ...

    async def generate_async(self, prompt: str) -> str:
        """
        Generate text asynchronously.

        This method has a default implementation that calls the synchronous
        generate method in an executor. Implementations can override this
        method to provide a more efficient asynchronous implementation.

        Args:
            prompt: The prompt to generate text from

        Returns:
            The generated text

        Raises:
            ModelError: If text generation fails
        """
        import asyncio

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.generate, prompt)


@runtime_checkable
class Validator(ChainComponent, Protocol):
    """Interface for output validators."""

    @abstractmethod
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
        ...

    async def validate_async(self, output: str) -> ValidationResult:
        """
        Validate an output asynchronously.

        This method has a default implementation that calls the synchronous
        validate method in an executor. Implementations can override this
        method to provide a more efficient asynchronous implementation.

        Args:
            output: The output to validate

        Returns:
            The validation result

        Raises:
            ValidationError: If validation fails
        """
        import asyncio

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.validate, output)


@runtime_checkable
class Improver(ChainComponent, Protocol):
    """Interface for output improvers."""

    @abstractmethod
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
        ...

    async def improve_async(self, output: str, validation_results: List[ValidationResult]) -> str:
        """
        Improve an output asynchronously.

        This method has a default implementation that calls the synchronous
        improve method in an executor. Implementations can override this
        method to provide a more efficient asynchronous implementation.

        Args:
            output: The output to improve
            validation_results: The validation results to use for improvement

        Returns:
            The improved output

        Raises:
            ImproverError: If improvement fails
        """
        import asyncio

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.improve(output, validation_results))


@runtime_checkable
class Formatter(ChainComponent, Protocol):
    """Interface for result formatters."""

    @abstractmethod
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
        ...

    async def format_async(self, output: str, validation_results: List[ValidationResult]) -> Any:
        """
        Format a result asynchronously.

        This method has a default implementation that calls the synchronous
        format method in an executor. Implementations can override this
        method to provide a more efficient asynchronous implementation.

        Args:
            output: The output to format
            validation_results: The validation results to include

        Returns:
            The formatted result

        Raises:
            FormatterError: If formatting fails
        """
        import asyncio

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.format(output, validation_results))


@runtime_checkable
class Plugin(ChainComponent, Protocol):
    """Interface for plugins."""

    @property
    @abstractmethod
    def version(self) -> str:
        """Get the plugin version."""
        ...

    @property
    @abstractmethod
    def component_type(self) -> str:
        """Get the component type this plugin provides."""
        ...

    @abstractmethod
    def create_component(self, config: Dict[str, Any]) -> Any:
        """
        Create a component instance.

        Args:
            config: The component configuration

        Returns:
            The component instance

        Raises:
            PluginError: If component creation fails
        """
        ...
