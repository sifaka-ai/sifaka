"""
Chain Interfaces Module

This module defines the core interfaces for the Sifaka chain system.
These interfaces establish a common contract for component behavior,
enabling better modularity, extensibility, and interoperability.

## Overview
The interfaces in this module form the foundation of the chain system's
component architecture. They define the contracts that components must
adhere to, ensuring consistent behavior and interoperability between
different implementations. These interfaces use Python's Protocol class
to enable structural subtyping, allowing components to implement the
interfaces without explicit inheritance.

## Components
1. **ChainComponent**: Base interface for all chain components
2. **ValidationResult**: Result model for validation operations
3. **Model**: Interface for text generation models
4. **Validator**: Interface for output validators
5. **Improver**: Interface for output improvers
6. **Formatter**: Interface for result formatters
7. **Plugin**: Interface for chain plugins

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

## Error Handling
The interfaces define consistent error handling patterns:
- ModelError: Raised when text generation fails
- ValidationError: Raised when validation fails
- ImproverError: Raised when improvement fails
- FormatterError: Raised when formatting fails

## Configuration
The interfaces are designed to work with the following components:
- Pydantic models for data validation and serialization
- Protocol classes for interface definition
- Asynchronous methods with default implementations
"""

from abc import abstractmethod
from typing import Any, Dict, List, Protocol, TypeVar, runtime_checkable
from pydantic import BaseModel, Field

# Type variables
T = TypeVar("T")


@runtime_checkable
class ChainComponent(Protocol):
    """
    Base interface for all chain components.

    This interface defines the common properties that all chain components
    must implement. It serves as the foundation for the component hierarchy
    in the chain system, ensuring consistent identification and description
    of components.

    ## Architecture
    The ChainComponent interface uses Python's Protocol class to define
    a structural interface that components can implement without explicit
    inheritance. This enables better flexibility and composition in the
    component architecture.

    ## Lifecycle
    Components implementing this interface should maintain consistent
    name and description properties throughout their lifecycle.

    ## Examples
    ```python
    class MyComponent:
        @property
        def name(self) -> str:
            return "my_component"

        @property
        def description(self) -> str:
            return "A custom chain component"
    ```
    """

    @property
    def name(self) -> str:
        """
        Get the component name.

        Returns:
            str: A unique identifier for the component
        """
        ...

    @property
    def description(self) -> str:
        """
        Get the component description.

        Returns:
            str: A human-readable description of the component
        """
        ...


class ValidationResult(BaseModel):
    """
    Result of a validation operation.

    This class represents the result of validating an output against a rule or
    validator. It includes information about whether the validation passed,
    a descriptive message, a numerical score, and optional lists of issues
    and improvement suggestions.

    ## Architecture
    The ValidationResult class is designed as an immutable Pydantic model
    with fields for storing validation results. It follows a value object
    pattern, where instances represent the outcome of a validation operation.

    ## Lifecycle
    1. Creation: Instantiated with validation outcome
    2. Usage: Accessed to retrieve validation status and details

    ## Examples
    ```python
    # Create a validation result
    result = ValidationResult(
        passed=True,
        message="Length validation passed",
        score=1.0
    )

    # Create a failed validation with issues and suggestions
    failed_result = ValidationResult(
        passed=False,
        message="Content validation failed",
        score=0.3,
        issues=["Contains inappropriate content"],
        suggestions=["Remove inappropriate content"]
    )

    # Check validation status
    if result.passed:
        print("Validation passed!")
    else:
        print(f"Validation failed: {result.message}")
        print(f"Issues: {result.issues}")
        print(f"Suggestions: {result.suggestions}")
    ```

    Attributes:
        passed (bool): Whether the validation passed
        message (str): Validation message
        score (float): Validation score (0.0 to 1.0)
        issues (List[str]): List of issues found
        suggestions (List[str]): List of improvement suggestions
        metadata (Dict[str, Any]): Additional metadata
    """

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

        loop = asyncio and asyncio.get_event_loop()
        return await loop and loop.run_in_executor(None, self.generate, prompt)


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

        loop = asyncio and asyncio.get_event_loop()
        return await loop and loop.run_in_executor(None, self.validate, output)


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

        loop = asyncio and asyncio.get_event_loop()
        return await loop and loop.run_in_executor(
            None, lambda: self.improve(output, validation_results)
        )


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

        loop = asyncio and asyncio.get_event_loop()
        return await loop and loop.run_in_executor(
            None, lambda: self.format(output, validation_results)
        )


# Import the core Plugin interface
from sifaka.core.interfaces import Plugin as CorePlugin


@runtime_checkable
class Plugin(ChainComponent, CorePlugin, Protocol):
    """
    Interface for chain plugins.

    This interface extends the core Plugin interface with chain-specific
    functionality. It ensures that chain plugins can be discovered, registered,
    and used consistently with other plugins in the Sifaka framework.
    """

    pass
