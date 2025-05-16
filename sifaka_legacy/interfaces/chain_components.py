"""
Chain Component Interfaces Module

This module defines the interfaces for chain components in the Sifaka framework.
These interfaces establish a common contract for chain component behavior,
enabling better modularity, extensibility, and interoperability.

## Overview
The interfaces in this module form the foundation of the chain system's
component architecture. They define the contracts that components must
adhere to, ensuring consistent behavior and interoperability between
different implementations. These interfaces use Python's Protocol class
to enable structural subtyping, allowing components to implement the
interfaces without explicit inheritance.

## Components
1. **ValidationResult**: Result model for validation operations
2. **Model**: Interface for text generation models
3. **Validator**: Interface for output validators
4. **Improver**: Interface for output improvers
5. **Formatter**: Interface for result formatters

## Error Handling
The interfaces define consistent error handling patterns:
- ModelError: Raised when text generation fails
- ValidationError: Raised when validation fails
- ImproverError: Raised when improvement fails
- FormatterError: Raised when formatting fails
"""

from abc import abstractmethod
from typing import Any, Dict, List, Optional, Protocol, TypeVar, Union, runtime_checkable
from pydantic import BaseModel, Field

from .component import ComponentProtocol
from sifaka.interfaces.core import PluginProtocol as CorePlugin

# Type variables with variance annotations
T = TypeVar("T")  # Generic type for results
InputT = TypeVar("InputT", contravariant=True)  # Input type (contravariant)
OutputT = TypeVar("OutputT", covariant=True)  # Output type (covariant)
PluginT = TypeVar("PluginT", covariant=True)  # Plugin type (covariant)


class ValidationResult(BaseModel):
    """
    Result of a validation operation.

    This class represents the result of validating an output against a rule or
    validator. It includes information about whether the validation passed,
    a descriptive message, a numerical score, and optional lists of issues
    and improvement suggestions.

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
class ModelProtocol(ComponentProtocol, Protocol[InputT, OutputT]):
    """
    Interface for text generation models.

    This protocol defines the interface for models that generate text from prompts.
    It uses contravariant input type and covariant output type to ensure proper
    type checking when used in generic contexts.
    """

    @abstractmethod
    def generate(self, prompt: InputT) -> OutputT:
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


@runtime_checkable
class ValidatorProtocol(ComponentProtocol, Protocol[InputT]):
    """
    Interface for output validators.

    This protocol defines the interface for validators that check outputs
    against rules or criteria. It uses a contravariant input type to ensure
    proper type checking when used in generic contexts.
    """

    @abstractmethod
    def validate(self, output: InputT) -> ValidationResult:
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


@runtime_checkable
class ImproverProtocol(ComponentProtocol, Protocol[InputT, OutputT]):
    """
    Interface for output improvers.

    This protocol defines the interface for improvers that enhance outputs
    based on validation results. It uses contravariant input type and covariant
    output type to ensure proper type checking when used in generic contexts.
    """

    @abstractmethod
    def improve(self, output: InputT, validation_results: List[ValidationResult]) -> OutputT:
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


@runtime_checkable
class FormatterProtocol(ComponentProtocol, Protocol[InputT, OutputT]):
    """
    Interface for result formatters.

    This protocol defines the interface for formatters that transform outputs
    into different formats. It uses contravariant input type and covariant
    output type to ensure proper type checking when used in generic contexts.
    """

    @abstractmethod
    def format(self, output: InputT, validation_results: List[ValidationResult]) -> OutputT:
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


@runtime_checkable
class ChainPluginProtocol(ComponentProtocol, CorePlugin, Protocol[PluginT]):
    """
    Interface for chain plugins.

    This interface extends the core Plugin interface with chain-specific
    functionality. It ensures that chain plugins can be discovered, registered,
    and used consistently with other plugins in the Sifaka framework.
    """

    pass
