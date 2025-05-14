"""
Validator interface for Sifaka.

This module defines the interface for output validators in the Sifaka framework.
These interfaces establish a common contract for validator behavior, enabling better
modularity and extensibility.

## Interface Hierarchy

1. **Validator**: Interface for output validators

## Usage

These interfaces are defined using Python's Protocol class from typing,
which enables structural subtyping. This means that classes don't need to
explicitly inherit from these interfaces; they just need to implement the
required methods and properties.
"""

from abc import abstractmethod
from typing import Protocol, runtime_checkable

from ..base import ChainComponent
from ..models import ValidationResult


@runtime_checkable
class Validator(ChainComponent, Protocol):
    """
    Interface for output validators.

    This interface defines the contract for components that validate outputs.
    It ensures that validators can validate outputs against rules and provide
    consistent validation results.

    ## Lifecycle

    1. **Initialization**: Set up validator resources
    2. **Validation**: Validate outputs against rules
    3. **Cleanup**: Release resources when no longer needed

    ## Implementation Requirements

    Classes implementing this interface must:
    - Provide a validate method to validate outputs
    - Return standardized ValidationResult objects
    - Optionally provide an async validate_async method for asynchronous validation
    """

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
