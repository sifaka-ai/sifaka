"""
Chain protocol interfaces for Sifaka.

This module defines the interfaces for chains in the Sifaka framework.
These interfaces establish a common contract for chain behavior, enabling better
modularity and extensibility.
"""

from abc import abstractmethod
from typing import Any, Dict, Generic, Protocol, TypeVar, runtime_checkable

from ...interfaces.core import Configurable, Identifiable

# Type variables
InputType = TypeVar("InputType", contravariant=True)
OutputType = TypeVar("OutputType", covariant=True)
ConfigType = TypeVar("ConfigType")


@runtime_checkable
class Chain(Identifiable, Configurable[ConfigType], Protocol[InputType, OutputType]):
    """
    Interface for chains.

    This interface defines the contract for components that orchestrate the
    validation and improvement flow between models, rules, and critics. It ensures
    that chains can run inputs through the flow and return standardized results.

    ## Lifecycle

    1. **Initialization**: Set up chain resources and configuration
    2. **Execution**: Run inputs through the flow
    3. **Result Handling**: Process and return results
    4. **Configuration Management**: Manage chain configuration
    5. **Cleanup**: Release resources when no longer needed

    ## Implementation Requirements

    Classes implementing this interface must:
    - Provide a run method to run inputs through the flow
    - Return standardized results
    - Provide name and description properties
    - Provide a config property to access the chain configuration
    - Provide an update_config method to update the chain configuration
    """

    @abstractmethod
    def run(self, input_value: InputType, **kwargs: Any) -> OutputType:
        """
        Run an input value through the chain.

        Args:
            input_value: The input value to run
            **kwargs: Additional run parameters

        Returns:
            A chain result

        Raises:
            ValueError: If the input value is invalid
            RuntimeError: If the chain execution fails
        """
        pass


@runtime_checkable
class AsyncChain(Protocol[InputType, OutputType]):
    """
    Interface for asynchronous chains.

    This interface defines the contract for components that orchestrate the
    validation and improvement flow between models, rules, and critics asynchronously.
    It ensures that chains can run inputs through the flow asynchronously and return
    standardized results.

    ## Lifecycle

    1. **Initialization**: Set up chain resources and configuration
    2. **Execution**: Run inputs through the flow asynchronously
    3. **Result Handling**: Process and return results
    4. **Configuration Management**: Manage chain configuration
    5. **Cleanup**: Release resources when no longer needed

    ## Implementation Requirements

    Classes implementing this interface must:
    - Provide an async run method to run inputs through the flow
    - Return standardized results
    - Provide name and description properties
    - Provide a config property to access the chain configuration
    - Provide an update_config method to update the chain configuration
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Get the chain name.

        Returns:
            The chain name
        """
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """
        Get the chain description.

        Returns:
            The chain description
        """
        pass

    @abstractmethod
    async def run(self, input_value: InputType, **kwargs: Any) -> OutputType:
        """
        Run an input value through the chain asynchronously.

        Args:
            input_value: The input value to run
            **kwargs: Additional run parameters

        Returns:
            A chain result

        Raises:
            ValueError: If the input value is invalid
            RuntimeError: If the chain execution fails
        """
        pass
