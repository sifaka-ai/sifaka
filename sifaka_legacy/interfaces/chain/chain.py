"""
Chain interface for Sifaka.

This module defines the interface for chains in the Sifaka framework.
These interfaces establish a common contract for chain behavior, enabling better
modularity and extensibility.

## Interface Hierarchy

1. **Chain**: Base interface for all chains

## Usage

These interfaces are defined using Python's Protocol class from typing,
which enables structural subtyping. This means that classes don't need to
explicitly inherit from these interfaces; they just need to implement the
required methods and properties.

## State Management

The interfaces support standardized state management:
- Single _state_manager attribute for all mutable state
- State initialization during construction
- State access through state manager methods
- Clear separation of configuration and state

## Error Handling

The interfaces define error handling patterns:
- ValueError for invalid inputs
- RuntimeError for execution failures
- TypeError for type mismatches
- ModelError: Raised when text generation fails
- ValidationError: Raised when validation fails
- ImproverError: Raised when improvement fails
- FormatterError: Raised when formatting fails
- Detailed error tracking and reporting

## Execution Tracking

The interfaces support execution tracking:
- Execution count tracking
- Execution time tracking
- Success/failure tracking
- Performance statistics
"""

from abc import abstractmethod
from typing import Any, Dict, Protocol, TypeVar, runtime_checkable

from sifaka.interfaces.core import ConfigurableProtocol, IdentifiableProtocol

# Type variables
ConfigType = TypeVar("ConfigType")
InputType = TypeVar("InputType", contravariant=True)
OutputType = TypeVar("OutputType", covariant=True)


@runtime_checkable
class Chain(
    IdentifiableProtocol,
    ConfigurableProtocol[ConfigType],
    Protocol[ConfigType, InputType, OutputType],
):
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
    5. **State Management**: Manage chain state
    6. **Error Handling**: Handle and track errors
    7. **Execution Tracking**: Track execution statistics
    8. **Cleanup**: Release resources when no longer needed

    ## Implementation Requirements

    Classes implementing this interface must:
    - Provide a run method to run inputs through the flow
    - Return standardized results
    - Provide name and description properties
    - Provide a config property to access the chain configuration
    - Provide an update_config method to update the chain configuration
    - Implement state management using _state_manager
    - Implement error handling and tracking
    - Implement execution tracking and statistics
    """

    @property
    @abstractmethod
    def _state_manager(self) -> Any:
        """
        Get the state manager.

        Returns:
            The state manager
        """

    @abstractmethod
    def initialize(self) -> None:
        """
        Initialize the chain.

        This method should be called after the chain is created to set up
        any resources or state needed for operation.

        Raises:
            RuntimeError: If initialization fails
        """

    @abstractmethod
    def cleanup(self) -> None:
        """
        Clean up the chain.

        This method should be called when the chain is no longer needed to
        release any resources it holds.

        Raises:
            RuntimeError: If cleanup fails
        """

    @abstractmethod
    def get_state(self) -> Dict[str, Any]:
        """
        Get the current state.

        Returns:
            The current state
        """

    @abstractmethod
    def set_state(self, state: Dict[str, Any]) -> None:
        """
        Set the state.

        Args:
            state: The new state

        Raises:
            ValueError: If the state is invalid
        """

    @abstractmethod
    def reset_state(self) -> None:
        """
        Reset the state to its initial values.

        Raises:
            RuntimeError: If state reset fails
        """

    @abstractmethod
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get execution statistics.

        Returns:
            A dictionary of execution statistics
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

    @property
    def name(self) -> str:
        """
        Get the chain name.

        Returns:
            The name of the chain
        """
        ...
