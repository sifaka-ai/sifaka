"""
Chain Interface Module

Protocol interfaces for Sifaka's chain system.

## Overview
This module defines the core protocol interfaces for chains in the Sifaka
framework. These interfaces establish a common contract for chain behavior,
enabling better modularity, extensibility, and interoperability between
different chain implementations.

## Components
1. **Chain**: Base synchronous chain interface
   - Input processing
   - Result generation
   - Configuration management
   - Resource handling

2. **AsyncChain**: Asynchronous chain interface
   - Async input processing
   - Async result generation
   - Configuration management
   - Resource handling

## Usage Examples
```python
from typing import Any
from sifaka.chain.interfaces.chain import Chain, AsyncChain
from sifaka.chain.result import ChainResult

# Synchronous chain implementation
class SimpleChain(Chain[str, str]):
    @property
    def name(self) -> str:
        return "SimpleChain"

    @property
    def description(self) -> str:
        return "A simple chain implementation"

    @property
    def config(self) -> Dict[str, Any]:
        return {"max_attempts": 3}

    def update_config(self, config: Dict[str, Any]) -> None:
        # Update configuration
        pass

    def run(self, input_value: str, **kwargs: Any) -> str:
        # Process input and return result
        return f"Processed: {input_value}"

# Asynchronous chain implementation
class AsyncSimpleChain(AsyncChain[str, str]):
    @property
    def name(self) -> str:
        return "AsyncSimpleChain"

    @property
    def description(self) -> str:
        return "An async chain implementation"

    async def run(self, input_value: str, **kwargs: Any) -> str:
        # Process input asynchronously
        return f"Processed async: {input_value}"
```

## Error Handling
- ValueError: Raised for invalid input values
- RuntimeError: Raised for chain execution failures
- ConfigurationError: Raised for invalid configurations

## Configuration
- name: Chain identifier
- description: Chain description
- config: Chain configuration dictionary
- run_kwargs: Additional run parameters
"""

from abc import abstractmethod
from typing import Any, Dict, Generic, Protocol, TypeVar, runtime_checkable

from ...core.interfaces import Configurable, Identifiable

# Type variables
InputType = TypeVar("InputType", contravariant=True)
OutputType = TypeVar("OutputType", covariant=True)
ConfigType = TypeVar("ConfigType")


@runtime_checkable
class Chain(Identifiable, Configurable[ConfigType], Protocol[InputType, OutputType]):
    """
    Interface for synchronous chains.

    Detailed description of what the class does, including:
    - Defines the contract for components that orchestrate validation and improvement flow
    - Ensures consistent behavior across different chain implementations
    - Handles input processing, flow orchestration, and result generation
    - Manages chain configuration and resources

    Type parameters:
        InputType: The type of input accepted by the chain
        OutputType: The type of output produced by the chain
        ConfigType: The type of configuration used by the chain

    Example:
        ```python
        class SimpleChain(Chain[str, str]):
            @property
            def name(self) -> str:
                return "SimpleChain"

            @property
            def description(self) -> str:
                return "A simple chain implementation"

            @property
            def config(self) -> Dict[str, Any]:
                return {"max_attempts": 3}

            def update_config(self, config: Dict[str, Any]) -> None:
                # Update configuration
                pass

            def run(self, input_value: str, **kwargs: Any) -> str:
                # Process input and return result
                return f"Processed: {input_value}"
        ```
    """

    @abstractmethod
    def run(self, input_value: InputType, **kwargs: Any) -> OutputType:
        """
        Run an input value through the chain.

        Detailed description of what the method does, including:
        - Processes an input value through the chain's validation and improvement flow
        - Coordinates the interaction between models, rules, and critics
        - Produces a standardized result with metadata
        - Handles errors and validation failures

        Args:
            input_value: The input value to process
            **kwargs: Additional run parameters

        Returns:
            The processed output

        Raises:
            ValueError: If the input value is invalid
            RuntimeError: If the chain execution fails

        Example:
            ```python
            # Run the chain
            result = chain.run("Write a short story")
            print(f"Output: {result}")
            ```
        """
        pass


@runtime_checkable
class AsyncChain(Protocol[InputType, OutputType]):
    """
    Interface for asynchronous chains.

    Detailed description of what the class does, including:
    - Defines the contract for components that orchestrate validation and improvement flow asynchronously
    - Ensures consistent behavior across different asynchronous chain implementations
    - Handles asynchronous input processing, flow orchestration, and result generation
    - Manages chain configuration and resources

    Type parameters:
        InputType: The type of input accepted by the chain
        OutputType: The type of output produced by the chain

    Example:
        ```python
        class AsyncSimpleChain(AsyncChain[str, str]):
            @property
            def name(self) -> str:
                return "AsyncSimpleChain"

            @property
            def description(self) -> str:
                return "An async chain implementation"

            async def run(self, input_value: str, **kwargs: Any) -> str:
                # Process input asynchronously
                return f"Processed async: {input_value}"
        ```
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Get the chain name.

        Detailed description of what the method does, including:
        - Returns the unique identifier for the chain
        - Used for logging and error reporting
        - Should be descriptive and unique

        Returns:
            The chain name

        Example:
            ```python
            # Get the chain name
            name = chain.name
            print(f"Chain name: {name}")
            ```
        """
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """
        Get the chain description.

        Detailed description of what the method does, including:
        - Returns a human-readable description of the chain
        - Used for documentation and logging
        - Should be clear and informative

        Returns:
            The chain description

        Example:
            ```python
            # Get the chain description
            description = chain.description
            print(f"Chain description: {description}")
            ```
        """
        pass

    @abstractmethod
    async def run(self, input_value: InputType, **kwargs: Any) -> OutputType:
        """
        Run an input value through the chain asynchronously.

        Detailed description of what the method does, including:
        - Processes an input value through the chain's validation and improvement flow asynchronously
        - Coordinates the asynchronous interaction between models, rules, and critics
        - Produces a standardized result with metadata
        - Handles errors and validation failures

        Args:
            input_value: The input value to process
            **kwargs: Additional run parameters

        Returns:
            The processed output

        Raises:
            ValueError: If the input value is invalid
            RuntimeError: If the chain execution fails

        Example:
            ```python
            # Run the chain asynchronously
            result = await chain.run("Write a short story")
            print(f"Output: {result}")
            ```
        """
        pass
