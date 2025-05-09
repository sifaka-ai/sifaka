"""
Core Interfaces Module

A module that defines the fundamental interfaces for the Sifaka framework.

## Overview
This module provides the core interfaces that establish the contract for all components
in the Sifaka framework. These interfaces ensure consistent behavior, enable better
modularity, and support extensibility across the framework.

## Components
1. Base Interfaces:
   - Component: Base interface for all components
   - Configurable: Interface for components with configuration
   - Stateful: Interface for components with state management
   - Identifiable: Interface for components with identity
   - Loggable: Interface for components with logging capabilities
   - Traceable: Interface for components with tracing capabilities

## Usage Examples
```python
from sifaka.core import Component, Configurable, Stateful

class MyComponent(Component, Configurable, Stateful):
    def initialize(self) -> None:
        # Initialize resources
        pass

    def cleanup(self) -> None:
        # Clean up resources
        pass

    @property
    def config(self):
        return self._config

    def update_config(self, config):
        # Update configuration
        pass

    def get_state(self):
        return self._state

    def set_state(self, state):
        # Update state
        pass

    def reset_state(self):
        # Reset state
        pass
```

## Error Handling
The interfaces define error handling patterns:
- RuntimeError for initialization and cleanup failures
- ValueError for invalid configuration or state
- Type checking for configuration and state objects
- Resource cleanup in cleanup methods

## Configuration
The interfaces support various configuration options:
- Component initialization parameters
- Configuration object structure
- State management patterns
- Logging and tracing capabilities
"""

from abc import abstractmethod
from typing import Any, Dict, Generic, Optional, Protocol, TypeVar, runtime_checkable

# Type variables
T = TypeVar("T")
ConfigType = TypeVar("ConfigType")
StateType = TypeVar("StateType")


@runtime_checkable
class Component(Protocol):
    """
    Base interface for all components in Sifaka.

    This interface defines the minimal contract that all components must fulfill.
    It ensures that components can be initialized, used, and cleaned up properly.

    ## Lifecycle

    1. **Initialization**: Component is created with required parameters
    2. **Operation**: Component performs its designated function
    3. **Cleanup**: Component releases resources when no longer needed

    ## Implementation Requirements

    Classes implementing this interface must:
    - Provide an initialize method to set up resources
    - Provide a cleanup method to release resources
    - Handle errors appropriately in all methods
    """

    @abstractmethod
    def initialize(self) -> None:
        """
        Initialize the component.

        This method should be called after the component is created to set up
        any resources or state needed for operation.

        Raises:
            RuntimeError: If initialization fails
        """
        pass

    @abstractmethod
    def cleanup(self) -> None:
        """
        Clean up the component.

        This method should be called when the component is no longer needed to
        release any resources it holds.

        Raises:
            RuntimeError: If cleanup fails
        """
        pass


@runtime_checkable
class Configurable(Protocol[ConfigType]):
    """
    Interface for components with configuration.

    This interface defines the contract for components that can be configured
    with a configuration object. It ensures that components can expose their
    configuration and update it when needed.

    ## Lifecycle

    1. **Configuration Access**: Get the current configuration
    2. **Configuration Update**: Update the configuration with new values
    3. **Configuration Validation**: Validate configuration changes

    ## Implementation Requirements

    Classes implementing this interface must:
    - Provide a config property to access the current configuration
    - Provide an update_config method to update the configuration
    - Validate configuration changes before applying them
    """

    @property
    @abstractmethod
    def config(self) -> ConfigType:
        """
        Get the current configuration.

        Returns:
            The current configuration object
        """
        pass

    @abstractmethod
    def update_config(self, config: ConfigType) -> None:
        """
        Update the configuration.

        Args:
            config: The new configuration object or values to update

        Raises:
            ValueError: If the configuration is invalid
        """
        pass


@runtime_checkable
class Stateful(Protocol[StateType]):
    """
    Interface for components with state management.

    This interface defines the contract for components that maintain internal
    state. It ensures that components can expose their state, update it, and
    reset it when needed.

    ## Lifecycle

    1. **State Access**: Get the current state
    2. **State Update**: Update the state with new values
    3. **State Reset**: Reset the state to its initial values

    ## Implementation Requirements

    Classes implementing this interface must:
    - Provide a get_state method to access the current state
    - Provide a set_state method to update the state
    - Provide a reset_state method to reset the state
    """

    @abstractmethod
    def get_state(self) -> StateType:
        """
        Get the current state.

        Returns:
            The current state object
        """
        pass

    @abstractmethod
    def set_state(self, state: StateType) -> None:
        """
        Set the state.

        Args:
            state: The new state object or values to update

        Raises:
            ValueError: If the state is invalid
        """
        pass

    @abstractmethod
    def reset_state(self) -> None:
        """
        Reset the state to its initial values.

        Raises:
            RuntimeError: If state reset fails
        """
        pass


@runtime_checkable
class Identifiable(Protocol):
    """
    Interface for components with identity.

    This interface defines the contract for components that have a unique
    identity. It ensures that components can be identified and described.

    ## Lifecycle

    1. **Identity Access**: Get the component's name and description
    2. **Identity Verification**: Verify the component's identity

    ## Implementation Requirements

    Classes implementing this interface must:
    - Provide a name property to access the component's name
    - Provide a description property to access the component's description
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Get the component's name.

        Returns:
            The component's name
        """
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """
        Get the component's description.

        Returns:
            The component's description
        """
        pass


@runtime_checkable
class Loggable(Protocol):
    """
    Interface for components with logging capabilities.

    This interface defines the contract for components that can log messages.
    It ensures that components can log messages at different levels and with
    different contexts.

    ## Lifecycle

    1. **Logger Access**: Get the component's logger
    2. **Message Logging**: Log messages at different levels

    ## Implementation Requirements

    Classes implementing this interface must:
    - Provide a logger property to access the component's logger
    - Provide methods to log messages at different levels
    """

    @property
    @abstractmethod
    def logger(self) -> Any:
        """
        Get the component's logger.

        Returns:
            The component's logger
        """
        pass

    @abstractmethod
    def log(self, level: str, message: str, **kwargs: Any) -> None:
        """
        Log a message.

        Args:
            level: The log level (e.g., "debug", "info", "warning", "error")
            message: The message to log
            **kwargs: Additional context for the log message

        Raises:
            ValueError: If the log level is invalid
        """
        pass


@runtime_checkable
class Traceable(Protocol):
    """
    Interface for components with tracing capabilities.

    This interface defines the contract for components that can trace their
    operations. It ensures that components can start, end, and annotate traces.

    ## Lifecycle

    1. **Tracer Access**: Get the component's tracer
    2. **Trace Management**: Start, end, and annotate traces

    ## Implementation Requirements

    Classes implementing this interface must:
    - Provide a tracer property to access the component's tracer
    - Provide methods to start, end, and annotate traces
    """

    @property
    @abstractmethod
    def tracer(self) -> Any:
        """
        Get the component's tracer.

        Returns:
            The component's tracer
        """
        pass

    @abstractmethod
    def start_trace(self, name: str, **kwargs: Any) -> Any:
        """
        Start a new trace.

        Args:
            name: The name of the trace
            **kwargs: Additional context for the trace

        Returns:
            A trace context that can be used to end the trace

        Raises:
            RuntimeError: If trace start fails
        """
        pass

    @abstractmethod
    def end_trace(self, trace: Any, **kwargs: Any) -> None:
        """
        End a trace.

        Args:
            trace: The trace context returned by start_trace
            **kwargs: Additional context for the trace

        Raises:
            RuntimeError: If trace end fails
        """
        pass

    @abstractmethod
    def annotate_trace(self, trace: Any, key: str, value: Any) -> None:
        """
        Annotate a trace with additional information.

        Args:
            trace: The trace context returned by start_trace
            key: The annotation key
            value: The annotation value

        Raises:
            RuntimeError: If trace annotation fails
        """
        pass
