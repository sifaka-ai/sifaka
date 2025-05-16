"""
Core interfaces for Sifaka.

This module defines the core interfaces that all components in the Sifaka framework
implement. These interfaces establish a common contract for component behavior,
enabling better modularity and extensibility.

## Interface Hierarchy

1. **IdentifiableProtocol**: Interface for components with identity
2. **ConfigurableProtocol**: Interface for components with configuration
3. **StatefulProtocol**: Interface for components with state management
4. **LoggableProtocol**: Interface for components with logging capabilities
5. **TraceableProtocol**: Interface for components with tracing capabilities
6. **ComponentProtocol**: Interface for all components (combines other interfaces)
7. **PluginProtocol**: Interface for plugins that extend component functionality

## Usage

These interfaces are defined using Python's Protocol class from typing,
which enables structural subtyping. This means that classes don't need to
explicitly inherit from these interfaces; they just need to implement the
required methods and properties.
"""

from abc import abstractmethod
from typing import Any, Dict, Generic, Protocol, TypeVar, runtime_checkable

# Type variables
ConfigType = TypeVar("ConfigType")


@runtime_checkable
class IdentifiableProtocol(Protocol):
    """
    Interface for components with identity.

    This interface defines the contract for components that have a name and
    description. It ensures that components can be identified and described
    in a standard way.

    ## Lifecycle

    1. **Identification**: Access component identity information

    ## Implementation Requirements

    Classes implementing this interface must:
    - Provide a name property to access the component name
    - Provide a description property to access the component description
    """

    @property
    def name(self) -> str:
        """
        Get the component name.

        Returns:
            The name of the component
        """
        ...

    @property
    def description(self) -> str:
        """
        Get the component description.

        Returns:
            The description of the component
        """
        ...


@runtime_checkable
class ConfigurableProtocol(Generic[ConfigType], Protocol):
    """
    Interface for components with configuration.

    This interface defines the contract for components that can be configured.
    It ensures that components can expose and update their configuration in a
    standard way.

    ## Lifecycle

    1. **Configuration Access**: Get component configuration
    2. **Configuration Update**: Update component configuration

    ## Implementation Requirements

    Classes implementing this interface must:
    - Provide a config property to access the component configuration
    - Provide an update_config method to update the component configuration
    """

    @property
    def config(self) -> ConfigType:
        """
        Get the component configuration.

        Returns:
            The configuration of the component
        """
        ...

    def update_config(self, config: ConfigType) -> None:
        """
        Update the component configuration.

        Args:
            config: The new configuration to apply

        Raises:
            ValueError: If the configuration is invalid
        """
        ...


@runtime_checkable
class StatefulProtocol(Protocol):
    """
    Interface for components with state management.

    This interface defines the contract for components that manage state.
    It ensures that components can get, set, and reset their state in a
    standard way.

    ## Lifecycle

    1. **State Access**: Get component state
    2. **State Update**: Set component state
    3. **State Reset**: Reset component state

    ## Implementation Requirements

    Classes implementing this interface must:
    - Provide a get_state method to get the component state
    - Provide a set_state method to set the component state
    - Provide a reset_state method to reset the component state
    """

    def get_state(self) -> Dict[str, Any]:
        """
        Get the component state.

        Returns:
            The current state of the component
        """
        ...

    def set_state(self, state: Dict[str, Any]) -> None:
        """
        Set the component state.

        Args:
            state: The new state to set

        Raises:
            ValueError: If the state is invalid
        """
        ...

    def reset_state(self) -> None:
        """
        Reset the component state to its initial values.

        Raises:
            RuntimeError: If state reset fails
        """
        ...


@runtime_checkable
class LoggableProtocol(Protocol):
    """
    Interface for components with logging capabilities.

    This interface defines the contract for components that support logging.
    It ensures that components can log messages at different levels in a
    standard way.

    ## Lifecycle

    1. **Log Message**: Log a message at a specific level

    ## Implementation Requirements

    Classes implementing this interface must:
    - Provide methods to log messages at different levels
    - Follow standard logging practices
    """

    def log(self, message: str, level: str = "info") -> None:
        """
        Log a message.

        Args:
            message: The message to log
            level: The log level (default: "info")
        """
        ...


@runtime_checkable
class TraceableProtocol(Protocol):
    """
    Interface for components with tracing capabilities.

    This interface defines the contract for components that support tracing.
    It ensures that components can record performance metrics and execution
    statistics in a standard way.

    ## Lifecycle

    1. **Trace Start**: Start tracing an operation
    2. **Trace End**: End tracing an operation
    3. **Trace Data Access**: Get trace data

    ## Implementation Requirements

    Classes implementing this interface must:
    - Provide methods to start and end tracing
    - Provide a method to get trace data
    """

    def start_trace(self, operation: str) -> None:
        """
        Start tracing an operation.

        Args:
            operation: The name of the operation to trace
        """
        ...

    def end_trace(self, operation: str) -> None:
        """
        End tracing an operation.

        Args:
            operation: The name of the operation to trace
        """
        ...

    def get_trace_data(self) -> Dict[str, Any]:
        """
        Get trace data.

        Returns:
            A dictionary of trace data
        """
        ...


@runtime_checkable
class ComponentProtocol(
    IdentifiableProtocol, ConfigurableProtocol[Any], StatefulProtocol, Protocol
):
    """
    Interface for all components.

    This interface defines the contract for all components in the Sifaka framework.
    It combines the requirements of IdentifiableProtocol, ConfigurableProtocol,
    and StatefulProtocol, and adds initialization and cleanup requirements.

    ## Lifecycle

    1. **Initialization**: Set up component resources
    2. **Operation**: Perform component operations
    3. **Cleanup**: Release component resources

    ## Implementation Requirements

    Classes implementing this interface must:
    - Provide an initialize method to set up resources
    - Provide a cleanup method to release resources
    - Implement all requirements from IdentifiableProtocol, ConfigurableProtocol,
      and StatefulProtocol
    """

    def initialize(self) -> None:
        """
        Initialize the component.

        This method should be called after the component is created to set up
        any resources or state needed for operation.

        Raises:
            RuntimeError: If initialization fails
        """
        ...

    def cleanup(self) -> None:
        """
        Clean up the component.

        This method should be called when the component is no longer needed to
        release any resources it holds.

        Raises:
            RuntimeError: If cleanup fails
        """
        ...


@runtime_checkable
class PluginProtocol(IdentifiableProtocol, Protocol):
    """
    Interface for plugins.

    This interface defines the contract for plugins that extend component
    functionality. It ensures that plugins can be identified and loaded in a
    standard way.

    ## Lifecycle

    1. **Identification**: Access plugin identity information
    2. **Loading**: Load plugin functionality

    ## Implementation Requirements

    Classes implementing this interface must:
    - Provide a name property to access the plugin name
    - Provide a description property to access the plugin description
    - Provide a component_type property to identify the component type
    - Provide an initialize method to set up resources
    - Provide a cleanup method to release resources
    """

    @property
    def component_type(self) -> str:
        """
        Get the component type this plugin extends.

        Returns:
            The component type
        """
        ...

    def initialize(self) -> None:
        """
        Initialize the plugin.

        This method should be called after the plugin is created to set up
        any resources or state needed for operation.

        Raises:
            RuntimeError: If initialization fails
        """
        ...

    def cleanup(self) -> None:
        """
        Clean up the plugin.

        This method should be called when the plugin is no longer needed to
        release any resources it holds.

        Raises:
            RuntimeError: If cleanup fails
        """
        ...
