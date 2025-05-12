"""
Component Initialization Module

This module provides standardized initialization patterns for Sifaka components.
It defines base classes and mixins for component initialization, ensuring consistent
behavior across all components.

## Overview
The initialization module provides a standardized approach to component lifecycle
management, including initialization, warm-up, and cleanup phases. It ensures that
all components follow a consistent pattern for resource management, configuration
validation, and dependency resolution.

## Components
- **Initializable Protocol**: Defines the interface for initializable components
- **StandardInitializer**: Provides utility functions for initializing components
- **InitializableMixin**: Mixin class that implements the Initializable protocol
- **BaseInitializable**: Base class for Pydantic-based initializable components

## Usage Examples
```python
from sifaka.core.initialization import InitializableMixin, StandardInitializer

# Create a component with standardized initialization
class MyComponent(InitializableMixin):
    def __init__(self, name, description, config=None):
        super().__init__(name, description, config)

    def _initialize_resources(self):
        # Initialize component-specific resources
        pass

    def _validate_configuration(self):
        # Validate component-specific configuration
        if not self.config.valid_param:
            raise ValueError("Invalid configuration")

# Initialize a component with StandardInitializer
component = MyComponent("my_component", "A sample component")
StandardInitializer.initialize_component(component)
StandardInitializer.warm_up_component(component)

# Use BaseInitializable for Pydantic-based components
from sifaka.core.initialization import BaseInitializable
from pydantic import Field

class MyPydanticComponent(BaseInitializable):
    # Configuration
    max_attempts: int = Field(5, description="Maximum number of attempts")
    timeout: float = Field(10.0, description="Timeout in seconds")

    def _initialize_resources(self):
        # Initialize component-specific resources
        pass
```

## Error Handling
The initialization module provides standardized error handling for component
lifecycle operations. It uses the `InitializationError` and `CleanupError`
exceptions from `sifaka.utils.errors` to ensure consistent error reporting.

## Configuration
Components can be configured through:
- Constructor parameters
- Configuration dictionaries
- Pydantic models (for BaseInitializable)
"""

import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Protocol, Type, TypeVar, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from sifaka.utils.errors.base import InitializationError, CleanupError
from sifaka.utils.logging import get_logger
from sifaka.utils.state import StateManager
from sifaka.utils.resources import ResourceManager, Resource

logger = get_logger(__name__)

T = TypeVar("T")


@runtime_checkable
class Initializable(Protocol):
    """
    Protocol for components that can be initialized.

    This protocol defines the interface for components that follow the standardized
    initialization lifecycle in the Sifaka framework. Components implementing this
    protocol can be managed by the StandardInitializer and follow a consistent
    pattern for initialization, warm-up, and cleanup.

    ## Architecture
    The Initializable protocol defines four key methods that form the component
    lifecycle:
    1. initialize: Initializes the component and its resources
    2. is_initialized: Checks if the component has been initialized
    3. warm_up: Prepares the component for use (e.g., loading models)
    4. cleanup: Releases resources when the component is no longer needed

    ## Lifecycle
    Components implementing this protocol follow a standard lifecycle:
    1. Creation: Component is instantiated
    2. Initialization: Resources are acquired and validated
    3. Warm-up: Component is prepared for use
    4. Operation: Component performs its function
    5. Cleanup: Resources are released

    ## Examples
    ```python
    @runtime_checkable
    class MyInitializable(Initializable):
        def initialize(self) -> None:
            # Initialize resources
            pass

        def is_initialized(self) -> bool:
            # Check initialization status
            return True

        def warm_up(self) -> None:
            # Prepare for use
            pass

        def cleanup(self) -> None:
            # Release resources
            pass
    ```
    """

    def initialize(self) -> None:
        """
        Initialize the component.

        This method initializes the component and its resources, preparing it
        for operation. It should be called before the component is used.

        Raises:
            InitializationError: If initialization fails
        """
        ...

    def is_initialized(self) -> bool:
        """
        Check if the component is initialized.

        This method checks whether the component has been successfully initialized
        and is ready for use.

        Returns:
            True if the component is initialized, False otherwise
        """
        ...

    def warm_up(self) -> None:
        """
        Prepare the component for use.

        This method performs any necessary warm-up operations to prepare the
        component for use, such as loading models or establishing connections.

        Raises:
            InitializationError: If warm-up fails
        """
        ...

    def cleanup(self) -> None:
        """
        Clean up component resources.

        This method releases any resources acquired by the component during
        initialization or use, such as file handles, network connections,
        or memory allocations.

        Raises:
            CleanupError: If cleanup fails
        """
        ...


class StandardInitializer:
    """
    Standard initializer for Sifaka components.

    This class provides utility methods for initializing, warming up, and cleaning up
    components that implement the Initializable protocol. It provides standardized
    error handling and logging for component lifecycle operations.

    ## Architecture
    The StandardInitializer follows a static utility class pattern, providing
    methods that operate on components without requiring an instance of the
    initializer. This allows for consistent initialization behavior across
    the framework without coupling components to a specific initializer instance.

    ## Error Handling
    The StandardInitializer provides standardized error handling for component
    lifecycle operations, catching exceptions and wrapping them in appropriate
    error types (InitializationError, CleanupError) with detailed context.

    ## Examples
    ```python
    from sifaka.core.initialization import StandardInitializer, InitializableMixin

    # Create a component
    component = MyComponent("name", "description")

    # Initialize the component
    StandardInitializer.initialize_component(component)

    # Warm up the component
    StandardInitializer.warm_up_component(component)

    # Clean up the component
    StandardInitializer.cleanup_component(component)

    # Initialize multiple components
    components = [component1, component2, component3]
    StandardInitializer.initialize_components(components)
    ```
    """

    @staticmethod
    def initialize_component(component: Any) -> None:
        """
        Initialize a component with standardized error handling.

        Args:
            component: The component to initialize

        Raises:
            InitializationError: If initialization fails
        """
        try:
            # Check if component is already initialized
            if hasattr(component, "is_initialized") and component.is_initialized():
                logger.debug(f"Component {component.__class__.__name__} already initialized")
                return

            # Initialize the component
            if hasattr(component, "initialize"):
                logger.debug(f"Initializing component {component.__class__.__name__}")
                component.initialize()
                logger.debug(f"Component {component.__class__.__name__} initialized successfully")
            else:
                logger.warning(f"Component {component.__class__.__name__} has no initialize method")

        except Exception as e:
            # Handle initialization error
            logger.error(f"Failed to initialize component {component.__class__.__name__}: {str(e)}")
            raise InitializationError(
                f"Failed to initialize component {component.__class__.__name__}: {str(e)}"
            ) from e

    @staticmethod
    def warm_up_component(component: Any) -> None:
        """
        Warm up a component with standardized error handling.

        Args:
            component: The component to warm up

        Raises:
            InitializationError: If warm-up fails
        """
        try:
            # Warm up the component
            if hasattr(component, "warm_up"):
                logger.debug(f"Warming up component {component.__class__.__name__}")
                component.warm_up()
                logger.debug(f"Component {component.__class__.__name__} warmed up successfully")
            else:
                logger.warning(f"Component {component.__class__.__name__} has no warm_up method")

        except Exception as e:
            # Handle warm-up error
            logger.error(f"Failed to warm up component {component.__class__.__name__}: {str(e)}")
            raise InitializationError(
                f"Failed to warm up component {component.__class__.__name__}: {str(e)}"
            ) from e

    @staticmethod
    def cleanup_component(component: Any) -> None:
        """
        Clean up a component with standardized error handling.

        Args:
            component: The component to clean up
        """
        try:
            # Clean up the component
            if hasattr(component, "cleanup"):
                logger.debug(f"Cleaning up component {component.__class__.__name__}")
                component.cleanup()
                logger.debug(f"Component {component.__class__.__name__} cleaned up successfully")
            else:
                logger.warning(f"Component {component.__class__.__name__} has no cleanup method")

        except Exception as e:
            # Handle cleanup error (log but don't raise)
            logger.error(f"Failed to clean up component {component.__class__.__name__}: {str(e)}")


class InitializableMixin:
    """
    Mixin for components that can be initialized.

    This mixin class provides a standard implementation of the Initializable protocol,
    making it easy to create components that follow the standardized initialization
    lifecycle. It manages component state, resources, and configuration, and provides
    methods for initialization, warm-up, and cleanup.

    ## Architecture
    The InitializableMixin uses the StateManager and ResourceManager from the utils
    module to manage component state and resources. It provides a standard implementation
    of the Initializable protocol methods, with hooks for component-specific behavior.

    ## Lifecycle
    Components using this mixin follow a standard lifecycle:
    1. Creation: Component is instantiated with name, description, and config
    2. Initialization: Resources are acquired and validated
    3. Warm-up: Component is prepared for use
    4. Operation: Component performs its function
    5. Cleanup: Resources are released

    ## Error Handling
    The mixin provides standardized error handling for component lifecycle operations,
    catching exceptions and wrapping them in appropriate error types with detailed context.

    ## Examples
    ```python
    from sifaka.core.initialization import InitializableMixin

    class MyComponent(InitializableMixin):
        def __init__(self, name, description, config=None):
            super().__init__(name, description, config)

        def _validate_configuration(self):
            # Validate component-specific configuration
            if not self.config.get("required_param"):
                raise ValueError("Missing required parameter")

        def _initialize_resources(self):
            # Initialize component-specific resources
            self._resource = SomeResource()
            self._resource_manager.register("my_resource", self._resource)
    ```
    """

    # State management
    _state_manager = None
    _resource_manager = None

    def __init__(
        self, name: str, description: str, config: Optional[Dict[str, Any]] = None, **kwargs: Any
    ) -> None:
        """
        Initialize the component.

        Args:
            name: The component name
            description: The component description
            config: Optional component configuration
            **kwargs: Additional component parameters
        """
        # Store name and description
        self._name = name
        self._description = description

        # Initialize state and resources
        self._initialize_state()
        self._initialize_resource_manager()

        # Store configuration
        self._config = config or {}

        # Store creation time
        self._creation_time = time.time()

        # Set initialization flag
        self._state_manager.update("initialized", False)
        self._state_manager.update("dependencies_validated", False)
        self._state_manager.update("resources_initialized", False)

    def _initialize_state(self) -> None:
        """Initialize component state."""
        # Create state manager
        self._state_manager = StateManager()

        # Initialize state
        self._state_manager.update("initialized", False)
        self._state_manager.update("cache", {})
        self._state_manager.set_metadata("component_type", self.__class__.__name__)
        self._state_manager.set_metadata("creation_time", time.time())

    def _initialize_resource_manager(self) -> None:
        """Initialize resource manager."""
        # Create resource manager
        self._resource_manager = ResourceManager()

    def initialize(self) -> None:
        """
        Initialize the component.

        This method initializes the component, preparing it for use.
        It validates the configuration, validates dependencies,
        initializes resources, and sets the initialization flag.

        Raises:
            InitializationError: If initialization fails
        """
        try:
            # Check if already initialized
            if self.is_initialized():
                logger.debug(f"Component {self.__class__.__name__} already initialized")
                return

            # Validate configuration
            self._validate_configuration()

            # Validate dependencies
            self._validate_dependencies()
            self._state_manager.update("dependencies_validated", True)

            # Register resources
            self._register_resources()

            # Initialize resources
            self._initialize_resources()
            self._state_manager.update("resources_initialized", True)

            # Set initialization flag
            self._state_manager.update("initialized", True)
            self._state_manager.set_metadata("initialization_time", time.time())

            logger.debug(f"Component {self.__class__.__name__} initialized successfully")

        except Exception as e:
            # Handle initialization error
            logger.error(f"Failed to initialize component {self.__class__.__name__}: {str(e)}")
            raise InitializationError(
                f"Failed to initialize component {self.__class__.__name__}: {str(e)}"
            ) from e

    def is_initialized(self) -> bool:
        """
        Check if the component is initialized.

        Returns:
            True if the component is initialized, False otherwise
        """
        return self._state_manager.get("initialized", False)

    def warm_up(self) -> None:
        """
        Prepare the component for use.

        This method prepares the component for use, performing any
        necessary warm-up operations.

        Raises:
            InitializationError: If warm-up fails
        """
        try:
            # Check if initialized
            if not self.is_initialized():
                logger.warning(
                    f"Component {self.__class__.__name__} not initialized, initializing now"
                )
                self.initialize()

            # Perform warm-up operations
            self._warm_up_resources()

            logger.debug(f"Component {self.__class__.__name__} warmed up successfully")

        except Exception as e:
            # Handle warm-up error
            logger.error(f"Failed to warm up component {self.__class__.__name__}: {str(e)}")
            raise InitializationError(
                f"Failed to warm up component {self.__class__.__name__}: {str(e)}"
            ) from e

    def cleanup(self) -> None:
        """
        Clean up component resources.

        This method cleans up component resources, releasing any
        resources that were acquired during initialization or use.
        """
        try:
            # Clean up resources using resource manager
            if self._resource_manager:
                self._resource_manager.cleanup_all()

            # Perform additional cleanup operations
            self._cleanup_resources()

            # Clear cache
            if hasattr(self, "clear_cache") and callable(getattr(self, "clear_cache")):
                self.clear_cache()
            else:
                self._state_manager.update("cache", {})

            # Reset initialization flags
            self._state_manager.update("initialized", False)
            self._state_manager.update("resources_initialized", False)
            self._state_manager.update("dependencies_validated", False)
            self._state_manager.set_metadata("cleanup_time", time.time())

            logger.debug(f"Component {self.__class__.__name__} cleaned up successfully")

        except Exception as e:
            # Handle cleanup error (log but don't raise)
            logger.error(f"Failed to clean up component {self.__class__.__name__}: {str(e)}")

    def _validate_configuration(self) -> None:
        """
        Validate component configuration.

        This method validates the component configuration, ensuring
        that all required parameters are present and valid.

        Raises:
            ValueError: If the configuration is invalid
        """
        # Base implementation does nothing
        pass

    def _validate_dependencies(self) -> None:
        """
        Validate component dependencies.

        This method validates that all required dependencies are present
        and compatible with the component.

        Raises:
            ValueError: If a required dependency is missing or incompatible
        """
        # Base implementation does nothing
        pass

    def _register_resources(self) -> None:
        """
        Register component resources.

        This method registers resources with the resource manager.
        Subclasses should override this method to register specific resources.

        Raises:
            ValueError: If resource registration fails
        """
        # Base implementation does nothing
        pass

    def _initialize_resources(self) -> None:
        """
        Initialize component resources.

        This method initializes component resources, acquiring any
        resources that are needed for component operation.

        By default, it initializes all resources registered with the resource manager.

        Raises:
            InitializationError: If resource initialization fails
        """
        # Initialize all resources using resource manager
        if self._resource_manager:
            try:
                self._resource_manager.initialize_all()
            except Exception as e:
                logger.error(f"Failed to initialize resources: {str(e)}")
                raise InitializationError(f"Failed to initialize resources: {str(e)}") from e

    def _warm_up_resources(self) -> None:
        """
        Warm up component resources.

        This method warms up component resources, preparing them
        for use.

        Raises:
            InitializationError: If resource warm-up fails
        """
        # Base implementation does nothing
        pass

    def _cleanup_resources(self) -> None:
        """
        Clean up component resources.

        This method cleans up component resources, releasing any
        resources that were acquired during initialization or use.

        Note: The base implementation already calls resource_manager.cleanup_all(),
        so subclasses only need to override this method if they have additional
        cleanup operations to perform.
        """
        # Base implementation does nothing (resource manager cleanup is handled in cleanup method)
        pass


class BaseInitializable(BaseModel, InitializableMixin):
    """
    Base class for initializable components using Pydantic.

    This class combines Pydantic's BaseModel with the InitializableMixin to provide
    a base class for components that need both data validation and standardized
    initialization. It's ideal for components that have configuration parameters
    that need validation and type checking.

    ## Architecture
    BaseInitializable combines the data validation capabilities of Pydantic with
    the lifecycle management of InitializableMixin. It uses Pydantic's model fields
    for configuration parameters and InitializableMixin's methods for initialization,
    warm-up, and cleanup.

    ## Lifecycle
    Components using this base class follow the standard lifecycle:
    1. Creation: Component is instantiated with validated parameters
    2. Initialization: Resources are acquired and validated
    3. Warm-up: Component is prepared for use
    4. Operation: Component performs its function
    5. Cleanup: Resources are released

    ## Examples
    ```python
    from sifaka.core.initialization import BaseInitializable
    from pydantic import Field

    class MyComponent(BaseInitializable):
        # Configuration parameters with validation
        max_attempts: int = Field(5, description="Maximum number of attempts")
        timeout: float = Field(10.0, description="Timeout in seconds")

        def _initialize_resources(self):
            # Initialize component-specific resources
            self._resource = SomeResource(self.timeout)
            self._resource_manager.register("my_resource", self._resource)

    # Create component with validated parameters
    component = MyComponent(name="my_component", description="A sample component", max_attempts=3)
    component.initialize()
    ```
    """

    # Component identification
    name: str = Field(description="Component name", min_length=1)
    description: str = Field(description="Component description", min_length=1)

    # Pydantic configuration
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # State management
    _state_manager = PrivateAttr(default_factory=StateManager)
    _resource_manager = PrivateAttr(default_factory=ResourceManager)

    def __init__(self, **data: Any) -> None:
        """
        Initialize the component.

        Args:
            **data: Component parameters
        """
        # Initialize BaseModel
        super().__init__(**data)

        # Initialize state and resources
        self._initialize_state()
        self._initialize_resource_manager()

    @classmethod
    def create(cls: Type[T], name: str, description: str, **kwargs: Any) -> T:
        """
        Create a new component instance.

        Args:
            name: The component name
            description: The component description
            **kwargs: Additional component parameters

        Returns:
            A new component instance
        """
        # Create component
        component = cls(name=name, description=description, **kwargs)

        # Initialize component
        component.initialize()

        return component

    @classmethod
    def create_with_dependencies(
        cls: Type[T], name: str, description: str, dependencies: Dict[str, Any], **kwargs: Any
    ) -> T:
        """
        Create a new component instance with dependencies.

        Args:
            name: The component name
            description: The component description
            dependencies: Dictionary of dependencies to inject
            **kwargs: Additional component parameters

        Returns:
            A new component instance with dependencies
        """
        # Create component
        component = cls(name=name, description=description, **kwargs)

        # Set dependencies
        for key, value in dependencies.items():
            setattr(component, f"_{key}", value)

        # Initialize component
        component.initialize()

        return component
