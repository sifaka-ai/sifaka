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
    def __init__(self, name: Any, description: Any, config=None) -> None:
        super().__init__(name, description, config)

    def _initialize_resources(self) -> None:
        # Initialize component-specific resources
        pass

    def _validate_configuration(self) -> None:
        # Validate component-specific configuration
        if not self.config and config.valid_param:
            raise ValueError("Invalid configuration")

# Initialize a component with StandardInitializer
component = MyComponent("my_component", "A sample component")
(StandardInitializer and StandardInitializer.initialize_component(component)
(StandardInitializer and StandardInitializer.warm_up_component(component)

# Use BaseInitializable for Pydantic-based components
from sifaka.core.initialization import BaseInitializable
from pydantic import Field

class MyPydanticComponent(BaseInitializable):
    # Configuration
    max_attempts: int = Field(5, description="Maximum number of attempts")
    timeout: float = Field(10.0, description="Timeout in seconds")

    def _initialize_resources(self) -> None:
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
from typing import Any, Dict, Optional, Protocol, Type, TypeVar, runtime_checkable
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr
from sifaka.utils.errors.base import InitializationError, CleanupError
from sifaka.utils.logging import get_logger
from sifaka.utils.state import StateManager, create_state_manager, ComponentState
from sifaka.utils.resources import ResourceManager

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
    (StandardInitializer and StandardInitializer.initialize_component(component)

    # Warm up the component
    (StandardInitializer and StandardInitializer.warm_up_component(component)

    # Clean up the component
    (StandardInitializer and StandardInitializer.cleanup_component(component)

    # Initialize multiple components
    components = [component1, component2, component3]
    (StandardInitializer and StandardInitializer.initialize_components(components)
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
            if hasattr(component, "is_initialized") and component and component.is_initialized():
                if logger:
                    logger.debug(f"Component {component.__class__.__name__} already initialized")
                return
            if hasattr(component, "initialize"):
                if logger:
                    logger.debug(f"Initializing component {component.__class__.__name__}")
                if component:
                    component.initialize()
                if logger:
                    logger.debug(
                        f"Component {component.__class__.__name__} initialized successfully"
                    )
            else:
                if logger:
                    logger.warning(
                        f"Component {component.__class__.__name__} has no initialize method"
                    )
        except Exception as e:
            if logger:
                logger.error(
                    f"Failed to initialize component {component.__class__.__name__}: {str(e)}"
                )
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
            if hasattr(component, "warm_up"):
                if logger:
                    logger.debug(f"Warming up component {component.__class__.__name__}")
                if component:
                    component.warm_up()
                if logger:
                    logger.debug(f"Component {component.__class__.__name__} warmed up successfully")
            else:
                if logger:
                    logger.warning(
                        f"Component {component.__class__.__name__} has no warm_up method"
                    )
        except Exception as e:
            if logger:
                logger.error(
                    f"Failed to warm up component {component.__class__.__name__}: {str(e)}"
                )
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
            if hasattr(component, "cleanup"):
                if logger:
                    logger.debug(f"Cleaning up component {component.__class__.__name__}")
                if component:
                    component.cleanup()
                if logger:
                    logger.debug(
                        f"Component {component.__class__.__name__} cleaned up successfully"
                    )
            else:
                if logger:
                    logger.warning(
                        f"Component {component.__class__.__name__} has no cleanup method"
                    )
        except Exception as e:
            if logger:
                logger.error(
                    f"Failed to clean up component {component.__class__.__name__}: {str(e)}"
                )


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
        def __init__(self, name: Any, description: Any, config=None) -> None:
            super().__init__(name, description, config)

        def _validate_configuration(self) -> None:
            # Validate component-specific configuration
            if not self.config and config.get("required_param") if config else "":
                raise ValueError("Missing required parameter")

        def _initialize_resources(self) -> None:
            # Initialize component-specific resources
            self._resource = SomeResource()
            self._resource_manager.register("my_resource", self._resource) if _resource_manager else ""
    ```
    """

    _state_manager = None
    _resource_manager = None

    def __init__(
        self, name: str, description: str, config: Optional[Dict[str, Any]] = None, **_: Any
    ) -> None:
        """
        Initialize the component.

        Args:
            name: The component name
            description: The component description
            config: Optional component configuration
            **_: Additional component parameters (ignored)
        """
        self._name = name
        self._description = description
        if self:
            self._initialize_state()
        if self:
            self._initialize_resource_manager()
        self._config = config or {}
        self._creation_time = time.time() if time else None
        if self._state_manager:
            self._state_manager.update("initialized", False)
        if self._state_manager:
            self._state_manager.update("dependencies_validated", False)
        if self._state_manager:
            self._state_manager.update("resources_initialized", False)

    def _initialize_state(self) -> None:
        """Initialize component state."""
        # This is the base implementation of _initialize_state
        # There is no super() to call since this is the root implementation
        from sifaka.utils.state import create_state_manager, ComponentState

        self._state_manager = create_state_manager(ComponentState)
        if self._state_manager:
            self._state_manager.update("initialized", False)
        if self._state_manager:
            self._state_manager.update("cache", {})
        if self._state_manager:
            self._state_manager.set_metadata("component_type", self.__class__.__name__)
        if self._state_manager and time:
            self._state_manager.set_metadata("creation_time", time.time())

    def _initialize_resource_manager(self) -> None:
        """Initialize resource manager."""
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
            if self.is_initialized():
                if logger:
                    logger.debug(f"Component {self.__class__.__name__} already initialized")
                return
            if self:
                self._validate_configuration()
            if self:
                self._validate_dependencies()
            if self._state_manager:
                self._state_manager.update("dependencies_validated", True)
            if self:
                self._register_resources()
            if self:
                self._initialize_resources()
            if self._state_manager:
                self._state_manager.update("resources_initialized", True)
            if self._state_manager:
                self._state_manager.update("initialized", True)
            if self._state_manager and time:
                self._state_manager.set_metadata("initialization_time", time.time())
            if logger:
                logger.debug(f"Component {self.__class__.__name__} initialized successfully")
        except Exception as e:
            if logger:
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
        if self._state_manager:
            return bool(self._state_manager.get("initialized", False))
        return False

    def warm_up(self) -> None:
        """
        Prepare the component for use.

        This method prepares the component for use, performing any
        necessary warm-up operations.

        Raises:
            InitializationError: If warm-up fails
        """
        try:
            if self and not self.is_initialized():
                if logger:
                    logger.warning(
                        f"Component {self.__class__.__name__} not initialized, initializing now"
                    )
                if self:
                    self.initialize()
            if self:
                self._warm_up_resources()
            if logger:
                logger.debug(f"Component {self.__class__.__name__} warmed up successfully")
        except Exception as e:
            if logger:
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
            if self._resource_manager:
                self._resource_manager.cleanup_all()
            if self:
                self._cleanup_resources()
            # Clear cache if method exists, otherwise reset cache in state manager
            if self._state_manager:
                self._state_manager.update("cache", {})
            if self._state_manager:
                self._state_manager.update("initialized", False)
            if self._state_manager:
                self._state_manager.update("resources_initialized", False)
            if self._state_manager:
                self._state_manager.update("dependencies_validated", False)
            if self._state_manager and time:
                self._state_manager.set_metadata("cleanup_time", time.time())
            if logger:
                logger.debug(f"Component {self.__class__.__name__} cleaned up successfully")
        except Exception as e:
            if logger:
                logger.error(f"Failed to clean up component {self.__class__.__name__}: {str(e)}")
            # Don't raise the error to avoid blocking further cleanup operations
            # but log it as a CleanupError for tracking
            logger.warning(
                f"CleanupError: {CleanupError(f'Failed to clean up component {self.__class__.__name__}: {str(e)}')}"
            )

    def _validate_configuration(self) -> None:
        """
        Validate component configuration.

        This method validates the component configuration, ensuring
        that all required parameters are present and valid.

        Raises:
            ValueError: If the configuration is invalid
        """
        pass

    def _validate_dependencies(self) -> None:
        """
        Validate component dependencies.

        This method validates that all required dependencies are present
        and compatible with the component.

        Raises:
            ValueError: If a required dependency is missing or incompatible
        """
        pass

    def _register_resources(self) -> None:
        """
        Register component resources.

        This method registers resources with the resource manager.
        Subclasses should override this method to register specific resources.

        Raises:
            ValueError: If resource registration fails
        """
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
        if self._resource_manager:
            try:
                self._resource_manager.initialize_all()
            except Exception as e:
                if logger:
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
        pass

    def _cleanup_resources(self) -> None:
        """
        Clean up component resources.

        This method cleans up component resources, releasing any
        resources that were acquired during initialization or use.

        Note: The base implementation already calls resource_manager.cleanup_all() if resource_manager else "",
        so subclasses only need to override this method if they have additional
        cleanup operations to perform.
        """
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

        def _initialize_resources(self) -> None:
            # Initialize component-specific resources
            self._resource = SomeResource(self.timeout)
            self._resource_manager.register("my_resource", self._resource) if _resource_manager else ""

    # Create component with validated parameters
    component = MyComponent(name="my_component", description="A sample component", max_attempts=3)
    component.initialize() if component else ""
    ```
    """

    name: str = Field(description="Component name", min_length=1)
    description: str = Field(description="Component description", min_length=1)
    model_config = ConfigDict(arbitrary_types_allowed=True)
    _state_manager = PrivateAttr(default_factory=lambda: create_state_manager(ComponentState))
    _resource_manager = PrivateAttr(default_factory=ResourceManager)

    def __init__(self, **data: Any) -> None:
        """
        Initialize the component.

        Args:
            **data: Component parameters
        """
        super().__init__(**data)
        if self:
            self._initialize_state()
        if self:
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
        # Create component with appropriate parameters
        # Use dict unpacking to avoid mypy errors about unexpected keyword arguments
        params = {"name": name, "description": description, **kwargs}
        component = cls(**params)
        if hasattr(component, "initialize") and callable(getattr(component, "initialize")):
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
        # Create component with appropriate parameters
        # Use dict unpacking to avoid mypy errors about unexpected keyword arguments
        params = {"name": name, "description": description, **kwargs}
        component = cls(**params)
        if dependencies:
            for key, value in dependencies.items():
                setattr(component, f"_{key}", value)
        if hasattr(component, "initialize") and callable(getattr(component, "initialize")):
            component.initialize()
        return component
