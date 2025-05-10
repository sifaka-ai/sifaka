"""
Component Initialization Module

This module provides standardized initialization patterns for Sifaka components.
It defines base classes and mixins for component initialization, ensuring consistent
behavior across all components.

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
```
"""

import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Protocol, Type, TypeVar, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from sifaka.utils.errors import InitializationError
from sifaka.utils.logging import get_logger
from sifaka.utils.state import StateManager

logger = get_logger(__name__)

T = TypeVar("T")


@runtime_checkable
class Initializable(Protocol):
    """Protocol for components that can be initialized."""

    def initialize(self) -> None:
        """Initialize the component."""
        ...

    def is_initialized(self) -> bool:
        """Check if the component is initialized."""
        ...

    def warm_up(self) -> None:
        """Prepare the component for use."""
        ...

    def cleanup(self) -> None:
        """Clean up component resources."""
        ...


class StandardInitializer:
    """Standard initializer for Sifaka components."""

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
    """Mixin for components that can be initialized."""

    # State management
    _state_manager = None

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

        # Initialize state
        self._initialize_state()

        # Store configuration
        self._config = config or {}

        # Store creation time
        self._creation_time = time.time()

        # Set initialization flag
        self._state_manager.update("initialized", False)

    def _initialize_state(self) -> None:
        """Initialize component state."""
        # Create state manager
        self._state_manager = StateManager()

        # Initialize state
        self._state_manager.update("initialized", False)
        self._state_manager.update("cache", {})
        self._state_manager.set_metadata("component_type", self.__class__.__name__)
        self._state_manager.set_metadata("creation_time", time.time())

    def initialize(self) -> None:
        """
        Initialize the component.

        This method initializes the component, preparing it for use.
        It validates the configuration, initializes resources, and
        sets the initialization flag.

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

            # Initialize resources
            self._initialize_resources()

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
            # Perform cleanup operations
            self._cleanup_resources()

            # Clear cache
            if hasattr(self, "clear_cache") and callable(getattr(self, "clear_cache")):
                self.clear_cache()

            # Reset initialization flag
            self._state_manager.update("initialized", False)

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

    def _initialize_resources(self) -> None:
        """
        Initialize component resources.

        This method initializes component resources, acquiring any
        resources that are needed for component operation.

        Raises:
            InitializationError: If resource initialization fails
        """
        # Base implementation does nothing
        pass

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
        """
        # Base implementation does nothing
        pass


class BaseInitializable(BaseModel, InitializableMixin):
    """Base class for initializable components."""

    # Component identification
    name: str = Field(description="Component name", min_length=1)
    description: str = Field(description="Component description", min_length=1)

    # Pydantic configuration
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # State management
    _state_manager = PrivateAttr(default_factory=StateManager)

    def __init__(self, **data: Any) -> None:
        """
        Initialize the component.

        Args:
            **data: Component parameters
        """
        # Initialize BaseModel
        super().__init__(**data)

        # Initialize state
        self._initialize_state()

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
