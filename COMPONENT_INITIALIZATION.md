# Component Initialization Improvement Plan

This document outlines the plan for improving component initialization in the Sifaka codebase.

## Current State

The Sifaka codebase currently has an `InitializableMixin` class in `sifaka/core/initialization.py` that provides:

1. A standardized way to initialize components
2. Methods for validating configuration
3. Methods for initializing resources
4. Methods for cleaning up resources

However, the current implementation has some limitations:

1. Inconsistent usage across the codebase
2. Limited validation for required dependencies
3. Inconsistent resource management
4. Inconsistent state management

## Improvement Goals

1. Standardize component initialization with InitializableMixin
2. Implement proper resource management
3. Add validation for required dependencies
4. Use state management consistently

## Implementation Plan

### 1. Standardize Component Initialization with InitializableMixin

#### 1.1 Update InitializableMixin

```python
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
        self._state_manager.update("resources_initialized", False)

    def _initialize_state(self) -> None:
        """
        Initialize the component state.

        This method initializes the component state, creating a state manager
        if one doesn't exist.
        """
        if self._state_manager is None:
            from sifaka.utils.state import StateManager
            self._state_manager = StateManager()

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

            # Validate dependencies
            self._validate_dependencies()

            # Initialize resources
            self._initialize_resources()

            # Set initialization flag
            self._state_manager.update("initialized", True)
            self._state_manager.set_metadata("initialization_time", time.time())

            logger.debug(f"Component {self.__class__.__name__} initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing component {self.__class__.__name__}: {str(e)}")
            raise InitializationError(f"Error initializing component {self.__class__.__name__}: {str(e)}") from e

    def _validate_configuration(self) -> None:
        """
        Validate the component configuration.

        This method validates the component configuration, ensuring that
        all required configuration parameters are present and valid.

        Raises:
            ValueError: If the configuration is invalid
        """
        # Default implementation does nothing
        pass

    def _validate_dependencies(self) -> None:
        """
        Validate the component dependencies.

        This method validates the component dependencies, ensuring that
        all required dependencies are present and valid.

        Raises:
            DependencyError: If a required dependency is missing or invalid
        """
        # Default implementation does nothing
        pass

    def _initialize_resources(self) -> None:
        """
        Initialize the component resources.

        This method initializes the component resources, preparing them for use.
        It should be overridden by subclasses to initialize specific resources.

        Raises:
            InitializationError: If resource initialization fails
        """
        # Default implementation does nothing
        self._state_manager.update("resources_initialized", True)

    def _release_resources(self) -> None:
        """
        Release the component resources.

        This method releases the component resources, cleaning up after use.
        It should be overridden by subclasses to release specific resources.

        Raises:
            CleanupError: If resource cleanup fails
        """
        # Default implementation does nothing
        self._state_manager.update("resources_initialized", False)

    def is_initialized(self) -> bool:
        """
        Check if the component is initialized.

        Returns:
            True if the component is initialized, False otherwise
        """
        return self._state_manager.get("initialized", False)

    def cleanup(self) -> None:
        """
        Clean up the component.

        This method cleans up the component, releasing any resources
        and preparing it for disposal.

        Raises:
            CleanupError: If cleanup fails
        """
        try:
            # Check if initialized
            if not self.is_initialized():
                logger.debug(f"Component {self.__class__.__name__} not initialized, nothing to clean up")
                return

            # Release resources
            self._release_resources()

            # Set initialization flag
            self._state_manager.update("initialized", False)
            self._state_manager.set_metadata("cleanup_time", time.time())

            logger.debug(f"Component {self.__class__.__name__} cleaned up successfully")
        except Exception as e:
            logger.error(f"Error cleaning up component {self.__class__.__name__}: {str(e)}")
            raise CleanupError(f"Error cleaning up component {self.__class__.__name__}: {str(e)}") from e

    def warm_up(self) -> None:
        """
        Warm up the component.

        This method warms up the component, preparing it for use.
        It should be overridden by subclasses to implement specific warm-up logic.

        Raises:
            InitializationError: If warm-up fails
        """
        # Default implementation does nothing
        pass

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
```

### 2. Implement Proper Resource Management

#### 2.1 Update Component Classes

```python
class ModelProvider(InitializableMixin):
    """Base model provider implementation."""

    def __init__(
        self,
        name: str,
        description: str,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ):
        """
        Initialize the model provider.

        Args:
            name: The provider name
            description: The provider description
            api_key: Optional API key
            model_name: Optional model name
            config: Optional provider configuration
            **kwargs: Additional provider parameters
        """
        # Initialize base component
        super().__init__(name=name, description=description, config=config)

        # Store parameters in state
        self._state_manager.update("api_key", api_key)
        self._state_manager.update("model_name", model_name)
        self._state_manager.update("client", None)

    def _initialize_resources(self) -> None:
        """
        Initialize the provider resources.

        This method initializes the provider resources, preparing them for use.
        It creates a client for the provider.

        Raises:
            InitializationError: If resource initialization fails
        """
        try:
            # Create client
            self._state_manager.update("client", self._create_client())
            self._state_manager.update("resources_initialized", True)
        except Exception as e:
            logger.error(f"Error initializing resources for {self.__class__.__name__}: {str(e)}")
            raise InitializationError(f"Error initializing resources for {self.__class__.__name__}: {str(e)}") from e

    def _release_resources(self) -> None:
        """
        Release the provider resources.

        This method releases the provider resources, cleaning up after use.
        It closes the client for the provider.

        Raises:
            CleanupError: If resource cleanup fails
        """
        try:
            # Close client
            client = self._state_manager.get("client")
            if client is not None and hasattr(client, "close"):
                client.close()
            self._state_manager.update("client", None)
            self._state_manager.update("resources_initialized", False)
        except Exception as e:
            logger.error(f"Error releasing resources for {self.__class__.__name__}: {str(e)}")
            raise CleanupError(f"Error releasing resources for {self.__class__.__name__}: {str(e)}") from e

    def _create_client(self) -> Any:
        """
        Create a client for the provider.

        This method creates a client for the provider, which is used to
        communicate with the provider's API.

        Returns:
            A client for the provider

        Raises:
            InitializationError: If client creation fails
        """
        # Default implementation returns None
        return None
```

### 3. Add Validation for Required Dependencies

#### 3.1 Update Component Classes

```python
class Critic(InitializableMixin):
    """Base critic implementation."""

    def __init__(
        self,
        name: str,
        description: str,
        llm_provider: Any,
        prompt_factory: Optional[Any] = None,
        config: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ):
        """
        Initialize the critic.

        Args:
            name: The critic name
            description: The critic description
            llm_provider: The language model provider
            prompt_factory: Optional prompt factory
            config: Optional critic configuration
            **kwargs: Additional critic parameters
        """
        # Initialize base component
        super().__init__(name=name, description=description, config=config)

        # Store dependencies in state
        self._state_manager.update("model", llm_provider)
        self._state_manager.update("prompt_factory", prompt_factory or self._create_default_prompt_factory())

    def _validate_dependencies(self) -> None:
        """
        Validate the critic dependencies.

        This method validates the critic dependencies, ensuring that
        all required dependencies are present and valid.

        Raises:
            DependencyError: If a required dependency is missing or invalid
        """
        # Validate model
        model = self._state_manager.get("model")
        if model is None:
            raise DependencyError("Model provider is required")

        # Validate prompt factory
        prompt_factory = self._state_manager.get("prompt_factory")
        if prompt_factory is None:
            raise DependencyError("Prompt factory is required")
```

### 4. Use State Management Consistently

#### 4.1 Update Component Classes

```python
class Rule(InitializableMixin):
    """Base rule implementation."""

    def __init__(
        self,
        name: str,
        description: str,
        validator: Any,
        config: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ):
        """
        Initialize the rule.

        Args:
            name: The rule name
            description: The rule description
            validator: The validator to use
            config: Optional rule configuration
            **kwargs: Additional rule parameters
        """
        # Initialize base component
        super().__init__(name=name, description=description, config=config)

        # Store dependencies in state
        self._state_manager.update("validator", validator)

    def validate(self, text: str) -> Any:
        """
        Validate text against the rule.

        Args:
            text: The text to validate

        Returns:
            The validation result
        """
        # Get validator
        validator = self._state_manager.get("validator")
        if validator is None:
            raise DependencyError("Validator is required")

        # Validate text
        return validator.validate(text)
```

## Success Criteria

1. Standardized component initialization with InitializableMixin
2. Proper resource management in all components
3. Validation for required dependencies
4. Consistent state management
5. Comprehensive documentation for component initialization
6. All components use InitializableMixin
7. Tests validate proper component initialization and cleanup
