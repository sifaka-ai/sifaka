"""
Component registry for Sifaka.

This module provides a central registry for component registration and retrieval.
It uses a singleton pattern to ensure a single registry instance throughout the application
and implements lazy loading to prevent circular dependencies.

Example:
    ```python
    from sifaka.registry import register_model, get_model_factory

    # Register a model factory
    @register_model("custom")
    def create_custom_model(model_name, **options):
        return CustomModel(model_name, **options)

    # Get a model factory
    factory = get_model_factory("custom")
    model = factory("my-model", temperature=0.7)
    ```
"""

import importlib
import logging
from typing import Callable, Dict, List, Optional, Set, TypeVar

from sifaka.interfaces import ImproverFactory, ModelFactory, ValidatorFactory

# Type variable for generic functions
T = TypeVar("T")

# Logger
logger = logging.getLogger(__name__)


class Registry:
    """Central registry for component registration and retrieval.

    This class uses a singleton pattern to ensure a single registry instance
    throughout the application. It provides methods for registering and
    retrieving component factories.

    The registry uses lazy loading to prevent circular imports. Components are
    only imported when they are needed, not when the registry is initialized.

    Attributes:
        _model_factories: Dictionary of model factories, keyed by provider name.
        _validator_factories: Dictionary of validator factories, keyed by name.
        _improver_factories: Dictionary of improver factories, keyed by name.
        _initialized_types: Set of component types that have been initialized.
        _lazy_imports: Dictionary of modules to import for each component type.
    """

    # Singleton instance
    _instance = None

    def __new__(cls) -> "Registry":
        """Create a new Registry instance or return the existing one."""
        if cls._instance is None:
            cls._instance = super(Registry, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        """Initialize the registry."""
        if getattr(self, "_initialized", False):
            return

        self._model_factories: Dict[str, ModelFactory] = {}
        self._validator_factories: Dict[str, ValidatorFactory] = {}
        self._improver_factories: Dict[str, ImproverFactory] = {}

        # Set of component types that have been initialized
        self._initialized_types: Set[str] = set()

        # Lazy imports to avoid circular dependencies
        self._lazy_imports: Dict[str, List[str]] = {
            "model": [
                "sifaka.models.openai",
                "sifaka.models.anthropic",
                "sifaka.models.gemini",
            ],
            "validator": [
                "sifaka.validators.length",
                "sifaka.validators.content",
                "sifaka.validators.format",
                "sifaka.validators.classifier",
                "sifaka.validators.guardrails",
            ],
            "improver": [
                "sifaka.critics.constitutional",
                "sifaka.critics.reflexion",
                "sifaka.critics.prompt",
                "sifaka.critics.n_critics",
                "sifaka.critics.self_rag",
                "sifaka.critics.self_refine",
                "sifaka.critics.retrieval_enhanced",
                "sifaka.critics.retrieval_factory",
            ],
            "retriever": [
                "sifaka.retrievers.base",
                "sifaka.retrievers.elasticsearch_retriever",
                "sifaka.retrievers.milvus_retriever",
                "sifaka.retrievers.augmenter",
            ],
        }

        self._initialized = True
        logger.debug("Registry initialized")

    def register_model(self, provider: str, factory: ModelFactory) -> None:
        """Register a model factory.

        Args:
            provider: The provider name (e.g., "openai", "anthropic").
            factory: The factory function that creates models for this provider.

        Raises:
            ValueError: If the provider is already registered.
        """
        provider = provider.lower()
        if provider in self._model_factories:
            logger.warning(f"Model provider '{provider}' already registered, overwriting")

        self._model_factories[provider] = factory
        logger.debug(f"Registered model factory for provider '{provider}'")

    def _initialize_component_type(self, component_type: str) -> None:
        """Initialize a component type by importing its modules.

        Args:
            component_type: The type of component (e.g., "model", "validator").

        Raises:
            ImportError: If a critical module cannot be imported.
        """
        import time

        start_time = time.time()

        # Skip if already initialized
        if component_type in self._initialized_types:
            logger.debug(f"Component type '{component_type}' already initialized, skipping")
            return

        logger.debug(f"Initializing component type '{component_type}'")

        # Import modules for this component type
        if component_type in self._lazy_imports:
            import_errors = []
            modules_imported = 0

            for module_name in self._lazy_imports[component_type]:
                try:
                    importlib.import_module(module_name)
                    modules_imported += 1
                    logger.debug(f"Successfully imported {module_name}")
                except ImportError as e:
                    import_errors.append((module_name, str(e)))
                    logger.debug(f"Could not import {module_name}: {str(e)}")
                except Exception as e:
                    import_errors.append((module_name, f"{type(e).__name__}: {str(e)}"))
                    logger.warning(f"Error importing {module_name}: {type(e).__name__}: {str(e)}")

            # Log summary of imports
            if import_errors:
                error_modules = [e[0] for e in import_errors]
                logger.warning(
                    f"Failed to import {len(import_errors)} modules for component type '{component_type}': {error_modules}"
                )

            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds

            logger.debug(
                f"Imported {modules_imported}/{len(self._lazy_imports[component_type])} modules "
                f"for component type '{component_type}' in {processing_time:.2f}ms"
            )
        else:
            logger.debug(f"No modules to import for component type '{component_type}'")

        # Mark as initialized
        self._initialized_types.add(component_type)
        logger.debug(f"Initialized component type '{component_type}'")

    def get_model_factory(self, provider: str) -> Optional[ModelFactory]:
        """Get a model factory.

        Args:
            provider: The provider name (e.g., "openai", "anthropic").

        Returns:
            The factory function, or None if not found.
        """
        # Initialize model components if needed
        self._initialize_component_type("model")

        provider = provider.lower()
        factory = self._model_factories.get(provider)
        if factory is None:
            logger.debug(f"Model factory for provider '{provider}' not found")

        return factory

    def register_validator(self, name: str, factory: ValidatorFactory) -> None:
        """Register a validator factory.

        Args:
            name: The name of the validator.
            factory: The factory function that creates validators.

        Raises:
            ValueError: If the validator is already registered.
        """
        name = name.lower()
        if name in self._validator_factories:
            logger.warning(f"Validator '{name}' already registered, overwriting")

        self._validator_factories[name] = factory
        logger.debug(f"Registered validator factory for '{name}'")

    def get_validator_factory(self, name: str) -> Optional[ValidatorFactory]:
        """Get a validator factory.

        Args:
            name: The name of the validator.

        Returns:
            The factory function, or None if not found.
        """
        # Initialize validator components if needed
        self._initialize_component_type("validator")

        name = name.lower()
        factory = self._validator_factories.get(name)
        if factory is None:
            logger.debug(f"Validator factory for '{name}' not found")

        return factory

    def register_improver(self, name: str, factory: ImproverFactory) -> None:
        """Register an improver factory.

        Args:
            name: The name of the improver.
            factory: The factory function that creates improvers.

        Raises:
            ValueError: If the improver is already registered.
        """
        name = name.lower()
        if name in self._improver_factories:
            logger.warning(f"Improver '{name}' already registered, overwriting")

        self._improver_factories[name] = factory
        logger.debug(f"Registered improver factory for '{name}'")

    def get_improver_factory(self, name: str) -> Optional[ImproverFactory]:
        """Get an improver factory.

        Args:
            name: The name of the improver.

        Returns:
            The factory function, or None if not found.
        """
        # Initialize improver components if needed
        self._initialize_component_type("improver")

        name = name.lower()
        factory = self._improver_factories.get(name)
        if factory is None:
            logger.debug(f"Improver factory for '{name}' not found")

        return factory

    def get_all_model_factories(self) -> Dict[str, ModelFactory]:
        """Get all registered model factories.

        Returns:
            Dictionary of model factories, keyed by provider name.
        """
        # Initialize model components if needed
        self._initialize_component_type("model")
        return dict(self._model_factories)

    def get_all_validator_factories(self) -> Dict[str, ValidatorFactory]:
        """Get all registered validator factories.

        Returns:
            Dictionary of validator factories, keyed by name.
        """
        # Initialize validator components if needed
        self._initialize_component_type("validator")
        return dict(self._validator_factories)

    def get_all_improver_factories(self) -> Dict[str, ImproverFactory]:
        """Get all registered improver factories.

        Returns:
            Dictionary of improver factories, keyed by name.
        """
        # Initialize improver components if needed
        self._initialize_component_type("improver")
        return dict(self._improver_factories)


# Singleton instance
_registry = Registry()


# Public API functions


def get_registry() -> Registry:
    """Get the registry instance.

    Returns:
        The registry instance.
    """
    return _registry


def register_model(provider: str) -> Callable[[ModelFactory], ModelFactory]:
    """Decorator for registering a model factory.

    Args:
        provider: The provider name (e.g., "openai", "anthropic").

    Returns:
        A decorator that registers the factory function.

    Example:
        ```python
        @register_model("custom")
        def create_custom_model(model_name, **options):
            return CustomModel(model_name, **options)
        ```
    """

    def decorator(factory: ModelFactory) -> ModelFactory:
        _registry.register_model(provider, factory)
        return factory

    return decorator


def get_model_factory(provider: str) -> Optional[ModelFactory]:
    """Get a model factory.

    Args:
        provider: The provider name (e.g., "openai", "anthropic").

    Returns:
        The factory function, or None if not found.
    """
    return _registry.get_model_factory(provider)


def register_validator(name: str) -> Callable[[ValidatorFactory], ValidatorFactory]:
    """Decorator for registering a validator factory.

    Args:
        name: The name of the validator.

    Returns:
        A decorator that registers the factory function.

    Example:
        ```python
        @register_validator("length")
        def create_length_validator(**options):
            return LengthValidator(**options)
        ```
    """

    def decorator(factory: ValidatorFactory) -> ValidatorFactory:
        _registry.register_validator(name, factory)
        return factory

    return decorator


def get_validator_factory(name: str) -> Optional[ValidatorFactory]:
    """Get a validator factory.

    Args:
        name: The name of the validator.

    Returns:
        The factory function, or None if not found.
    """
    return _registry.get_validator_factory(name)


def register_improver(name: str) -> Callable[[ImproverFactory], ImproverFactory]:
    """Decorator for registering an improver factory.

    Args:
        name: The name of the improver.

    Returns:
        A decorator that registers the factory function.

    Example:
        ```python
        @register_improver("clarity")
        def create_clarity_improver(model, **options):
            return ClarityImprover(model, **options)
        ```
    """

    def decorator(factory: ImproverFactory) -> ImproverFactory:
        _registry.register_improver(name, factory)
        return factory

    return decorator


def get_improver_factory(name: str) -> Optional[ImproverFactory]:
    """Get an improver factory.

    Args:
        name: The name of the improver.

    Returns:
        The factory function, or None if not found.
    """
    return _registry.get_improver_factory(name)


def initialize_registry() -> None:
    """Initialize the registry with default components.

    This function is called automatically when the package is imported.
    It ensures that the basic components are registered and available.

    Raises:
        ImportError: If critical modules cannot be imported.
    """
    # Initialize only the essential component types
    _registry._initialize_component_type("model")
    logger.debug("Registry initialized with default components")


def initialize_all() -> None:
    """Initialize all component types.

    This function imports all modules for all component types to ensure
    that all components are registered. It is useful for applications
    that want to preload all components.

    Raises:
        ImportError: If critical modules cannot be imported.
    """
    import time

    start_time = time.time()

    logger.debug("Initializing all component types")

    # Track initialization results
    component_types = list(_registry._lazy_imports.keys())
    successful_types = []
    failed_types = []

    # Initialize each component type
    for component_type in component_types:
        try:
            _registry._initialize_component_type(component_type)
            successful_types.append(component_type)
        except Exception as e:
            failed_types.append((component_type, str(e)))
            logger.error(f"Failed to initialize component type '{component_type}': {str(e)}")

    # Calculate processing time
    processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds

    # Log summary
    if failed_types:
        failed_type_names = [t[0] for t in failed_types]
        logger.warning(
            f"Initialized {len(successful_types)}/{len(component_types)} component types "
            f"in {processing_time:.2f}ms. Failed types: {failed_type_names}"
        )
    else:
        logger.debug(
            f"Successfully initialized all {len(component_types)} component types "
            f"in {processing_time:.2f}ms"
        )
